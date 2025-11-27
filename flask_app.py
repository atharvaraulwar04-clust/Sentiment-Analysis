from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback
import pickle
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json
from datetime import datetime
import threading
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
model = None
tokenizer = None
analyzer = None
MAX_LEN = 100
is_training = False
training_progress = {
    'status': 'idle',
    'epoch': 0,
    'total_epochs': 0,
    'accuracy': 0,
    'loss': 0,
    'val_accuracy': 0,
    'val_loss': 0
}

class TrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global training_progress
        training_progress.update({
            'epoch': epoch + 1,
            'accuracy': logs.get('accuracy', 0),
            'loss': logs.get('loss', 0),
            'val_accuracy': logs.get('val_accuracy', 0),
            'val_loss': logs.get('val_loss', 0)
        })
        # Emit progress via SocketIO
        socketio.emit('training_progress', training_progress)
        time.sleep(0.1)  # Small delay to ensure frontend updates

class SentimentAnalyzer:
    def __init__(self, model, tokenizer, max_len=100):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict_sentiment(self, review_text):
        cleaned = self.preprocess_text(review_text)
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')
        
        predictions = self.model.predict(padded, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        sentiment_emojis = ['😞', '😐', '😊']
        
        if predicted_class == 0:  # Negative
            predicted_rating = 2 if confidence > 0.7 else 1
        elif predicted_class == 1:  # Neutral
            predicted_rating = 3
        else:  # Positive
            predicted_rating = 5 if confidence > 0.7 else 4
        
        return {
            'review': review_text,
            'cleaned_review': cleaned,
            'predicted_sentiment': sentiment_labels[predicted_class],
            'emoji': sentiment_emojis[predicted_class],
            'confidence': float(confidence),
            'predicted_rating': predicted_rating,
            'probabilities': {
                'Negative': float(predictions[0]),
                'Neutral': float(predictions[1]),
                'Positive': float(predictions[2])
            }
        }

def load_models():
    """Load the trained model and tokenizer"""
    global model, tokenizer, analyzer
    try:
        if os.path.exists('sentiment_lstm_model.h5'):
            model = load_model('sentiment_lstm_model.h5')
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            analyzer = SentimentAnalyzer(model, tokenizer)
            socketio.emit('notification', {
                'type': 'success',
                'message': '✅ Models loaded successfully!',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            return True
        else:
            socketio.emit('notification', {
                'type': 'warning',
                'message': '⚠️ No trained model found. Please train a model first.',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            return False
    except Exception as e:
        socketio.emit('notification', {
            'type': 'error',
            'message': f'❌ Error loading models: {str(e)}',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        return False

def prepare_udemy_data(df):
    """Prepare dataset for training"""
    df = df.copy()
    df = df.dropna(subset=['Review'])
    
    # Handle different column names for ratings
    rating_column = None
    for col in ['Label', 'Rating', 'label', 'rating']:
        if col in df.columns:
            rating_column = col
            break
    
    if rating_column is None:
        raise ValueError("No rating column found in dataset. Expected 'Label' or 'Rating'")
    
    df[rating_column] = pd.to_numeric(df[rating_column], errors='coerce')
    df = df.dropna(subset=[rating_column])
    df = df[df[rating_column].between(1, 5)]
    df['sentiment'] = df[rating_column].apply(lambda x: 'Negative' if x <= 2 else 'Neutral' if x == 3 else 'Positive')
    df['Review'] = df['Review'].astype(str)
    df = df[df['Review'].str.len() >= 3]
    return df

def preprocess_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_data(df, max_words=5000, max_len=100):
    """Prepare data for model training"""
    df['cleaned_review'] = df['Review'].apply(preprocess_text)
    df = df[df['cleaned_review'].str.len() > 0]
    
    sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['sentiment_encoded'] = df['sentiment'].map(sentiment_mapping)
    
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['cleaned_review'])
    
    sequences = tokenizer.texts_to_sequences(df['cleaned_review'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return padded_sequences, df['sentiment_encoded'].values, tokenizer, df

def build_lstm_model(vocab_size, embedding_dim=128, max_len=100):
    """Build LSTM model architecture"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_training_visualizations(history, y_test, y_pred, df):
    """Create visualization plots"""
    # Training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    colors = ['#ff4444', '#ffaa44', '#44ff44']
    sentiment_counts.plot(kind='bar', color=colors, edgecolor='black')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
            autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Sentiment Distribution')
    
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Rating distribution
    rating_column = None
    for col in ['Label', 'Rating', 'label', 'rating']:
        if col in df.columns:
            rating_column = col
            break
    
    if rating_column:
        rating_counts = df[rating_column].value_counts().sort_index()
        plt.figure(figsize=(8, 5))
        colors = ['#ff4444', '#ff8844', '#ffaa44', '#88ff44', '#44ff44']
        plt.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='black')
        plt.title('Rating Distribution (1-5 Stars)')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

def train_model_thread(csv_filepath, epochs=20):
    """Train model in a separate thread"""
    global is_training, model, tokenizer, analyzer, training_progress
    
    try:
        is_training = True
        training_progress.update({
            'status': 'loading_data',
            'epoch': 0,
            'total_epochs': epochs
        })
        socketio.emit('training_progress', training_progress)
        
        # Load and prepare data
        socketio.emit('notification', {
            'type': 'info',
            'message': '📥 Loading dataset...',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
        df = pd.read_csv(csv_filepath)
        df = prepare_udemy_data(df)
        
        if len(df) < 50:
            socketio.emit('notification', {
                'type': 'error',
                'message': '❌ Dataset too small for training (min 50 reviews required)',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            is_training = False
            return
        
        # Preprocess data
        socketio.emit('notification', {
            'type': 'info',
            'message': '🔄 Preprocessing data...',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
        MAX_WORDS = 5000
        X, y, tokenizer, df_processed = prepare_data(df, max_words=MAX_WORDS, max_len=MAX_LEN)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Build and train model
        socketio.emit('notification', {
            'type': 'info',
            'message': '🏗️ Building model architecture...',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
        vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
        model = build_lstm_model(vocab_size, max_len=MAX_LEN)
        
        training_progress['status'] = 'training'
        socketio.emit('training_progress', training_progress)
        
        socketio.emit('notification', {
            'type': 'info',
            'message': '🚀 Starting model training...',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
        # Train model with progress callback
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[TrainingCallback()],
            verbose=0
        )
        
        # Evaluate model
        socketio.emit('notification', {
            'type': 'info',
            'message': '📊 Evaluating model performance...',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        
        # Save model and tokenizer
        model.save('sentiment_lstm_model.h5')
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        
        # Create visualizations
        create_training_visualizations(history, y_test, y_pred, df)
        
        analyzer = SentimentAnalyzer(model, tokenizer)
        
        training_progress['status'] = 'completed'
        socketio.emit('training_progress', training_progress)
        
        socketio.emit('notification', {
            'type': 'success',
            'message': f'✅ Training completed! Test accuracy: {test_accuracy:.2%}',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
        # Force reload of models and files
        socketio.emit('reload_required', {})
        
    except Exception as e:
        training_progress['status'] = 'error'
        socketio.emit('training_progress', training_progress)
        socketio.emit('notification', {
            'type': 'error',
            'message': f'❌ Training failed: {str(e)}',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    finally:
        is_training = False

def get_generated_files():
    """Get list of generated files"""
    files = []
    file_info = {
        'sentiment_lstm_model.h5': 'Trained LSTM Model',
        'tokenizer.pkl': 'Text Tokenizer',
        'training_history.png': 'Training History Plot',
        'confusion_matrix.png': 'Confusion Matrix',
        'sentiment_distribution.png': 'Sentiment Distribution',
        'rating_distribution.png': 'Rating Distribution'
    }
    
    for filename, description in file_info.items():
        if os.path.exists(filename):
            file_stats = os.stat(filename)
            files.append({
                'name': filename,
                'description': description,
                'size': f"{file_stats.st_size / 1024:.1f} KB",
                'modified': datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    return files

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('training_status', {'is_training': is_training, 'progress': training_progress})
    load_models()

@socketio.on('start_training')
def handle_start_training(data):
    """Start model training"""
    global is_training
    
    if is_training:
        emit('notification', {
            'type': 'warning',
            'message': '⚠️ Training already in progress',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        return
    
    # Handle file upload for training
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        filename = f"temp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file.save(filename)
        csv_file = filename
    else:
        csv_file = data.get('csv_file', 'udemy.csv')
    
    epochs = data.get('epochs', 20)
    
    # Start training in separate thread
    thread = threading.Thread(target=train_model_thread, args=(csv_file, epochs))
    thread.daemon = True
    thread.start()
    
    emit('notification', {
        'type': 'info',
        'message': '🎯 Training started in background...',
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })

# Flask Routes
@app.route('/')
def home():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze single review"""
    if analyzer is None:
        return jsonify({'error': 'Models not loaded. Please train a model first.'}), 400
    
    data = request.get_json()
    review_text = data.get('review', '')
    
    if not review_text.strip():
        return jsonify({'error': 'No review text provided'}), 400
    
    try:
        result = analyzer.predict_sentiment(review_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple reviews with progress"""
    if analyzer is None:
        return jsonify({'error': 'Models not loaded. Please train a model first.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read CSV file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        # Check for review column (case insensitive)
        review_column = None
        for col in df.columns:
            if col.lower() == 'review':
                review_column = col
                break
        
        if review_column is None:
            return jsonify({'error': 'CSV must contain a "Review" column'}), 400
        
        # Analyze with progress
        results = []
        total_reviews = len(df)
        
        for i, review in enumerate(df[review_column]):
            if pd.notna(review) and str(review).strip():
                result = analyzer.predict_sentiment(str(review))
                results.append(result)
                
                # Emit progress every 10 reviews
                if i % 10 == 0 or i == total_reviews - 1:
                    progress = (i + 1) / total_reviews * 100
                    socketio.emit('batch_progress', {
                        'current': i + 1,
                        'total': total_reviews,
                        'progress': progress
                    })
        
        # Create summary
        sentiment_counts = pd.Series([r['predicted_sentiment'] for r in results]).value_counts()
        rating_counts = pd.Series([r['predicted_rating'] for r in results]).value_counts().sort_index()
        
        summary = {
            'total_reviews': len(results),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'rating_distribution': rating_counts.to_dict(),
            'average_confidence': float(np.mean([r['confidence'] for r in results]))
        }
        
        socketio.emit('batch_complete', {})
        
        return jsonify({
            'summary': summary,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo_reviews')
def demo_reviews():
    """Get sample demo reviews for testing"""
    samples = [
        {
            'text': "Absolutely phenomenal course! The instructor explains complex concepts with such clarity and patience. The real-world projects were incredibly valuable.",
            'sentiment': 'Positive',
            'rating': 5
        },
        {
            'text': "The course content is decent but could use some improvements. Some sections felt rushed while others were too basic.",
            'sentiment': 'Neutral', 
            'rating': 3
        },
        {
            'text': "Extremely disappointing! The course content is completely outdated and the code examples don't work with current versions.",
            'sentiment': 'Negative',
            'rating': 1
        },
        {
            'text': "Outstanding quality! The production value is professional, the concepts are explained clearly, and the pacing is perfect.",
            'sentiment': 'Positive',
            'rating': 5
        },
        {
            'text': "Mixed feelings about this one. The theoretical parts are well-explained but the practical examples are somewhat outdated.",
            'sentiment': 'Neutral',
            'rating': 3
        }
    ]
    return jsonify(samples)

@app.route('/view_csv', methods=['POST'])
def view_csv():
    """View and analyze uploaded CSV file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read CSV file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        # Get basic info about the CSV
        csv_info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'preview': df.head(10).replace({np.nan: None}).to_dict('records'),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return jsonify(csv_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_file/<filename>')
def get_file(filename):
    """Serve generated files"""
    safe_filename = os.path.basename(filename)
    if os.path.exists(safe_filename):
        return send_file(safe_filename, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/system_status')
def system_status():
    """Get system status"""
    status = {
        'models_loaded': model is not None,
        'is_training': is_training,
        'training_progress': training_progress,
        'generated_files': get_generated_files(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add visualizations status
    visualizations = {}
    viz_files = ['training_history.png', 'confusion_matrix.png', 'sentiment_distribution.png', 'rating_distribution.png']
    for viz_file in viz_files:
        visualizations[viz_file] = os.path.exists(viz_file)
    
    status['visualizations'] = visualizations
    
    return jsonify(status)

if __name__ == '__main__':
    # Load existing models on startup
    load_models()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)