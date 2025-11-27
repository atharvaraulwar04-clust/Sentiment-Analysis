"""
Automated Sentiment Classification of Course Reviews using RNN
================================================================
A complete implementation using LSTM networks for sentiment analysis
of online course reviews (Coursera, Udemy, etc.)
Modified to work with actual Udemy dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ============================================================================
# STEP 1: LOAD AND PREPARE UDEMY DATASET
# ============================================================================

def load_udemy_dataset(filepath='udemy.csv'):
    """
    Load Udemy dataset from CSV file
    Expected columns: CourseId, Review, Label (1-5 rating)
    """
    print(f"\nLoading dataset from {filepath}...")
    
    try:
        # Try different encodings
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='latin-1')
        
        print(f"✓ Dataset loaded successfully!")
        print(f"✓ Total records: {len(df)}")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Display basic info
        print(f"\n📊 Dataset Overview:")
        print(df.head())
        
        return df
    
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found!")
        print("Please make sure the file is in the same directory as this script.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

def convert_rating_to_sentiment(rating):
    """
    Convert numeric rating (1-5) to sentiment label
    1-2: Negative
    3: Neutral
    4-5: Positive
    """
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'

def prepare_udemy_data(df):
    """
    Prepare Udemy dataset for training
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Remove any rows with missing reviews
    df = df.dropna(subset=['Review'])
    
    # Convert Label to numeric if it's not already
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
    
    # Remove rows with invalid labels
    df = df.dropna(subset=['Label'])
    df = df[df['Label'].between(1, 5)]
    
    # Convert rating to sentiment
    df['sentiment'] = df['Label'].apply(convert_rating_to_sentiment)
    
    # Clean reviews - convert to string and handle any issues
    df['Review'] = df['Review'].astype(str)
    
    # Remove very short reviews (less than 3 characters)
    df = df[df['Review'].str.len() >= 3]
    
    print(f"\n✓ Data preparation complete!")
    print(f"✓ Valid reviews: {len(df)}")
    print(f"\n📊 Sentiment Distribution:")
    print(df['sentiment'].value_counts())
    print(f"\n📊 Rating Distribution:")
    print(df['Label'].value_counts().sort_index())
    
    return df

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================

def preprocess_text(text):
    """
    Clean and preprocess text data
    """
    # Convert to string (in case of any non-string values)
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep punctuation that might indicate sentiment
    text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_data(df, max_words=5000, max_len=100):
    """
    Prepare data for RNN model
    """
    print("\n🔄 Preprocessing reviews...")
    
    # Preprocess reviews
    df['cleaned_review'] = df['Review'].apply(preprocess_text)
    
    # Remove any empty reviews after cleaning
    df = df[df['cleaned_review'].str.len() > 0]
    
    # Encode sentiments: Positive=2, Neutral=1, Negative=0
    sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['sentiment_encoded'] = df['sentiment'].map(sentiment_mapping)
    
    # Tokenization
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['cleaned_review'])
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['cleaned_review'])
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    print(f"✓ Preprocessing complete!")
    print(f"✓ Final dataset size: {len(df)}")
    
    return padded_sequences, df['sentiment_encoded'].values, tokenizer, df

# ============================================================================
# STEP 3: BUILD RNN MODEL (LSTM)
# ============================================================================

def build_lstm_model(vocab_size, embedding_dim=128, max_len=100):
    """
    Build a Bidirectional LSTM model for sentiment classification
    """
    model = Sequential([
        # Embedding layer
        Embedding(input_dim=vocab_size, 
                  output_dim=embedding_dim, 
                  input_length=max_len),
        
        # Bidirectional LSTM layers
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.4),
        
        Dense(32, activation='relu'),
        
        # Output layer (3 classes: Negative, Neutral, Positive)
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# STEP 4: TRAINING PIPELINE
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, vocab_size, max_len=100, epochs=20):
    """
    Train the LSTM model with callbacks
    """
    # Build model
    model = build_lstm_model(vocab_size, max_len=max_len)
    
    print("\n📐 Model Architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print("\n🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history

# ============================================================================
# STEP 5: EVALUATION AND VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """
    Visualize training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history plot saved as 'training_history.png'")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Neutral', 'Positive']):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()

def plot_sentiment_distribution(df):
    """
    Plot sentiment distribution in dataset
    """
    sentiment_counts = df['sentiment'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    colors = ['#ff4444', '#ffaa44', '#44ff44']
    sentiment_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
    axes[0].set_title('Sentiment Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie chart
    axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=colors, startangle=90,
                explode=(0.05, 0.05, 0.05), shadow=True)
    axes[1].set_title('Sentiment Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Sentiment distribution plot saved as 'sentiment_distribution.png'")
    plt.show()

def plot_rating_distribution(df):
    """
    Plot rating distribution (1-5 stars)
    """
    rating_counts = df['Label'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    colors = ['#ff4444', '#ff8844', '#ffaa44', '#88ff44', '#44ff44']
    bars = plt.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.title('Rating Distribution (1-5 Stars)', fontsize=14, fontweight='bold')
    plt.xlabel('Rating (Stars)')
    plt.ylabel('Number of Reviews')
    plt.xticks(rating_counts.index)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('rating_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Rating distribution plot saved as 'rating_distribution.png'")
    plt.show()

def display_sample_reviews(df, n=5):
    """
    Display sample reviews from each sentiment category
    """
    print("\n" + "="*80)
    print("SAMPLE REVIEWS FROM DATASET")
    print("="*80)
    
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        print(f"\n{'='*80}")
        print(f"📝 {sentiment.upper()} REVIEWS:")
        print(f"{'='*80}")
        
        samples = df[df['sentiment'] == sentiment].sample(min(n, len(df[df['sentiment'] == sentiment])))
        
        for idx, row in samples.iterrows():
            print(f"\n⭐ Rating: {int(row['Label'])} stars")
            print(f"📖 Review: {row['Review'][:200]}{'...' if len(row['Review']) > 200 else ''}")
            print("-" * 80)

# ============================================================================
# STEP 6: PREDICTION AND EXPLANATION
# ============================================================================

class SentimentAnalyzer:
    """
    Wrapper class for sentiment analysis with explanations
    """
    
    def __init__(self, model, tokenizer, max_len=100):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sentiment_labels = ['Negative', 'Neutral', 'Positive']
        self.sentiment_emojis = ['😞', '😐', '😊']
    
    def predict_sentiment(self, review_text):
        """
        Predict sentiment for a single review with confidence scores
        """
        # Preprocess
        cleaned = preprocess_text(review_text)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')
        
        # Predict
        predictions = self.model.predict(padded, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        # Predict star rating (1-5)
        # Negative: 1-2 stars, Neutral: 3 stars, Positive: 4-5 stars
        if predicted_class == 0:  # Negative
            predicted_rating = 2 if confidence > 0.7 else 1
        elif predicted_class == 1:  # Neutral
            predicted_rating = 3
        else:  # Positive
            predicted_rating = 5 if confidence > 0.7 else 4
        
        return {
            'review': review_text,
            'cleaned_review': cleaned,
            'predicted_sentiment': self.sentiment_labels[predicted_class],
            'emoji': self.sentiment_emojis[predicted_class],
            'confidence': float(confidence),
            'predicted_rating': predicted_rating,
            'probabilities': {
                'Negative': float(predictions[0]),
                'Neutral': float(predictions[1]),
                'Positive': float(predictions[2])
            }
        }
    
    def analyze_keywords(self, review_text, top_n=5):
        """
        Extract important keywords from the review
        """
        cleaned = preprocess_text(review_text)
        words = cleaned.split()
        
        # Sentiment-indicating words (expanded list)
        positive_words = ['excellent', 'great', 'amazing', 'best', 'love', 'perfect', 
                         'fantastic', 'outstanding', 'brilliant', 'wonderful', 'good',
                         'helpful', 'informative', 'comprehensive', 'rewarding', 'useful',
                         'recommend', 'nice', 'interesting', 'learned', 'easy']
        
        negative_words = ['terrible', 'worst', 'bad', 'poor', 'waste', 'disappointed',
                         'horrible', 'awful', 'boring', 'useless', 'confusing', 'hard',
                         'difficult', 'unclear', 'misleading', 'slow', 'outdated']
        
        neutral_words = ['okay', 'average', 'fine', 'decent', 'acceptable', 'satisfactory',
                        'basic', 'standard', 'reasonable', 'fair']
        
        keywords = {
            'positive': [w for w in words if w in positive_words],
            'negative': [w for w in words if w in negative_words],
            'neutral': [w for w in words if w in neutral_words]
        }
        
        return keywords
    
    def display_prediction(self, review_text):
        """
        Display formatted prediction results
        """
        result = self.predict_sentiment(review_text)
        keywords = self.analyze_keywords(review_text)
        
        print("\n" + "="*80)
        print("SENTIMENT ANALYSIS RESULT")
        print("="*80)
        print(f"\n📝 Original Review:")
        print(f"   {result['review']}")
        print(f"\n🔍 Cleaned Review:")
        print(f"   {result['cleaned_review']}")
        print(f"\n🎯 Predicted Sentiment: {result['predicted_sentiment']} {result['emoji']}")
        print(f"⭐ Predicted Rating: {result['predicted_rating']} / 5 stars")
        print(f"📊 Confidence Score: {result['confidence']:.2%}")
        print(f"\n📈 Probability Distribution:")
        for sentiment, prob in result['probabilities'].items():
            bar_length = int(prob * 50)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"   {sentiment:8s}: {bar} {prob:.2%}")
        
        print(f"\n🔑 Key Sentiment Indicators:")
        if keywords['positive']:
            print(f"   ✅ Positive words: {', '.join(keywords['positive'])}")
        if keywords['negative']:
            print(f"   ❌ Negative words: {', '.join(keywords['negative'])}")
        if keywords['neutral']:
            print(f"   ⚖️  Neutral words: {', '.join(keywords['neutral'])}")
        
        if not any(keywords.values()):
            print(f"   ℹ️  No explicit sentiment keywords detected")
        
        print("\n" + "="*80 + "\n")
        
        return result

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main(csv_filepath='udemy.csv'):
    """
    Complete pipeline for sentiment analysis with Udemy dataset
    """
    print("\n" + "="*80)
    print("COURSE REVIEW SENTIMENT ANALYSIS USING RNN (LSTM)")
    print("Working with Udemy Dataset")
    print("="*80 + "\n")
    
    # Step 1: Load Dataset
    print("Step 1: Loading Udemy dataset...")
    df = load_udemy_dataset(csv_filepath)
    
    if df is None:
        print("\n❌ Failed to load dataset. Please check the file path and try again.")
        return None, None, None
    
    # Step 2: Prepare Dataset
    print("\nStep 2: Preparing dataset...")
    df = prepare_udemy_data(df)
    
    if len(df) < 50:
        print("\n⚠️ Warning: Dataset is too small for effective training.")
        print("At least 50 reviews are recommended.")
        return None, None, None
    
    # Visualize distributions
    plot_sentiment_distribution(df)
    plot_rating_distribution(df)
    
    # Display sample reviews
    display_sample_reviews(df, n=3)
    
    # Step 3: Preprocess Data
    print("\nStep 3: Preprocessing data...")
    MAX_WORDS = 5000
    MAX_LEN = 100
    X, y, tokenizer, df_processed = prepare_data(df, max_words=MAX_WORDS, max_len=MAX_LEN)
    print(f"✓ Vocabulary size: {len(tokenizer.word_index) + 1}")
    print(f"✓ Sequence length: {MAX_LEN}")
    print(f"✓ Data shape: {X.shape}\n")
    
    # Step 4: Split Data
    print("Step 4: Splitting data...")
    
    # Check if we have enough data for each class
    min_class_size = df_processed['sentiment_encoded'].value_counts().min()
    
    if min_class_size < 2:
        print("⚠️ Warning: Some sentiment classes have very few samples.")
        print("Using simple train-test split without validation set.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_val, y_val = X_test, y_test  # Use test set as validation
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Validation samples: {len(X_val)}")
    print(f"✓ Test samples: {len(X_test)}\n")
    
    # Step 5: Train Model
    print("Step 5: Building and training LSTM model...")
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    model, history = train_model(X_train, y_train, X_val, y_val, vocab_size, max_len=MAX_LEN, epochs=25)
    
    # Step 6: Evaluate Model
    print("\n" + "="*80)
    print("Step 6: Evaluating model on test set...")
    print("="*80)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"✓ Test Loss: {test_loss:.4f}\n")
    
    # Predictions
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    # Classification report
    print("\n📊 Classification Report:")
    print("="*80)
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
    
    # Visualizations
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    
    # Step 7: Create Sentiment Analyzer
    print("\nStep 7: Creating sentiment analyzer...")
    analyzer = SentimentAnalyzer(model, tokenizer, max_len=MAX_LEN)
    print("✓ Sentiment analyzer ready\n")
    
    # Step 8: Test with actual reviews from dataset
    print("Step 8: Testing with actual reviews from your dataset...\n")
    
    # Get sample reviews from each category
    test_samples = []
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        samples = df_processed[df_processed['sentiment'] == sentiment].sample(
            min(2, len(df_processed[df_processed['sentiment'] == sentiment]))
        )
        test_samples.extend(samples['Review'].tolist())
    
    for review in test_samples:
        analyzer.display_prediction(review)
    
    # Step 9: Save model and tokenizer
    print("\nStep 9: Saving model and tokenizer...")
    model.save('sentiment_lstm_model.h5')
    
    # Save tokenizer
    import pickle
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("✓ Model saved as 'sentiment_lstm_model.h5'")
    print("✓ Tokenizer saved as 'tokenizer.pkl'\n")
    
    print("="*80)
    print("✅ SENTIMENT ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
    
    print("📁 Generated Files:")
    print("   • sentiment_lstm_model.h5 - Trained LSTM model")
    print("   • tokenizer.pkl - Text tokenizer")
    print("   • training_history.png - Training/validation curves")
    print("   • confusion_matrix.png - Model performance visualization")
    print("   • sentiment_distribution.png - Dataset sentiment distribution")
    print("   • rating_distribution.png - Dataset rating distribution")
    
    print("\n💡 Usage Examples:")
    print("   # Analyze a new review:")
    print("   analyzer.display_prediction('This course is amazing! I learned so much.')")
    print("\n   # Get prediction programmatically:")
    print("   result = analyzer.predict_sentiment('Great course!')")
    print("   print(result['predicted_sentiment'], result['predicted_rating'])")
    
    return model, tokenizer, analyzer

# ============================================================================
# RUN THE COMPLETE PIPELINE
# ============================================================================

if __name__ == "__main__":
    # Run with your Udemy CSV file
    model, tokenizer, analyzer = main('Udemy.csv')
    
    if analyzer is not None:
        print("\n" + "="*80)
        print("🎉 System is ready! You can now analyze custom reviews.")
        print("="*80)
        print("\nTry it out:")
        print(">>> analyzer.display_prediction('Your custom review here')")