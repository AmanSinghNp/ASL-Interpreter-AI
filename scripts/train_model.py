"""
ASL Model Training Script
Trains a feed-forward neural network to classify ASL letters from hand landmarks
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras
from tensorflow.keras import layers
import os

# Configuration
CSV_FILE = 'asl_data.csv'
MODEL_DIR = '../public/models'
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# ASL letters (must match data_collection.py)
ASL_LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

def load_data():
    """Load and preprocess data from CSV"""
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"Data file '{CSV_FILE}' not found. "
            "Please run data_collection.py first to collect data."
        )
    
    # Read CSV
    df = pd.read_csv(CSV_FILE)
    
    # Extract features (42 columns: 21 x-coords, 21 y-coords)
    feature_columns = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
    X = df[feature_columns].values
    
    # Extract labels
    y = df['letter'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Loaded {len(X)} samples")
    print(f"Number of classes: {len(ASL_LETTERS)}")
    print(f"Features per sample: {X.shape[1]}")
    
    return X, y_encoded, label_encoder

def create_model(input_dim, num_classes):
    """Create feed-forward neural network model"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,), name='dense_1'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', name='dense_2'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', name='dense_3')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Main training function"""
    print("="*50)
    print("ASL Model Training")
    print("="*50)
    
    # Load data
    print("\n1. Loading data...")
    X, y, label_encoder = load_data()
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    print("\n3. Creating model...")
    model = create_model(input_dim=X.shape[1], num_classes=len(ASL_LETTERS))
    model.summary()
    
    # Train model
    print("\n4. Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
    )
    
    # Evaluate model
    print("\n5. Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model in TensorFlow format
    print("\n6. Saving model...")
    os.makedirs('saved_model', exist_ok=True)
    model.save('saved_model/asl_model')
    print("Model saved to: saved_model/asl_model/")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"\nTo use the model in the desktop app:")
    print(f"Run: python asl_app.py")
    print("="*50)

if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    train_model()


