"""
ASL Model Training Script
Trains a feed-forward neural network to classify ASL letters from hand landmarks.

Usage:
    python -m scripts.train_model
    python -m scripts.train_model --epochs 100 --batch_size 64
"""

import sys
import os
import argparse

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

from utils import ASL_LETTERS, CSV_FILE, CLASSES_PATH


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ASL recognition model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')
    return parser.parse_args()


def load_data(csv_file):
    """Load and preprocess data from CSV."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Data file '{csv_file}' not found. "
            "Please run data_collection.py or process_dataset.py first."
        )
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Extract features (42 columns: 21 x-coords, 21 y-coords)
    feature_columns = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
    X = df[feature_columns].values
    
    # Extract labels
    y = df['letter'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Loaded {len(X)} samples")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    return X, y_encoded, label_encoder


def create_model(input_dim, num_classes):
    """Create feed-forward neural network model."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', name='dense_1'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', name='dense_2'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', name='dense_3'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training history plot saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def train_model(args):
    """Main training function."""
    print("=" * 60)
    print("ASL Model Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Validation Split: {args.validation_split}")
    
    # Load data
    print("\n" + "-" * 40)
    print("Step 1: Loading data...")
    print("-" * 40)
    X, y, label_encoder = load_data(CSV_FILE)
    
    # Split data
    print("\n" + "-" * 40)
    print("Step 2: Splitting data...")
    print("-" * 40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.validation_split, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    print("\n" + "-" * 40)
    print("Step 3: Creating model...")
    print("-" * 40)
    num_classes = len(label_encoder.classes_)
    model = create_model(input_dim=X.shape[1], num_classes=num_classes)
    model.summary()
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'saved_model/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
    ]
    
    # Train model
    print("\n" + "-" * 40)
    print("Step 4: Training model...")
    print("-" * 40)
    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n" + "-" * 40)
    print("Step 5: Evaluating model...")
    print("-" * 40)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.1%})")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Generate predictions for metrics
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    # Classification report
    print("\n" + "-" * 40)
    print("Classification Report:")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Generate plots
    if not args.no_plots:
        print("\n" + "-" * 40)
        print("Step 6: Generating plots...")
        print("-" * 40)
        plot_training_history(history)
        plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)
    
    # Save model
    print("\n" + "-" * 40)
    print("Step 7: Saving model...")
    print("-" * 40)
    os.makedirs('saved_model', exist_ok=True)
    model.save('saved_model/asl_model')
    print("Model saved to: saved_model/asl_model/")
    
    # Save label encoder classes for reference (used by the app for correct mapping)
    with open(CLASSES_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(label_encoder.classes_))
    print(f"Classes saved to: {CLASSES_PATH}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  - Test Accuracy: {test_accuracy:.1%}")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"\nTo use the model:")
    print(f"  python asl_app.py")
    print("=" * 60)
    
    return model, history


if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    args = parse_args()
    train_model(args)
