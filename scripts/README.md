# ASL Model Training Scripts

This directory contains Python scripts for training your own ASL recognition model.

## Quick Start

```bash
# From the project root directory:

# Option 1: Collect data interactively
python -m scripts.data_collection

# Option 2: Process an existing image dataset
python -m scripts.process_dataset

# Train the model
python -m scripts.train_model
```

## Scripts Overview

### 1. `data_collection.py` - Interactive Data Collection

Record hand landmarks for each ASL letter using your webcam.

```bash
python -m scripts.data_collection
```

**Controls:**
- Press a letter key (A-Y, excluding J/Z) to start recording that letter
- Show the ASL sign to the camera
- Samples are recorded automatically when a hand is detected
- Press `R` to reset/deselect the current letter
- Press `Q` to quit

**Output:** Appends data to `asl_data.csv`

### 2. `process_dataset.py` - Batch Image Processing

Convert a folder of ASL images into training data.

```bash
python -m scripts.process_dataset
python -m scripts.process_dataset --dataset_dir path/to/images --max_samples 500
```

**Arguments:**
| Argument | Default | Description |
| :------- | :------ | :---------- |
| `--dataset_dir` | `dataset/asl_alphabet_train` | Path to image folders |
| `--output` | `asl_data.csv` | Output CSV file |
| `--max_samples` | `1000` | Max images per letter |

**Expected Folder Structure:**
```
dataset/asl_alphabet_train/
├── A/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── B/
│   └── ...
└── ...
```

### 3. `train_model.py` - Model Training

Train the neural network classifier.

```bash
python -m scripts.train_model
python -m scripts.train_model --epochs 100 --batch_size 64
```

**Arguments:**
| Argument | Default | Description |
| :------- | :------ | :---------- |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `32` | Training batch size |
| `--validation_split` | `0.2` | Fraction of data for validation |
| `--no_plots` | `false` | Skip generating visualization plots |

**Output:**
- `saved_model/asl_model/` - TensorFlow SavedModel
- `saved_model/classes.txt` - List of class labels
- `training_history.png` - Accuracy/loss plots
- `confusion_matrix.png` - Per-class performance visualization

## Model Architecture

```
Input (42 features)
    ↓
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(64) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(32) + ReLU + Dropout(0.2)
    ↓
Dense(24) + Softmax
    ↓
Output (24 classes: A-Y excluding J, Z)
```

## Tips for Better Results

### Data Collection

1. **Vary hand position:** Move your hand around the frame (left, right, center)
2. **Vary distance:** Near and far from the camera
3. **Vary lighting:** Different lighting conditions
4. **Multiple people:** Have different people contribute data

### Sample Counts

| Quality Level | Samples per Letter |
| :------------ | :----------------- |
| Minimum | 50 |
| Recommended | 100-200 |
| Ideal | 500+ |

### Training Tips

- Monitor validation loss - if it starts increasing while training loss decreases, you're overfitting
- Early stopping is enabled by default (patience=10 epochs)
- Learning rate reduction is automatic when validation loss plateaus

## Troubleshooting

### "Data file not found"

Run data collection or dataset processing first:
```bash
python -m scripts.data_collection
# or
python -m scripts.process_dataset
```

### Low Accuracy

- Collect more diverse training data
- Ensure consistent, clear hand signs
- Try collecting data from multiple people
- Increase training epochs

### Camera Not Working

- Check camera permissions
- Try a different camera index: edit `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`
- Ensure no other application is using the camera

### MediaPipe Not Detecting Hands

- Improve lighting conditions
- Ensure hand is fully visible in frame
- Try a plain background
- Move hand closer to camera

## File Reference

| File | Input | Output |
| :--- | :---- | :----- |
| `data_collection.py` | Webcam | `asl_data.csv` |
| `process_dataset.py` | Image folders | `asl_data.csv` |
| `train_model.py` | `asl_data.csv` | `saved_model/asl_model/` |

## Notes

- Letters **J** and **Z** are excluded because they require motion/gesture recognition
- The model only supports static hand signs
- For production use, collect data from multiple signers for better generalization
- All scripts can be run as modules from the project root using `python -m scripts.<script_name>`
