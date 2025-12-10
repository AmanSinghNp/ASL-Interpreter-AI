# ASL Model Training Scripts

This directory contains Python scripts for training your own ASL recognition model.

## Prerequisites

Install the required Python packages:

```bash
pip install opencv-python mediapipe pandas scikit-learn tensorflow tensorflowjs
```

## Usage

### Step 1: Collect Data

Run the data collection script to record hand landmarks for each ASL letter:

```bash
python data_collection.py
```

**Instructions:**
- Press a letter key (A-Y, excluding J/Z) to start recording
- Show that ASL letter sign to the camera
- The script automatically records 100 samples per letter
- Press 'q' to quit
- Press 'r' to reset current letter

The script will save data to `asl_data.csv` with:
- 21 x-coordinates (normalized)
- 21 y-coordinates (normalized)
- Letter label

**Recommended:** Collect at least 100 samples per letter for good model performance.

### Step 2: Train Model

Once you have collected data, train the neural network:

```bash
python train_model.py
```

This script will:
1. Load and preprocess the collected data
2. Create a feed-forward neural network (128 → 64 → 24 neurons)
3. Train the model with validation split
4. Evaluate performance
5. Save the model in TensorFlow format
6. Convert to TensorFlow.js format

### Step 3: Deploy Model

The trained model will be converted to TensorFlow.js format and saved to:
```
../public/models/
```

**Files generated:**
- `model.json` - Model architecture
- `weights.bin` - Model weights (or multiple .bin files)

Copy these files to your Next.js `public/models/` directory, replacing the placeholder files.

## Model Architecture

The model is a simple feed-forward neural network:
- **Input:** 42 features (21 x-coords, 21 y-coords)
- **Layer 1:** Dense(128) + ReLU + Dropout(0.3)
- **Layer 2:** Dense(64) + ReLU + Dropout(0.3)
- **Output:** Dense(24) + Softmax (one for each letter)

## Tips for Better Results

1. **Collect diverse data:**
   - Vary hand position (left, right, center)
   - Vary distance from camera
   - Vary lighting conditions
   - Have multiple people collect data

2. **More samples = better accuracy:**
   - Minimum: 50 samples per letter
   - Recommended: 100+ samples per letter
   - Ideal: 200+ samples per letter

3. **Ensure good data quality:**
   - Make sure hand is fully visible
   - Avoid blurry images
   - Use consistent signing style

4. **Training parameters:**
   - Adjust `EPOCHS` in `train_model.py` if needed
   - Monitor validation loss to prevent overfitting
   - Early stopping is enabled by default

## Troubleshooting

**Problem:** "Data file not found"
- Solution: Run `data_collection.py` first to collect data

**Problem:** "TensorFlow.js conversion failed"
- Solution: Install tensorflowjs manually: `pip install tensorflowjs`
- Then run conversion manually: `tensorflowjs_converter --input_format=tf_saved_model --output_dir=../public/models saved_model/asl_model`

**Problem:** Low accuracy
- Solution: Collect more diverse training data
- Try collecting data from multiple signers
- Ensure consistent hand positioning

**Problem:** Camera not working
- Solution: Make sure camera permissions are granted
- Try a different camera device index in `cv2.VideoCapture(0)`

## File Structure

```
scripts/
├── data_collection.py    # Collect hand landmark data
├── train_model.py        # Train the neural network
├── README.md            # This file
└── asl_data.csv         # Generated data file (after collection)
```

## Notes

- Letters J and Z are excluded because they require motion/gesture recognition
- The model only supports static hand signs
- For production use, consider collecting data from multiple signers for better generalization


