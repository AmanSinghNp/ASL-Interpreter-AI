# ASL Interpreter AI

A real-time American Sign Language (ASL) recognition application powered by Computer Vision and Deep Learning. This desktop application detects hand gestures via webcam and translates them into text instantly using MediaPipe and TensorFlow.

## Features

- **Real-Time Detection:** Instant feedback using MediaPipe's efficient hand tracking.
- **High Accuracy:** Custom-trained neural network for classifying ASL alphabets (A-Y, excluding dynamic gestures J/Z).
- **Offline Capability:** Fully functional without an internet connection after initial setup.
- **User-Friendly Interface:** Simple OpenCV-based GUI with word construction features.
- **Extensible:** Includes scripts for data collection and retraining the model on your own dataset.

## Installation

### Prerequisites

- Python 3.8 or higher
- A working webcam

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ASL-Interpreter-AI.git
   cd ASL-Interpreter-AI
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

To start the ASL Interpreter, run the main application script:

```bash
python asl_app.py
```

### Controls

| Key           | Action                          |
| :------------ | :------------------------------ |
| **Space**     | Add a space to the current word |
| **Backspace** | Delete the last character       |
| **C**         | Clear the current word          |
| **Q**         | Quit the application            |

_Ensure your hand is clearly visible to the camera for best results._

## Training a Custom Model

This repository includes a pre-trained model (`saved_model/`), but you can train your own to improve accuracy or add new signs.

### 1. Collect Data

Use the data collection script to record landmarks for each letter.

```bash
python scripts/data_collection.py
```

_Follow the on-screen instructions to record samples for each sign._

### 2. Process Data

Convert the raw landmarks into a dataset suitable for training.

```bash
python scripts/process_dataset.py
```

_This generates `asl_data.csv`._

### 3. Train the Model

Train the neural network using the processed data.

```bash
python scripts/train_model.py
```

_The new model will automatically replace the one in `saved_model/`._

## Project Structure

```
ASL-Interpreter-AI/
├── asl_app.py              # Main desktop application
├── asl_data.csv            # Dataset containing normalized landmarks
├── requirements.txt        # Python dependencies
├── saved_model/            # Pre-trained TensorFlow model
└── scripts/                # Utility scripts
    ├── data_collection.py  # Data gathering tool
    ├── process_dataset.py  # Data preprocessing
    └── train_model.py      # Model training script
```

## Technologies

- **[MediaPipe](https://developers.google.com/mediapipe):** For robust hand landmark detection.
- **[TensorFlow](https://www.tensorflow.org/):** For building and running the neural network classifier.
- **[OpenCV](https://opencv.org/):** For image processing and the graphical user interface.

## License

This project is licensed under the MIT License.
