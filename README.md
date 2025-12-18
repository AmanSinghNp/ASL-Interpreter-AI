# ASL Interpreter AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/tensorflow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A real-time American Sign Language (ASL) recognition application powered by Computer Vision and Deep Learning. This desktop application detects hand gestures via webcam and translates them into text instantly using MediaPipe and TensorFlow.

<!-- Add a demo GIF here: ![Demo](docs/demo.gif) -->

## Features

- **Real-Time Detection** - Instant feedback using MediaPipe's efficient hand tracking
- **High Accuracy** - Custom-trained neural network for classifying ASL alphabets (A-Y, excluding dynamic gestures J/Z)
- **Visual Feedback** - Confidence meter, FPS counter, and letter confirmation animations
- **Offline Capability** - Fully functional without an internet connection after initial setup
- **User-Friendly Interface** - Clean OpenCV-based GUI with word construction features
- **Extensible** - Includes scripts for data collection and retraining the model on your own dataset

## Quick Start

### Prerequisites

- Python 3.8 or higher
- A working webcam
- pip (Python package manager)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ASL-Interpreter-AI.git
   cd ASL-Interpreter-AI
   ```

2. **Create a Virtual Environment** (recommended)

   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
python asl_app.py
```

## Controls

| Key         | Action                          |
| :---------- | :------------------------------ |
| **Space**   | Add a space to the current word |
| **Backspace** | Delete the last character     |
| **C**       | Clear the current word          |
| **Q**       | Quit the application            |

> **Tip:** Ensure your hand is clearly visible to the camera with good lighting for best results.

## Training a Custom Model

The model and training data are **generated artifacts** and are not intended to be committed to Git. You can train your own model to improve accuracy or customize for your needs.

### Option 1: Collect Your Own Data

Use the interactive data collection tool to record hand landmarks:

```bash
python -m scripts.data_collection
```

Follow the on-screen instructions to record samples for each ASL letter.

### Option 2: Process an Existing Dataset

If you have an ASL image dataset organized in folders by letter:

```bash
python -m scripts.process_dataset --dataset_dir path/to/your/dataset
```

### Train the Model

Once you have data in `asl_data.csv`, train the neural network:

```bash
python -m scripts.train_model
```

**Training Options:**

```bash
python -m scripts.train_model --epochs 100 --batch_size 64
```

The trained model will be saved to `saved_model/asl_model/`.
The label list will be saved to `saved_model/classes.txt` (used by the app for correct index→label mapping).

## Project Structure

```
ASL-Interpreter-AI/
├── asl_app.py              # Main desktop application
├── utils.py                # Shared utilities and configuration
├── asl_data.csv            # Training data (generated)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── saved_model/            # Generated TensorFlow model + classes.txt (not committed)
│   └── asl_model/
└── scripts/                # Utility scripts
    ├── data_collection.py  # Interactive data gathering tool
    ├── process_dataset.py  # Batch image processing
    ├── train_model.py      # Model training script
    └── README.md           # Scripts documentation
```

## Technologies

| Technology | Purpose |
| :--------- | :------ |
| [MediaPipe](https://developers.google.com/mediapipe) | Hand landmark detection |
| [TensorFlow](https://www.tensorflow.org/) | Neural network training and inference |
| [OpenCV](https://opencv.org/) | Image processing and GUI |
| [scikit-learn](https://scikit-learn.org/) | Data preprocessing and metrics |

## Model Architecture

The classifier is a feed-forward neural network:

- **Input:** 42 features (21 x-coordinates + 21 y-coordinates, normalized)
- **Hidden Layers:** Dense(128) → Dense(64) → Dense(32) with BatchNorm and Dropout
- **Output:** Softmax over 24 classes (A-Y, excluding J and Z)

## Troubleshooting

### Camera Issues

**Problem:** "Could not open webcam" error

**Solutions:**
- Ensure your webcam is connected and not in use by another application
- Try a different camera by changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `asl_app.py`
- Check camera permissions in your system settings

### Model Issues

**Problem:** "Error loading model" message

**Solutions:**
- Ensure the model exists at `saved_model/asl_model/`
- Run `python -m scripts.train_model` to train a new model
- Check that TensorFlow is properly installed: `python -c "import tensorflow as tf; print(tf.__version__)"`

### Low Accuracy

**Solutions:**
- Ensure good lighting conditions
- Keep your hand clearly visible and centered in frame
- Try retraining with more diverse data
- Adjust `CONFIDENCE_THRESHOLD` in `utils.py` (default: 0.7)

### Performance Issues

**Problem:** Low FPS or laggy response

**Solutions:**
- Close other applications using the webcam
- Reduce webcam resolution if supported
- Ensure you're not running on a very low-powered machine

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Ideas for Contributions

- Add support for dynamic gestures (J, Z)
- Implement text-to-speech output
- Add word prediction/autocomplete
- Create a web-based version
- Improve the UI/UX design

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ASL alphabet images from various open datasets
- MediaPipe team for the excellent hand tracking solution
- TensorFlow team for the machine learning framework
