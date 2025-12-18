# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2024-12-18

### Added
- Shared utilities module (`utils.py`) for centralized configuration
- FPS counter display in main application
- Confidence bar visualization
- Letter confirmation flash animation
- Visual feedback when letters are added to word
- Command-line arguments for training script (--epochs, --batch_size, etc.)
- Confusion matrix visualization after training
- Classification report output during training
- Batch normalization layers in neural network
- Progress indicators in data collection and processing scripts
- Comprehensive troubleshooting section in README
- Contributing guidelines in README
- MIT LICENSE file
- CHANGELOG.md for version tracking
- requirements-dev.txt for development dependencies
- Unit tests for utility functions

### Changed
- Refactored all scripts to use shared `utils.py` module
- Improved UI overlay design in main application
- Enhanced training script with better logging and visualizations
- Updated data collection UI with progress bars and sample counts
- Improved dataset processing with better error handling
- Updated requirements.txt for cross-platform compatibility
- Reorganized README.md with better structure and more details
- Updated scripts/README.md with accurate information

### Fixed
- Typo: "Backsapce" → "Backspace" in console output
- Removed platform-specific `tensorflow-intel` dependency
- Removed unused `MODEL_DIR` variable in training script
- Fixed import paths for running scripts as modules

### Removed
- Duplicate `normalize_landmarks()` function definitions
- Duplicate `ASL_LETTERS` constant definitions
- References to TensorFlow.js conversion (not implemented)

## [1.0.0] - 2024-01-01

### Added
- Initial release
- Real-time ASL alphabet recognition (A-Y, excluding J and Z)
- MediaPipe hand landmark detection
- TensorFlow neural network classifier
- OpenCV-based desktop application
- Data collection script
- Dataset processing script
- Model training script
- Pre-trained model
- Basic documentation

---

## Version History Summary

| Version | Date | Highlights |
| :------ | :--- | :--------- |
| 1.1.0 | 2024-12-18 | Code refactoring, UI improvements, better documentation |
| 1.0.0 | 2024-01-01 | Initial release |

