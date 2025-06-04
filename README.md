# Voice-based Gender Recognition

## Overview
This project implements a voice-based gender recognition system using multiple machine learning models (SVM, Random Forest, and CNN). The system analyzes audio features to predict the speaker's gender, providing a modern GUI interface for easy interaction.

## Features
- Multiple model support (SVM, Random Forest, CNN)
- Interactive GUI for training and prediction
- Real-time analysis and visualization
- Batch prediction capabilities
- Performance analysis and visualization tools

## Prerequisites
- Python 3.8 or higher
- Required Python packages (install using `pip install -r requirements.txt`):
  - tkinter
  - PIL (Pillow)
  - matplotlib
  - seaborn
  - numpy
  - scikit-learn
  - torch
  - librosa
  - pandas

## Installation
1. Clone the repository:

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset Setup
This project requires specific audio datasets that are not included in the repository due to size constraints. Follow these steps to set up the datasets:

### Training Dataset
1. Download the Mozilla Common Voice dataset:
   - Visit [Common Voice Dataset](https://commonvoice.mozilla.org/en/datasets)
   - Create an account if needed
   - Download the English dataset (cv-corpus-x.x-YYYY-MM-DD)
   - Extract the downloaded file
   - Place the audio files in the `audio/train` directory

### Testing Dataset
1. Create a directory named `audio/test`
2. You can use:
   - A portion of the Common Voice dataset
   - Your own audio recordings (WAV format)
   - Other voice datasets like VoxCeleb

### Directory Structure
```
voice-based-Gender-Recognition/
├── audio/
│   ├── train/
│   │   └── [training audio files]
│   └── test/
│       └── [testing audio files]
├── saved_models/
├── figures/
├── app_ui.py
├── index.py
└── requirements.txt
```

## Usage
1. Start the application:
```bash
python app_ui.py
```

2. Using the GUI:
   - **Train Models**: Select models to train and specify post-training evaluation directory
   - **Predict from Folder**: Choose a folder with audio files for gender prediction
   - **Show Analysis**: View performance metrics and visualizations

## Model Information
- **SVM**: Uses extracted audio features for classification
- **Random Forest**: Ensemble learning approach for robust prediction
- **CNN**: Deep learning model analyzing spectrograms

## Contributing
Feel free to submit issues and enhancement requests!

## Author
Mohammad Tosifi


## License
This project is licensed under the MIT License - see the LICENSE file for details.