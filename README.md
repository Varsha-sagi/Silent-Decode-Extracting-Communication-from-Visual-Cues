# Silent-Decode-Extracting-Communication-from-Visual-Cues

This project is a comprehensive LipSync system focused on extracting communication from visual cues like lip movements.
It applies deep learning models for lip reading, translation, and analysis, enabling seamless speech-to-text without audio input.

# Features

**- Lip Reading:** Detects and interprets speech from video frames of lip movements.
**- Translation:** Converts recognized lip-read text into multiple languages.
**- Lie Detection:** Identifies inconsistencies in spoken vs. intended communication.
**- Video Processing:** Uses CNN and LSTM for feature extraction and sequence modeling.
**- Web Interface:** Flask-based interface for uploading videos and viewing results.

# How to Use

- [] Clone the repository to your local machine.
- [] Install the required dependencies using pip install -r requirements.txt.
- [] Set up the deep learning models (pre-trained weights or training from scratch).
- [] Run the Flask application with python app.py.
- [] Open http://localhost:5000 in your browser.
- [] Upload a video and view the lip reading, translation, and analysis results.

# Requirements
- Python 3.8+
- TensorFlow / PyTorch
- OpenCV
- Flask
- Other dependencies (see requirements.txt)
