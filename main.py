from imutils import face_utils
import imutils
import cv2
import numpy as np
import torch
import torch.nn as nn
from utils.tools.lipnet import LipReadingNetwork
from utils.tools.text_preprocess import ctc_convert_array_to_text
import dlib
import editdistance
from gtts import gTTS
from googletrans import Translator
import google.generativeai as genai
import os
from dotenv import load_dotenv

dict_labels = {
    'verb': ['BIN', 'LAY', 'PLACE', 'SET'],
    'colour': ['BLUE', 'GREEN', 'RED', 'WHITE'],
    'prep': ['AT', 'BY', 'IN', 'WITH'],
    'letter': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z'],
    'digit': ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE'],
    'adverb': ['AGAIN', 'NOW', 'PLEASE', 'SOON']
}

# Load facial landmark model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("utils/model/shape_predictor_68_face_landmarks.dat")  

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LipReadingNetwork().to(device)
net = nn.DataParallel(model).to(device)
pretrained_dict = torch.load("utils/model/lipread.pt", map_location=device,weights_only=True)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
load_dotenv()
api_key = os.getenv("API")
genai.configure(api_key=api_key)

def process_video(video_path):
    """
    Processes a video file, extracts mouth region, and predicts lip-synced text.
    """
    print("[INFO] Loading video:", video_path)
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame and convert to grayscale
        frame = imutils.resize(frame, width=1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[48:68]

            # Get bounding box for the mouth region
            (x, y, w, h) = cv2.boundingRect(np.array([mouth]))
            mouth_roi = frame[y-50:y+h+50, x-50:x+w+50]

            # Resize to match model input (128x64)
            mouth_roi = cv2.resize(mouth_roi, (128, 64), interpolation=cv2.INTER_CUBIC)
            frames.append(mouth_roi)

    cap.release()

    if len(frames) == 0:
        print("[ERROR] No faces detected!")
        return None

    # Convert frames to a tensor and reshape
    video_tensor = torch.FloatTensor(np.array(frames))
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, F, H, W)
    video_tensor = video_tensor.reshape(1, 3, len(frames), 64, 128).to(device)

    with torch.no_grad():
        output = net(video_tensor)
        pred_txt = ctc_decode(output)[0]
        #corrected_pred_text = correct_prediction(pred_txt)
        video_file = genai.upload_file(path=video_path)
        video_file = genai.get_file(video_file.name)
        prompt = """
                You are a expert lip reader. Your task is to tell me what the user is saying even though no audio is given. 
                Analyse frame by frame slowly. Track the mouth pattern and try to predict the sentence. No preamble only predicted text.
            """
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content([prompt, video_file],
                                            request_options={"timeout": 600})
        genai.delete_file(video_file.name)
    return response.text


def ctc_decode(y):
    """Decodes the output of the model."""
    y = y.argmax(-1)
    return [ctc_convert_array_to_text(y[_], start=1) for _ in range(y.size(0))]

def correct_prediction(prediction):
    """Corrects the prediction based on predefined dictionary labels."""
    new_pred = []
    list_pred_words = prediction.split(" ")
    idx_dict = 0
    for pred in list_pred_words:
        lowest_cer = float('inf')
        idx_lowest_label = 0
        for idx, label in enumerate(dict_labels[list(dict_labels.keys())[idx_dict]]):
            cer = editdistance.eval(pred, label)
            if cer < lowest_cer:
                idx_lowest_label = idx
                lowest_cer = cer
        new_pred.append(dict_labels[list(dict_labels.keys())[idx_dict]][idx_lowest_label])
        idx_dict += 1
    return ' '.join(new_pred)


def create_audio(predicted_text,language):
    tts_english = gTTS(text=predicted_text, lang=language)
    tts_english.save(f"static/assets/audio/{language}_audio.mp3")

async def translate(predicted_text,lang):
    # Translator
    async with Translator() as translator:
        result = await translator.translate(predicted_text, src="en", dest=lang)
        return result.text

def calculate_lie_confidence(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Store movements
    vertical_movements = []
    horizontal_movements = []
    lip_tension_values = []
    lip_compression_values = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)

            # Correct Lip Landmarks
            upper_lip_mid = landmarks.part(51).y
            lower_lip_mid = landmarks.part(57).y
            left_corner = landmarks.part(49).x
            right_corner = landmarks.part(55).x
            
            # Calculate Vertical Lip Movement (Open/Close)
            vertical_distance = abs(upper_lip_mid - lower_lip_mid)
            vertical_movements.append(vertical_distance)
            
            # Calculate Horizontal Lip Movement (Stretching)
            horizontal_distance = abs(left_corner - right_corner)
            horizontal_movements.append(horizontal_distance)
            
            # Calculate Lip Compression (Sudden Lip Tightening)
            upper_lip_thickness = landmarks.part(62).y - landmarks.part(51).y
            lower_lip_thickness = landmarks.part(66).y - landmarks.part(57).y
            lip_compression = abs(upper_lip_thickness + lower_lip_thickness)
            lip_compression_values.append(lip_compression)
            
            # Calculate Lip Tension (Horizontal Stretching)
            lip_tension = abs(right_corner - left_corner)
            lip_tension_values.append(lip_tension)
    
    # Release Video Capture
    cap.release()
    
    # Calculate Standard Deviations (edge)
    vertical_edge = np.std(vertical_movements)
    horizontal_edge = np.std(horizontal_movements)
    compression_edge = np.std(lip_compression_values)
    tension_edge = np.std(lip_tension_values)
    
    # Calculate Total edge (Lie Probability)
    total_edge = (vertical_edge + horizontal_edge + compression_edge + tension_edge) / 4
    lie_probability = min(100, total_edge * 10)
    
    if lie_probability < 20:
        result = "Truth"
    elif lie_probability > 75:
        result = "Lie"
    elif 35 <= lie_probability <= 60:
        result = "Suspicious"
    else:
        result = "Uncertain"
    
    # Return Result With Confidence
    confidence = round(lie_probability, 2)
    return {
        'status': result,
        'confidence': f"{100-confidence}%"
    }