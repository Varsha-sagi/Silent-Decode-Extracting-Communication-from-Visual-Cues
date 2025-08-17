from flask import Flask, render_template,request,session,redirect,url_for
import os
import asyncio
from werkzeug.utils import secure_filename
from main import process_video,create_audio,translate, calculate_lie_confidence

UPLOAD_FOLDER = os.path.join('static/assets','uploads')
ALLOWED_EXTENSIONS = {'mpg' ,'mp4','wmv'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "LipSync"


@app.route('/',methods=['POST','GET'])
def index():
    data = {
        "video": session.pop("uploaded_video_filepath", None),
        "prediction": session.pop("prediction", None),
        "translated_predictions": [
            session.pop("translated_prediction_ta", None),
            session.pop("translated_prediction_te", None),
            session.pop("translated_prediction_hi", None),
        ],
        "audios": {
            "en": session.pop("en_audio", None),
            "ta": session.pop("ta_audio", None),
            "te": session.pop("te_audio", None),
            "hi": session.pop("hi_audio", None),
        },
        "lie_detection": session.pop("lie_detection", None),
    }
    session.clear()
    return render_template("index.html", **data)


@app.route("/predict",methods=['POST','GET'])
def predict():
    if request.method=="POST":
        video = request.files["video"]
        video_filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        session['uploaded_video_filepath'] = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video_filepath = session.get('uploaded_video_filepath', None)
        
        session['lie_detection'] = calculate_lie_confidence(video_filepath)

        converted_prediction = process_video(video_filepath)

        session['prediction'] = converted_prediction
        create_audio(converted_prediction,"en")
        session['en_audio'] = f"static/assets/audio/en_audio.mp3"
        
        session['translated_prediction_ta'] = asyncio.run(translate(converted_prediction,"ta"))
        create_audio(session.get('translated_prediction_ta', None),"ta")
        session['ta_audio'] = f"static/assets/audio/ta_audio.mp3"

        session['translated_prediction_te'] = asyncio.run(translate(converted_prediction,"te"))
        create_audio(session.get('translated_prediction_te', None),"te")
        session['te_audio'] = f"static/assets/audio/te_audio.mp3"

        session['translated_prediction_hi'] = asyncio.run(translate(converted_prediction,"hi"))
        create_audio(session.get('translated_prediction_hi', None),"hi")
        session['hi_audio'] = f"static/assets/audio/hi_audio.mp3"
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run()