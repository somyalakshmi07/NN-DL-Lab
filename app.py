# app.py
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import StringField, SubmitField
import cv2
import numpy as np
import onnxruntime as ort
import os
from tracker import PersonTracker
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'crowdcount2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_OUTPUT'] = 'static'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_OUTPUT'], exist_ok=True)

# Load CSRNet
session = ort.InferenceSession("csrnet.onnx", providers=['CPUExecutionProvider'])

class UploadForm(FlaskForm):
    file = FileField()
    submit = SubmitField('Analyze')

class RTSPForm(FlaskForm):
    rtsp_url = StringField('RTSP URL')
    submit = SubmitField('Start Stream')

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    tracker = PersonTracker()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    total_crowd = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        h, w = frame.shape[:2]
        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, 20, (w, h))

        # Crowd count
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (224, 224), swapRB=True, crop=False)
        density_map = session.run(None, {session.get_inputs()[0].name: blob})[0][0, 0]
        crowd_count = int(np.sum(density_map))
        total_crowd += crowd_count

        # Track
        tracked_frame, entry_count = tracker.update(frame.copy())

        # Overlay
        cv2.putText(tracked_frame, f"Crowd: {crowd_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)
        out.write(tracked_frame)

    cap.release()
    if out: out.release()
    return total_crowd // max(frame_count, 1), len(tracker.entry_counts)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        output_filename = f"output_{filename.split('.')[0]}.mp4"
        output_path = os.path.join(app.config['STATIC_OUTPUT'], output_filename)

        avg_crowd, tracked = process_video(input_path, output_path)

        return render_template('result.html', crowd_count=avg_crowd, tracked_count=tracked,
                               video_url=output_filename)

    return render_template('index.html', form=form)

@app.route('/stream', methods=['POST'])
def stream():
    rtsp_url = request.form['rtsp_url']
    # In production: run in background thread or return stream URL
    return f"<h2>RTSP Stream will run in CLI:</h2><pre>python realtime.py (set SOURCE='{rtsp_url}')</pre><a href='/'>Back</a>"

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['STATIC_OUTPUT'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)