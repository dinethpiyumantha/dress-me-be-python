import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import pose_detection as pd
import json
import numpy as np
import _utils as ult

app = Flask(__name__)

UPLOAD_FOLDER = 'UPLOAD_VIDEOS'  # Specify the path where you want to save the uploaded videos
UPLOAD_IMAGES = 'UPLOAD_IMAGES'  # Specify the path where you want to save the uploaded videos
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'jpg', 'png'}  # Specify the allowed file extensions

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_IMAGES'] = UPLOAD_IMAGES

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return {'error': 'No video file provided'}, 400
    
    source_file = request.files['video']
    
    if source_file and allowed_file(source_file.filename):
        # Generate a unique filename or use the original filename
        filename = secure_filename(source_file.filename)
        
        # Create the destination folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save the video file to the specified path
        source_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Optionally, perform any additional processing or operations on the video
        
        
        return {'message': 'Video uploaded successfully'}, 200
    
    return {'error': 'Invalid video file'}, 400


@app.route('/api/predict/keypoints', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {'error': 'No video file provided'}, 400
    
    source_file = request.files['file']
    
    if source_file and allowed_file(source_file.filename):
        # Generate a unique filename or use the original filename
        filename = secure_filename(source_file.filename)
        
        # Create the destination folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_IMAGES'], exist_ok=True)

        # Save the video file to the specified path
        file_path=os.path.join(app.config['UPLOAD_IMAGES'], filename)
        source_file.save(file_path)
        
        # Optionally, perform any additional processing or operations on the video
        res=pd.get_list_of_keypoints_by_path(file_path)
        res=ult.ndarray_to_list(res)
        return {'message': 'image upload and prediction successfully', 'result': json.dumps(res, default=ult.handle_non_serializable)}, 200
    
    return {'error': 'Invalid video file'}, 400

if __name__ == '__main__':
    app.run()