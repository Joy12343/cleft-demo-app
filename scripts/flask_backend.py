#!/usr/bin/env python3
"""
Flask backend for face inpainting web application.
Integrates with the existing wrapper.py script.
"""

import os
import tempfile
import shutil
import uuid
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import time

# Import the existing wrapper functions
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# We'll need to modify the wrapper.py to be importable
from wrapper_modified import (
    convert_pngs_to_jpgs,
    generate_landmarks,
    generate_binary_mask,
    run_inpainting_process
)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Store for cleanup - remove old files periodically
cleanup_store = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """Remove files older than 1 hour"""
    while True:
        try:
            current_time = datetime.now()
            for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                        if current_time - file_time > timedelta(hours=1):
                            os.remove(filepath)
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        time.sleep(3600)  # Run every hour

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

@app.route('/api/process', methods=['POST'])
def process_images():
    try:
        # Check if files are present
        if 'source' not in request.files or 'mask' not in request.files:
            return jsonify({'error': 'Both source and mask images are required'}), 400
        
        source_file = request.files['source']
        mask_file = request.files['mask']
        
        # Validate files
        if source_file.filename == '' or mask_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if not (allowed_file(source_file.filename) and allowed_file(mask_file.filename)):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files'}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Save uploaded files
        source_filename = secure_filename(f"source_{source_file.filename}")
        mask_filename = secure_filename(f"mask_{mask_file.filename}")
        
        source_path = os.path.join(session_folder, source_filename)
        mask_path = os.path.join(session_folder, mask_filename)
        
        source_file.save(source_path)
        mask_file.save(mask_path)
        
        # Process images using the wrapper functions
        try:
            # Convert PNGs to JPGs if needed
            convert_pngs_to_jpgs(session_folder)
            
            # Update filenames if they were converted
            for f in os.listdir(session_folder):
                if f.startswith('source_') and f.endswith('.jpg'):
                    source_filename = f
                elif f.startswith('mask_') and f.endswith('.jpg'):
                    mask_filename = f
            
            # Generate landmarks
            landmark_jpg, landmark_txt = generate_landmarks(session_folder, source_filename)
            
            # Generate binary mask
            binary_mask = generate_binary_mask(session_folder, mask_filename)
            
            # Run the inpainting process
            result_path = run_inpainting_process(
                session_folder, 
                source_filename, 
                binary_mask, 
                landmark_txt
            )
            
            # Move results to results folder
            result_session_folder = os.path.join(RESULTS_FOLDER, session_id)
            os.makedirs(result_session_folder, exist_ok=True)
            
            # Copy result files
            final_result_path = os.path.join(result_session_folder, 'result.jpg')
            final_landmark_path = os.path.join(result_session_folder, 'landmarks.jpg')
            
            shutil.copy(result_path, final_result_path)
            shutil.copy(os.path.join(session_folder, landmark_jpg), final_landmark_path)
            
            # Clean up session folder
            shutil.rmtree(session_folder)
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'result_url': f'/api/download/{session_id}/result.jpg',
                'landmark_url': f'/api/download/{session_id}/landmarks.jpg'
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(session_folder):
                shutil.rmtree(session_folder)
            raise e
            
    except Exception as e:
        print(f"Processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<session_id>/<filename>')
def download_file(session_id, filename):
    try:
        file_path = os.path.join(RESULTS_FOLDER, session_id, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
