import os
import io
import tempfile
from flask import Flask, request, jsonify
import whisper
import torch
from pathlib import Path
import time
from waitress import serve
import threading

app = Flask(__name__)

# Global variable to store the model (for reuse across requests)
model = None

def load_model():
    """Load Whisper model from local file with optimizations for serverless"""
    global model
    if model is None:
        # Path configuration
        model_dir = os.environ.get('MODEL_DIR', '/models/whisper')
        model_size = os.environ.get('WHISPER_MODEL', 'small')
        model_path = os.path.join(model_dir, f"{model_size}.pt")
        
        # Load model with GPU/CPU detection
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Whisper {model_size} model from {model_path} on {device}...")
        
        # Check if local model exists
        if os.path.exists(model_path):
            print(f"Found local model at {model_path}")
            model = whisper.load_model(model_path, device=device)
        else:
            #don't download model if it doesn't exist

            print(f"Local model not found at {model_path}, exiting.")
            # Create model directory if it doesn't exist
            exit(1)
        
        print("Model loaded successfully")
    
    return model

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "whisper-transcription"})

@app.route('/predict', methods=['POST'])
def predict():
        start = time.time()
        data = {"success": False}
        print("predicting...")
        try:
            # Check if audio file is present under 'data' or 'payload'
            if 'data' not in request.files and 'payload' not in request.files:
                return jsonify(data), 400
            audio_file = request.files.get('data') or request.files.get('payload')
            if audio_file is None or audio_file.filename == '':
                return jsonify(data), 400
            # Load model (cached after first load)
            whisper_model = load_model()
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                audio_file.save(tmp_file.name)
                # Only lock the model inference part
                    # Transcribe audio
                result = whisper_model.transcribe(
                    tmp_file.name,
                    fp16=False,  # Disable fp16 for CPU compatibility
                    task='transcribe'
                )
                os.unlink(tmp_file.name)
            output = result["text"].strip()
            data["success"] = True
            data["output"] = output
            print("output:", output)
            print("elapsed: ", time.time() - start, " with success ", data["success"])
            return jsonify(data)
        except Exception as e:
            print("error:", str(e))
            return jsonify(data), 500



if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Model loaded")
    # """Run the app with Waitress for production use"""
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Whisper Flask app with Waitress...")
    print(f"Listening on http://{host}:{port}")
    serve(app, host="0.0.0.0", port=5000, backlog=10, connection_limit=10)
    