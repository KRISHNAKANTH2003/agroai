from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Configure matplotlib for non-GUI environment
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / 'static'
MODEL_DIR = BASE_DIR / 'model'

# Ensure directories exist
STATIC_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# File paths
MODEL_PATH = MODEL_DIR / 'model.pkl'
LABEL_ENCODER_PATH = MODEL_DIR / 'label_encoder.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'

# Initialize variables
model = None
label_encoder = None
scaler = None

try:
    # Load model files with explicit checks
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    if LABEL_ENCODER_PATH.exists():
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    else:
        raise FileNotFoundError(f"Label encoder file not found at {LABEL_ENCODER_PATH}")
    
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
    else:
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

except Exception as e:
    print(f"Initialization error: {str(e)}", file=sys.stderr)

def clean_old_plots():
    """Remove plot files older than 1 hour"""
    try:
        now = datetime.now()
        for file in STATIC_DIR.glob('*.png'):
            if (now - datetime.fromtimestamp(file.stat().st_mtime)) > timedelta(hours=1):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Could not delete {file}: {str(e)}", file=sys.stderr)
    except Exception as e:
        print(f"Error cleaning old plots: {str(e)}", file=sys.stderr)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            # Check if models are loaded
            if model is None or label_encoder is None or scaler is None:
                return render_template('error.html', 
                                    error="System not properly initialized. Please try again later."), 500

            # Get form data
            form_data = {
                'nitrogen': float(request.form.get('nitrogen', 0)),
                'phosphorus': float(request.form.get('phosphorus', 0)),
                'potassium': float(request.form.get('potassium', 0)),
                'temperature': float(request.form.get('temperature', 0)),
                'humidity': float(request.form.get('humidity', 0)),
                'ph': float(request.form.get('ph', 0)),
                'rainfall': float(request.form.get('rainfall', 0))
            }

            # Prepare and scale input
            sample_input = np.array([list(form_data.values())])
            sample_input = scaler.transform(sample_input)

            # Predict probabilities
            probabilities = model.predict_proba(sample_input)[0]
            classes = model.classes_
            crop_names = label_encoder.inverse_transform(classes)

            # Create list of crops with their probabilities
            crop_probabilities = list(zip(crop_names, probabilities))
            # Sort by probability (descending)
            crop_probabilities.sort(key=lambda x: x[1], reverse=True)

            # Clean old plots
            clean_old_plots()

            # Generate probability plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = STATIC_DIR / f'probabilities_{timestamp}.png'
            
            plt.figure(figsize=(10, 6))
            crops, probs = zip(*crop_probabilities[:10])  # Show top 10 crops
            plt.barh(crops, probs, color='green')
            plt.xlabel("Prediction Probability")
            plt.title("Top Recommended Crops with Probabilities")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

            return render_template('result.html',
                                top_crop=crop_probabilities[0][0].title(),
                                top_probability=f"{crop_probabilities[0][1]*100:.2f}%",
                                crop_probabilities=crop_probabilities,
                                plot_path=plot_path.relative_to(BASE_DIR).as_posix())

        except Exception as e:
            print(f"Recommendation error: {str(e)}", file=sys.stderr)
            return render_template('error.html',
                                error="Sorry, we couldn't process your request. Please try again."), 500

    return render_template('recommend.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/debug')
def debug():
    """Debug endpoint to check file loading"""
    return {
        'model_loaded': model is not None,
        'data_loaded': df is not None,
        'files_exist': {
            'model': MODEL_PATH.exists(),
            'label_encoder': LABEL_ENCODER_PATH.exists(),
            'scaler': SCALER_PATH.exists(),
            'data': DATA_PATH.exists()
        }
    }

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=PORT)