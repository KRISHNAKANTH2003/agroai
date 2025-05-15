from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
DATA_PATH = MODEL_DIR / "Crop_production_data.csv"

# Initialize variables
model = None
label_encoder = None
scaler = None
df = None

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
    
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

except Exception as e:
    print(f"Initialization error: {str(e)}", file=sys.stderr)

# Define recommended crops and mapping
recommended_crops = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
    'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
    'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
    'coconut', 'cotton', 'jute', 'coffee'
]

# Process data if loaded successfully
if df is not None:
    try:
        # Clean and standardize data
        df['State_Name'] = df['State_Name'].str.strip().str.title()
        df['District_Name'] = df['District_Name'].str.strip().str.title()
        df['Crop'] = df['Crop'].str.strip().str.lower()

        # Crop name mapping
        crop_mapping = {
            'moong(green gram)': 'mungbean',
            'urad': 'blackgram',
            'arhar/tur': 'pigeonpeas',
            'gram': 'chickpea',
            'masoor': 'lentil',
            'pome granet': 'pomegranate',
            'pome fruit': 'pomegranate',
            'water melon': 'watermelon',
            'muskmelon': 'muskmelon',
            'cotton(lint)': 'cotton'
        }
        df['Crop'] = df['Crop'].replace(crop_mapping)
        df = df[df['Crop'].isin(recommended_crops)]
    except Exception as e:
        print(f"Data processing error: {str(e)}", file=sys.stderr)

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

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            if model is None or label_encoder is None or scaler is None or df is None:
                return render_template('error.html', 
                                       error="System not properly initialized. Please try again later."), 500

            # 1. Get form data
            form_data = {
                'nitrogen': float(request.form.get('nitrogen', 0)),
                'phosphorus': float(request.form.get('phosphorus', 0)),
                'potassium': float(request.form.get('potassium', 0)),
                'temperature': float(request.form.get('temperature', 0)),
                'humidity': float(request.form.get('humidity', 0)),
                'ph': float(request.form.get('ph', 0)),
                'rainfall': float(request.form.get('rainfall', 0))
            }

            # 2. Prepare input
            input_features = list(form_data.values())
            print("User Input:", input_features)  # Debug line

            sample_input = np.array([input_features])
            scaled_input = scaler.transform(sample_input)

            # 3. Predict
            probabilities = model.predict_proba(scaled_input)[0]
            classes = model.classes_
            crop_names = label_encoder.inverse_transform(classes)

            # 4. Sort by probability
            sorted_indices = np.argsort(probabilities)[::-1]
            crop_names_sorted = crop_names[sorted_indices]
            probabilities_sorted = probabilities[sorted_indices]

            print("Predictions:", list(zip(crop_names_sorted, probabilities_sorted)))  # Debug line

            # 5. Top N crops
            top_n = 5
            top_crop_names = crop_names_sorted[:top_n]
            top_probabilities = probabilities_sorted[:top_n]

            predicted_crop = top_crop_names[0].lower()

            # 6. Generate dynamic plot
            clean_old_plots()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = STATIC_DIR / f'plot_{timestamp}.png'

            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_probabilities, y=top_crop_names, palette='coolwarm')
            plt.xlabel("Prediction Probability")
            plt.title(f"Top {top_n} Crop Predictions Based on Input")
            plt.xlim(0, 1.0)
            plt.tight_layout()
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

            return render_template('result.html',
                                   crop=predicted_crop.title(),
                                   plot_path=f'static/{plot_path.name}')

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