from flask import Flask, render_template, request, url_for
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import sys
import traceback

# Configure matplotlib for non-GUI environment
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / 'static'
MODEL_DIR = BASE_DIR / 'model'

# Ensure directories exist
STATIC_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

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

# Load models and data
try:
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    label_encoder = joblib.load(LABEL_ENCODER_PATH) if LABEL_ENCODER_PATH.exists() else None
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
    
    if df is not None:
        # Data cleaning and processing
        df['State_Name'] = df['State_Name'].str.strip().str.title()
        df['District_Name'] = df['District_Name'].str.strip().str.title()
        df['Crop'] = df['Crop'].str.strip().str.lower()
        
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
        
        recommended_crops = [
            'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
            'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
            'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
            'coconut', 'cotton', 'jute', 'coffee'
        ]
        df = df[df['Crop'].isin(recommended_crops)]
        
except Exception as e:
    print(f"Initialization error: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)

def clean_old_plots():
    """Remove plot files older than 1 hour"""
    try:
        now = datetime.now()
        for file in STATIC_DIR.glob('*.png'):
            if (now - datetime.fromtimestamp(file.stat().st_mtime)) > timedelta(hours=1):
                file.unlink()
    except Exception as e:
        print(f"Error cleaning old plots: {str(e)}", file=sys.stderr)

def generate_graph(data, title, xlabel, ylabel, filename, kind='bar', color='green', rotation=0):
    """Helper function to generate and save plots"""
    try:
        plt.figure(figsize=(10, 6))
        if kind == 'bar':
            data.plot(kind=kind, color=color)
        elif kind == 'barh':
            data.plot(kind=kind, color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=rotation)
        plt.tight_layout()
        plt.savefig(STATIC_DIR / filename, bbox_inches='tight')
        plt.close()
        return filename
    except Exception as e:
        print(f"Error generating {filename}: {str(e)}", file=sys.stderr)
        return None

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            # Validate system is ready
            if None in (model, label_encoder, scaler, df):
                return render_template('error.html', 
                                    error="System not properly initialized. Please try again later."), 500

            # Get form data with defaults
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

            # Sort crops by probability
            sorted_indices = np.argsort(probabilities)[::-1]
            crop_names_sorted = crop_names[sorted_indices]
            probabilities_sorted = probabilities[sorted_indices]
            predicted_crop = crop_names_sorted[0].lower()

            # Clean old plots
            clean_old_plots()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # Generate probability plot
            prob_data = pd.Series(probabilities_sorted[:10], index=crop_names_sorted[:10])
            prob_filename = generate_graph(
                prob_data,
                "Top 10 Crop Recommendation Probabilities",
                "Prediction Probability",
                "Crop",
                f'prob_{timestamp}.png',
                kind='barh',
                rotation=0
            )

            # Generate crop-specific plots if data exists
            crop_data = df[df['Crop'].str.lower() == predicted_crop.lower()]
            state_filename = district_filename = None
            
            if not crop_data.empty:
                # State-wise production
                state_production = crop_data.groupby('State_Name')['Production'].sum().nlargest(10)
                state_filename = generate_graph(
                    state_production,
                    f'Top 10 States for {predicted_crop.title()} Production',
                    "State",
                    "Production (tons)",
                    f'state_{timestamp}.png',
                    rotation=45
                )

                # District-wise production
                district_production = crop_data.groupby('District_Name')['Production'].sum().nlargest(10)
                district_filename = generate_graph(
                    district_production,
                    f'Top 10 Districts for {predicted_crop.title()} Production',
                    "District",
                    "Production (tons)",
                    f'district_{timestamp}.png',
                    rotation=45
                )

            return render_template('result.html',
                                crop=predicted_crop.title(),
                                prob_filename=prob_filename,
                                state_filename=state_filename,
                                district_filename=district_filename,
                                probabilities=dict(zip(crop_names_sorted, probabilities_sorted)))

        except Exception as e:
            print(f"Recommendation error: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
            return render_template('error.html',
                                error="Sorry, we couldn't process your request. Please try again."), 500

    return render_template('recommend.html')


@app.route('/')
def home():
    return render_template('index.html')

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