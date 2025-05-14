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
DATA_PATH = MODEL_DIR / "Crop_Production_data.csv"

# Initialize variables to None
model = None
label_encoder = None
scaler = None
df = None

try:
    # Load the model and other necessary files with existence checks
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder file not found at {LABEL_ENCODER_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)
    
except Exception as e:
    print(f"Initialization error: {str(e)}", file=sys.stderr)
    # Don't raise here - let the app start but handle errors in routes

# Define recommended crops and mapping
recommended_crops = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
    'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
    'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
    'coconut', 'cotton', 'jute', 'coffee'
]

# Data processing (only if df was loaded successfully)
if df is not None:
    try:
        # Clean and standardize crop names
        df['State_Name'] = df['State_Name'].str.strip().str.title()
        df['District_Name'] = df['District_Name'].str.strip().str.title()
        df['Crop'] = df['Crop'].str.strip().str.lower()

        # Map original crop names to standard names
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
            'cotton(lint)': 'cotton',
            'rice': 'rice',
            'maize': 'maize',
            'banana': 'banana',
            'mango': 'mango',
            'grapes': 'grapes',
            'orange': 'orange',
            'papaya': 'papaya',
            'coconut': 'coconut',
            'jute': 'jute',
            'coffee': 'coffee',
            'lentil': 'lentil',
            'apple': 'apple'
        }

        # Replace old names with mapped ones
        df['Crop'] = df['Crop'].replace(crop_mapping)

        # Filter dataset to only include recommended crops
        df = df[df['Crop'].isin(recommended_crops)]
    except Exception as e:
        print(f"Data processing error: {str(e)}", file=sys.stderr)
        df = None

def clean_old_plots():
    """Remove plot files older than 1 hour"""
    try:
        now = datetime.now()
        for filename in os.listdir(STATIC_DIR):
            if filename.endswith('.png'):
                filepath = STATIC_DIR / filename
                file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                if now - file_time > timedelta(hours=1):
                    try:
                        filepath.unlink()
                    except Exception as e:
                        print(f"Could not delete {filepath}: {str(e)}", file=sys.stderr)
    except Exception as e:
        print(f"Error cleaning old plots: {str(e)}", file=sys.stderr)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            # Check if models and data are loaded
            if None in (model, label_encoder, scaler, df):
                raise RuntimeError("System not properly initialized. Missing model or data files.")
            
            # Get form data with validation
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

            # Clean old plots before creating new ones
            clean_old_plots()

            # Generate unique filenames using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            predicted_crop = crop_names_sorted[0].lower()
            
            # Create plots
            plots = {}
            for plot_type in ['barplot', 'state_demand', 'district_demand']:
                try:
                    plot_path = STATIC_DIR / f'{plot_type}_{timestamp}.png'
                    
                    plt.figure(figsize=(10, 6))
                    if plot_type == 'barplot':
                        plt.barh(crop_names_sorted, probabilities_sorted, color='green')
                        plt.xlabel("Prediction Probability")
                        plt.title("Crop Recommendation Probabilities")
                        plt.gca().invert_yaxis()
                    elif plot_type == 'state_demand':
                        crop_df = df[df['Crop'] == predicted_crop]
                        state_demand = crop_df.groupby('State_Name')['Production'].sum().reset_index()
                        state_demand = state_demand.sort_values(by='Production', ascending=False)
                        sns.barplot(data=state_demand, x='State_Name', y='Production', palette='viridis')
                        plt.xticks(rotation=90)
                        plt.title(f'Top Production of {predicted_crop.title()} Across States')
                        plt.xlabel('State')
                        plt.ylabel('Total Production')
                    elif plot_type == 'district_demand':
                        crop_df = df[df['Crop'] == predicted_crop]
                        demand = crop_df.groupby(['State_Name', 'District_Name'])['Production'].sum().reset_index()
                        demand['Label'] = demand['District_Name'] + " (" + demand['State_Name'] + ")"
                        top_demand = demand.sort_values(by='Production', ascending=False).head(5)
                        sns.barplot(data=top_demand, x='Label', y='Production', palette='rocket')
                        plt.xticks(rotation=45, ha='right')
                        plt.title(f'Top 5 Districts by Production {predicted_crop.title()}')
                        plt.xlabel('District (State)')
                        plt.ylabel('Total Production')
                    
                    plt.tight_layout()
                    plt.savefig(plot_path, bbox_inches='tight')
                    plt.close()
                    plots[plot_type] = plot_path.relative_to(BASE_DIR).as_posix()
                    
                except Exception as e:
                    print(f"Error generating {plot_type} plot: {str(e)}", file=sys.stderr)
                    plots[plot_type] = None

            return render_template('result.html', 
                                crop=predicted_crop.title(), 
                                plot_path=plots['barplot'],
                                state_plot_path=plots['state_demand'],
                                district_plot_path=plots['district_demand'])

        except Exception as e:
            print(f"Recommendation error: {str(e)}", file=sys.stderr)
            return render_template('error.html', error=str(e)), 500

    return render_template('recommend.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/details')
def details():
    return render_template('details.html')

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=PORT)