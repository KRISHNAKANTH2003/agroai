from flask import Flask, render_template, request, url_for
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
DATA_PATH = MODEL_DIR / "Crop_production_data.csv"

# Initialize variables
model = None
label_encoder = None
scaler = None
df = None

# [Previous initialization and data loading code remains the same until generate_crop_graph]

def generate_crop_graphs(crop_name):
    """Generate multiple graphs showing production data for the recommended crop"""
    try:
        # Filter data for the recommended crop
        crop_data = df[df['Crop'].str.lower() == crop_name.lower()]
        
        if crop_data.empty:
            print(f"No data found for crop: {crop_name}")
            return None, None
            
        # Clean old plots
        clean_old_plots()
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Generate state-wise production plot
        state_filename = f'state_{crop_name.lower()}_{timestamp}.png'
        state_path = STATIC_DIR / state_filename
        
        plt.figure(figsize=(10, 6))
        state_production = crop_data.groupby('State_Name')['Production'].sum().nlargest(10)
        state_production.plot(kind='bar', color='green')
        plt.title(f'Top 10 States for {crop_name.title()} Production')
        plt.ylabel('Production (tons)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(state_path, bbox_inches='tight')
        plt.close()
        
        # Generate district-wise production plot
        district_filename = f'district_{crop_name.lower()}_{timestamp}.png'
        district_path = STATIC_DIR / district_filename
        
        plt.figure(figsize=(10, 6))
        district_production = crop_data.groupby('District_Name')['Production'].sum().nlargest(10)
        district_production.plot(kind='bar', color='blue')
        plt.title(f'Top 10 Districts for {crop_name.title()} Production')
        plt.ylabel('Production (tons)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(district_path, bbox_inches='tight')
        plt.close()
        
        return state_filename, district_filename
    
    except Exception as e:
        print(f"Error generating crop graphs: {str(e)}", file=sys.stderr)
        return None, None

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            # [Previous form data processing and prediction code...]
            
            # Get the recommended crop
            predicted_crop = crop_names_sorted[0].lower()
            print(f"Recommended crop: {predicted_crop}")
            
            # Generate crop-specific graphs
            state_filename, district_filename = generate_crop_graphs(predicted_crop)
            
            # Generate probability plot
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            prob_filename = f'prob_{timestamp}.png'
            prob_path = STATIC_DIR / prob_filename
            
            plt.figure(figsize=(10, 6))
            plt.barh(crop_names_sorted[:10], probabilities_sorted[:10], color='green')
            plt.xlabel("Prediction Probability")
            plt.title("Top 10 Crop Recommendation Probabilities")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(prob_path, bbox_inches='tight')
            plt.close()
            
            return render_template('result.html',
                                crop=predicted_crop.title(),
                                prob_filename=prob_filename,
                                state_filename=state_filename,
                                district_filename=district_filename,
                                probabilities=dict(zip(crop_names_sorted, probabilities_sorted)))

        except Exception as e:
            print(f"Recommendation error: {str(e)}", file=sys.stderr)
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