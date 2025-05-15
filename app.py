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
        # Get form data
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare and scale input
        sample_input = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        sample_input = scaler.transform(sample_input)

        # Predict probabilities
        probabilities = model.predict_proba(sample_input)[0]
        classes = model.classes_
        crop_names = label_encoder.inverse_transform(classes)

        # Sort crops by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        crop_names_sorted = crop_names[sorted_indices]
        probabilities_sorted = probabilities[sorted_indices]

        # Plot and save bar chart
        plt.figure(figsize=(10, 6))
        plt.barh(crop_names_sorted, probabilities_sorted, color='green')
        plt.xlabel("Prediction Probability")
        plt.title("Crop Recommendation Probabilities")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plot_path = '//static//barplot.png'
        plt.savefig(plot_path)
        plt.close()

        # Predicted crop
        predicted_crop = crop_names_sorted[0].lower()  # Convert to lowercase for matching

        # Create state demand plot
        crop_df = df[df['Crop'] == predicted_crop]
        state_demand = crop_df.groupby('State_Name')['Production'].sum().reset_index()
        state_demand = state_demand.sort_values(by='Production', ascending=False)

        plt.figure(figsize=(14, 8))
        sns.barplot(data=state_demand, x='State_Name', y='Production', palette='viridis')
        plt.xticks(rotation=90)
        plt.title(f'Top Production of {predicted_crop.title()} Across States')
        plt.xlabel('State')
        plt.ylabel('Total Production')
        plt.tight_layout()
        state_plot_path = '//static//state_demand.png'
        plt.savefig(state_plot_path)
        plt.close()

        # Create district demand plot
        demand = crop_df.groupby(['State_Name', 'District_Name'])['Production'].sum().reset_index()
        demand['Label'] = demand['District_Name'] + " (" + demand['State_Name'] + ")"
        top_demand = demand.sort_values(by='Production', ascending=False).head(5)

        plt.figure(figsize=(14, 8))
        sns.barplot(data=top_demand, x='Label', y='Production', palette='rocket')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top 10 Districts by Production {predicted_crop.title()}')
        plt.xlabel('District (State)')
        plt.ylabel('Total Production')
        plt.tight_layout()
        district_plot_path = '//static//district_demand.png'
        plt.savefig(district_plot_path)
        plt.close()

        return render_template('result.html', 
                             crop=predicted_crop.title(), 
                             plot_path=plot_path,
                             state_plot_path=state_plot_path,
                             district_plot_path=district_plot_path)

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