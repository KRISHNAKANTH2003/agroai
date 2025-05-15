from flask import Flask, render_template, request, url_for
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback
from pathlib import Path

# Configure matplotlib for non-GUI environment
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Configuration - using Path for cross-platform compatibility
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
DATA_PATH = MODEL_DIR / 'Crop_Production_data.csv'

# Initialize variables
model = None
label_encoder = None
scaler = None
df = None

try:
    # Load models and data with error handling
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)
    
    # Define recommended crops
    recommended_crops = [
        'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
        'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
        'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
        'coconut', 'cotton', 'jute', 'coffee'
    ]

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
    print(f"Initialization error: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)

def generate_plot_filename(prefix):
    """Generate unique filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.png"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            # Validate system is ready
            if None in (model, label_encoder, scaler, df):
                return render_template('error.html', 
                                    error="System not properly initialized. Please try again later."), 500

            # Get form data with error handling
            try:
                form_data = {
                    'nitrogen': float(request.form.get('nitrogen', 0)),
                    'phosphorus': float(request.form.get('phosphorus', 0)),
                    'potassium': float(request.form.get('potassium', 0)),
                    'temperature': float(request.form.get('temperature', 0)),
                    'humidity': float(request.form.get('humidity', 0)),
                    'ph': float(request.form.get('ph', 0)),
                    'rainfall': float(request.form.get('rainfall', 0))
                }
            except ValueError as e:
                return render_template('error.html',
                                    error="Invalid input values. Please check your inputs and try again."), 400

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

            # Generate plot filenames
            prob_filename = generate_plot_filename('prob')
            state_filename = generate_plot_filename('state')
            district_filename = generate_plot_filename('district')

            # Generate probability plot
            plt.figure(figsize=(10, 6))
            plt.barh(crop_names_sorted[:10], probabilities_sorted[:10], color='green')
            plt.xlabel("Prediction Probability")
            plt.title("Top 10 Crop Recommendation Probabilities")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(STATIC_DIR / prob_filename, bbox_inches='tight')
            plt.close()

            # Generate state and district plots if crop data exists
            crop_df = df[df['Crop'] == predicted_crop]
            if not crop_df.empty:
                # State demand plot
                state_demand = crop_df.groupby('State_Name')['Production'].sum().reset_index()
                state_demand = state_demand.sort_values(by='Production', ascending=False).head(10)

                plt.figure(figsize=(14, 8))
                sns.barplot(data=state_demand, x='State_Name', y='Production', palette='viridis')
                plt.xticks(rotation=90)
                plt.title(f'Top Production of {predicted_crop.title()} Across States')
                plt.xlabel('State')
                plt.ylabel('Total Production')
                plt.tight_layout()
                plt.savefig(STATIC_DIR / state_filename, bbox_inches='tight')
                plt.close()

                # District demand plot
                demand = crop_df.groupby(['State_Name', 'District_Name'])['Production'].sum().reset_index()
                demand['Label'] = demand['District_Name'] + " (" + demand['State_Name'] + ")"
                top_demand = demand.sort_values(by='Production', ascending=False).head(10)

                plt.figure(figsize=(14, 8))
                sns.barplot(data=top_demand, x='Label', y='Production', palette='rocket')
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Top 10 Districts by Production {predicted_crop.title()}')
                plt.xlabel('District (State)')
                plt.ylabel('Total Production')
                plt.tight_layout()
                plt.savefig(STATIC_DIR / district_filename, bbox_inches='tight')
                plt.close()
            else:
                state_filename = None
                district_filename = None

            return render_template('result.html', 
                                 crop=predicted_crop.title(),
                                 prob_filename=prob_filename,
                                 state_filename=state_filename,
                                 district_filename=district_filename)

        except Exception as e:
            print(f"Recommendation error: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
            return render_template('error.html',
                                error="An unexpected error occurred. Please try again."), 500

    return render_template('recommend.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/details')
def details():
    return render_template('details.html')

if __name__ == '__main__':
    app.run(debug=True)