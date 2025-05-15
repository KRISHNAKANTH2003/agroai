from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import traceback
import sys

# Configure matplotlib for server use
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'static'
MODEL_DIR = BASE_DIR / 'model'

# Fixed output plot filenames
BARPLOT_PATH = STATIC_DIR / 'barplot.png'
STATE_DEMAND_PATH = STATIC_DIR / 'state_demand.png'
DISTRICT_DEMAND_PATH = STATIC_DIR / 'district_demand.png'

# Load model assets
try:
    model = joblib.load(MODEL_DIR / 'model.pkl')
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    label_encoder = joblib.load(MODEL_DIR / 'label_encoder.pkl')
    df = pd.read_csv(MODEL_DIR / 'Crop_production_data.csv')

    # Clean and filter dataset
    recommended_crops = [
        'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
        'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
        'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
        'coconut', 'cotton', 'jute', 'coffee'
    ]

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
    df = df[df['Crop'].isin(recommended_crops)]

except Exception as e:
    print(f"Initialization error: {e}\n{traceback.format_exc()}", file=sys.stderr)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            # Input validation
            form_data = {
                'nitrogen': float(request.form.get('nitrogen', 0)),
                'phosphorus': float(request.form.get('phosphorus', 0)),
                'potassium': float(request.form.get('potassium', 0)),
                'temperature': float(request.form.get('temperature', 0)),
                'humidity': float(request.form.get('humidity', 0)),
                'ph': float(request.form.get('ph', 0)),
                'rainfall': float(request.form.get('rainfall', 0))
            }

            sample_input = np.array([list(form_data.values())])
            scaled_input = scaler.transform(sample_input)

            # Prediction
            probabilities = model.predict_proba(scaled_input)[0]
            classes = model.classes_
            crop_names = label_encoder.inverse_transform(classes)
            sorted_indices = np.argsort(probabilities)[::-1]
            crop_names_sorted = crop_names[sorted_indices]
            probabilities_sorted = probabilities[sorted_indices]
            predicted_crop = crop_names_sorted[0].lower()

            # Plot 1: barplot.png — Crop probabilities
            plt.figure(figsize=(10, 6))
            plt.barh(crop_names_sorted[:10], probabilities_sorted[:10], color='green')
            plt.xlabel("Prediction Probability")
            plt.title("Top 10 Crop Recommendation Probabilities")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(BARPLOT_PATH, bbox_inches='tight')
            plt.close()

            # Get crop data for demand plots
            crop_df = df[df['Crop'] == predicted_crop]

            # Plot 2: state_demand.png
            if not crop_df.empty:
                state_data = crop_df.groupby('State_Name')['Production'].sum().nlargest(10).reset_index()
                plt.figure(figsize=(12, 6))
                sns.barplot(data=state_data, x='State_Name', y='Production', palette='viridis')
                plt.xticks(rotation=45)
                plt.title(f'{predicted_crop.title()} Production Across Top States')
                plt.tight_layout()
                plt.savefig(STATE_DEMAND_PATH, bbox_inches='tight')
                plt.close()

                # Plot 3: district_demand.png
                district_data = crop_df.groupby(['State_Name', 'District_Name'])['Production'].sum().nlargest(10).reset_index()
                district_data['Label'] = district_data['District_Name'] + " (" + district_data['State_Name'] + ")"
                plt.figure(figsize=(12, 6))
                sns.barplot(data=district_data, x='Label', y='Production', palette='rocket')
                plt.xticks(rotation=45, ha='right')
                plt.title(f'{predicted_crop.title()} Production in Top Districts')
                plt.tight_layout()
                plt.savefig(DISTRICT_DEMAND_PATH, bbox_inches='tight')
                plt.close()

            return render_template(
                'result.html',
                crop=predicted_crop.title(),
                prob_img='barplot.png',
                state_img='state_demand.png',
                district_img='district_demand.png'
            )

        except Exception as e:
            print(f"Error in recommend(): {e}\n{traceback.format_exc()}", file=sys.stderr)
            return render_template('error.html', error="Something went wrong. Please try again."), 500

    return render_template('recommend.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/details')
def details():
    return render_template('details.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
