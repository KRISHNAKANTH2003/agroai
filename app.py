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
import sys

# Configure matplotlib for server environments
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'static'
MODEL_DIR = BASE_DIR / 'model'

STATIC_DIR.mkdir(exist_ok=True, parents=True)

# File paths
MODEL_PATH = MODEL_DIR / 'model.pkl'
LABEL_ENCODER_PATH = MODEL_DIR / 'label_encoder.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'
DATA_PATH = MODEL_DIR / "Crop_Production_data.csv"

# Load assets
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)

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

def generate_plot_filename(prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.png"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            if None in (model, label_encoder, scaler, df):
                return render_template('error.html', error="Model not loaded properly."), 500

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

            probabilities = model.predict_proba(scaled_input)[0]
            classes = model.classes_
            crop_names = label_encoder.inverse_transform(classes)

            sorted_indices = np.argsort(probabilities)[::-1]
            crop_names_sorted = crop_names[sorted_indices]
            probabilities_sorted = probabilities[sorted_indices]
            predicted_crop = crop_names_sorted[0].lower()

            # Create plot filenames
            prob_filename = generate_plot_filename("prob")
            state_filename = generate_plot_filename("state")
            district_filename = generate_plot_filename("district")

            # Probability plot
            plt.figure(figsize=(10, 6))
            plt.barh(crop_names_sorted[:10], probabilities_sorted[:10], color='green')
            plt.xlabel("Prediction Probability")
            plt.title("Top 10 Crop Recommendation Probabilities")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(STATIC_DIR / prob_filename)
            plt.close()

            crop_df = df[df['Crop'] == predicted_crop]
            if not crop_df.empty:
                # State-wise
                state_data = crop_df.groupby('State_Name')['Production'].sum().nlargest(10).reset_index()
                plt.figure(figsize=(12, 6))
                sns.barplot(data=state_data, x='State_Name', y='Production', palette='viridis')
                plt.title(f"{predicted_crop.title()} Production Across States")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(STATIC_DIR / state_filename)
                plt.close()

                # District-wise
                district_data = crop_df.groupby(['State_Name', 'District_Name'])['Production'].sum().nlargest(10).reset_index()
                district_data['Label'] = district_data['District_Name'] + " (" + district_data['State_Name'] + ")"
                plt.figure(figsize=(12, 6))
                sns.barplot(data=district_data, x='Label', y='Production', palette='rocket')
                plt.title(f"{predicted_crop.title()} Production Across Districts")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(STATIC_DIR / district_filename)
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
            print(f"Recommendation error: {e}\n{traceback.format_exc()}", file=sys.stderr)
            return render_template('error.html', error="Internal Server Error."), 500

    return render_template('recommend.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/details')
def details():
    return render_template('details.html')

# Only for development (Not used in production with gunicorn)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
