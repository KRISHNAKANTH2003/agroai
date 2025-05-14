from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import shutil

# Configure matplotlib for non-GUI environment
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Ensure directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
DATA_PATH = os.path.join(MODEL_DIR, 'Crop_Production_data.csv')

try:
    # Load the model and other necessary files
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

# Define recommended crops and mapping
recommended_crops = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
    'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
    'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
    'coconut', 'cotton', 'jute', 'coffee'
]

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

def clean_old_plots():
    """Remove plot files older than 1 hour"""
    now = datetime.now()
    for filename in os.listdir(STATIC_DIR):
        if filename.endswith('.png'):
            filepath = os.path.join(STATIC_DIR, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if now - file_time > timedelta(hours=1):
                try:
                    os.remove(filepath)
                except:
                    pass

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            # Get form data
            nitrogen = float(request.form['nitrogen'])
            phosphorus = float(request.form['phosphorus'])
            potassium = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Prepare and scale input
            sample_input = np.array([[nitrogen, phosphorus, potassium, 
                                    temperature, humidity, ph, rainfall]])
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
            plot_path = os.path.join(STATIC_DIR, f'barplot_{timestamp}.png')
            state_plot_path = os.path.join(STATIC_DIR, f'state_demand_{timestamp}.png')
            district_plot_path = os.path.join(STATIC_DIR, f'district_demand_{timestamp}.png')

            # Plot and save bar chart
            plt.figure(figsize=(10, 6))
            plt.barh(crop_names_sorted, probabilities_sorted, color='green')
            plt.xlabel("Prediction Probability")
            plt.title("Crop Recommendation Probabilities")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            # Predicted crop
            predicted_crop = crop_names_sorted[0].lower()

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
            plt.savefig(state_plot_path)
            plt.close()

            # Create district demand plot
            demand = crop_df.groupby(['State_Name', 'District_Name'])['Production'].sum().reset_index()
            demand['Label'] = demand['District_Name'] + " (" + demand['State_Name'] + ")"
            top_demand = demand.sort_values(by='Production', ascending=False).head(5)

            plt.figure(figsize=(14, 8))
            sns.barplot(data=top_demand, x='Label', y='Production', palette='rocket')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Top 5 Districts by Production {predicted_crop.title()}')
            plt.xlabel('District (State)')
            plt.ylabel('Total Production')
            plt.tight_layout()
            plt.savefig(district_plot_path)
            plt.close()

            # Convert paths to relative for web access
            def get_relative_path(full_path):
                return os.path.relpath(full_path, start=BASE_DIR).replace('\\', '/')

            return render_template('result.html', 
                                crop=predicted_crop.title(), 
                                plot_path=get_relative_path(plot_path),
                                state_plot_path=get_relative_path(state_plot_path),
                                district_plot_path=get_relative_path(district_plot_path))

        except Exception as e:
            return render_template('error.html', error=str(e))

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