from flask import Flask, render_template, request, url_for
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import sys
import traceback

# Configure matplotlib
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
DATA_PATH = MODEL_DIR / 'Crop_Production_data.csv'

# Initialize variables
model = None
label_encoder = None
scaler = None
df = None

try:
    # Load required files
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)
    
    # Data cleaning
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
    
    # Filter to recommended crops
    recommended_crops = [
        'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
        'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
        'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
        'coconut', 'cotton', 'jute', 'coffee'
    ]
    df = df[df['Crop'].isin(recommended_crops)]

except Exception as e:
    print(f"Initialization error: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)

def generate_filename(prefix):
    """Generate unique filename with timestamp"""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            # Validate system is ready
            if None in (model, label_encoder, scaler, df):
                return render_template('error.html', 
                                    error="System not properly initialized"), 500

            # Get form data with defaults
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
            except ValueError:
                return render_template('error.html',
                                    error="Invalid input values"), 400

            # Prepare and scale input
            sample_input = np.array([list(form_data.values())])
            try:
                sample_input = scaler.transform(sample_input)
            except Exception as e:
                return render_template('error.html',
                                    error="Error scaling input data"), 500

            # Predict probabilities
            try:
                probabilities = model.predict_proba(sample_input)[0]
                classes = model.classes_
                crop_names = label_encoder.inverse_transform(classes)
            except Exception as e:
                return render_template('error.html',
                                    error="Error making prediction"), 500

            # Sort results
            sorted_indices = np.argsort(probabilities)[::-1]
            crop_names_sorted = crop_names[sorted_indices]
            probabilities_sorted = probabilities[sorted_indices]
            predicted_crop = crop_names_sorted[0].lower()

            # Generate plots
            prob_filename = generate_filename('prob')
            state_filename = generate_filename('state')
            district_filename = generate_filename('district')

            # Probability plot
            try:
                plt.figure(figsize=(10, 6))
                plt.barh(crop_names_sorted[:10], probabilities_sorted[:10], color='green')
                plt.xlabel("Probability")
                plt.title("Top 10 Recommended Crops")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(STATIC_DIR / prob_filename, bbox_inches='tight')
                plt.close()
            except Exception as e:
                prob_filename = None
                print(f"Error generating probability plot: {str(e)}")

            # State and district plots
            crop_data = df[df['Crop'] == predicted_crop]
            if not crop_data.empty:
                try:
                    # State plot
                    state_data = crop_data.groupby('State_Name')['Production'].sum().nlargest(10).reset_index()
                    plt.figure(figsize=(12, 6))
                    sns.barplot(data=state_data, x='State_Name', y='Production', palette='viridis')
                    plt.xticks(rotation=45)
                    plt.title(f"Top States for {predicted_crop.title()}")
                    plt.tight_layout()
                    plt.savefig(STATIC_DIR / state_filename, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    state_filename = None
                    print(f"Error generating state plot: {str(e)}")

                try:
                    # District plot
                    district_data = crop_data.groupby(['State_Name', 'District_Name'])['Production'].sum().nlargest(10).reset_index()
                    district_data['Label'] = district_data['District_Name'] + ' (' + district_data['State_Name'] + ')'
                    plt.figure(figsize=(12, 6))
                    sns.barplot(data=district_data, x='Label', y='Production', palette='rocket')
                    plt.xticks(rotation=45)
                    plt.title(f"Top Districts for {predicted_crop.title()}")
                    plt.tight_layout()
                    plt.savefig(STATIC_DIR / district_filename, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    district_filename = None
                    print(f"Error generating district plot: {str(e)}")
            else:
                state_filename = None
                district_filename = None

            return render_template('result.html',
                                crop=predicted_crop.title(),
                                prob_filename=prob_filename,
                                state_filename=state_filename,
                                district_filename=district_filename)

        except Exception as e:
            print(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            return render_template('error.html',
                                error="An unexpected error occurred"), 500

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

# Only for development (Not used in production with gunicorn)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
