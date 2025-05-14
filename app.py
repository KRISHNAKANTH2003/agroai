from flask import Flask, render_template, request
import pickle 
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (non-GUI)

app = Flask(__name__)

# Load the model and other necessary files
model = joblib.load('C://Users//KRISHNA KANTH//Desktop//project final pro//model//model.pkl')
label_encoder = joblib.load('C://Users//KRISHNA KANTH//Desktop//project final pro//model//label_encoder.pkl')
scaler = joblib.load('C://Users//KRISHNA KANTH//Desktop//project final pro//model//scaler.pkl')
df = pd.read_csv('C://Users//KRISHNA KANTH//Desktop//project final pro//model//Crop_Production_data.csv')

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

@app.route('/')
def home():
    return render_template('index.html')

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
        plot_path = 'C://Users//KRISHNA KANTH//Desktop//project final pro//static//barplot.png'
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
        state_plot_path = 'C://Users//KRISHNA KANTH//Desktop//project final pro//static//state_demand.png'
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
        district_plot_path = 'C://Users//KRISHNA KANTH//Desktop//project final pro//static//district_demand.png'
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

if __name__ == '__main__':
    app.run(debug=True)