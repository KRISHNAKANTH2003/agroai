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

# Configuration and initialization code remains the same until the generate_crop_graph function

def generate_crop_graph(crop_name):
    """Generate a graph showing production data for the recommended crop"""
    try:
        # Filter data for the recommended crop (case-insensitive match)
        crop_data = df[df['Crop'].str.lower() == crop_name.lower()]
        
        if crop_data.empty:
            print(f"No data found for crop: {crop_name}")
            return None
            
        # Clean old plots
        clean_old_plots()
        
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = STATIC_DIR / f'crop_plot_{timestamp}.png'
        
        # Create figure with subplots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Production by State (Top 10)
        plt.subplot(2, 2, 1)
        state_production = crop_data.groupby('State_Name')['Production'].sum().nlargest(10)
        state_production.plot(kind='bar', color='green')
        plt.title(f'Top 10 States for {crop_name.title()} Production')
        plt.ylabel('Production (tons)')
        plt.xticks(rotation=45)
        
        # Plot 2: Area vs Production
        plt.subplot(2, 2, 2)
        plt.scatter(crop_data['Area'], crop_data['Production'], alpha=0.5, color='blue')
        plt.title(f'Area vs Production for {crop_name.title()}')
        plt.xlabel('Area (hectares)')
        plt.ylabel('Production (tons)')
        
        # Plot 3: Season-wise distribution
        plt.subplot(2, 2, 3)
        season_counts = crop_data['Season'].value_counts()
        season_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title(f'Season Distribution for {crop_name.title()}')
        
        # Plot 4: Year-wise trend (if available)
        plt.subplot(2, 2, 4)
        if 'Crop_Year' in crop_data.columns:
            year_production = crop_data.groupby('Crop_Year')['Production'].sum()
            year_production.plot(kind='line', marker='o', color='red')
            plt.title(f'Year-wise Production Trend for {crop_name.title()}')
            plt.xlabel('Year')
            plt.ylabel('Production (tons)')
        else:
            # Alternative plot if year data not available
            district_production = crop_data.groupby('District_Name')['Production'].sum().nlargest(10)
            district_production.plot(kind='bar', color='purple')
            plt.title(f'Top 10 Districts for {crop_name.title()} Production')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        return plot_path.relative_to(BASE_DIR).as_posix()
    
    except Exception as e:
        print(f"Error generating crop graph for {crop_name}: {str(e)}", file=sys.stderr)
        return None

# The rest of your Flask routes remain the same, but ensure the recommend route uses the correct crop name:

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            # [Previous code for getting form data and making prediction...]
            
            # Get the recommended crop (ensure lowercase for matching)
            predicted_crop = crop_names_sorted[0].lower()
            print(f"Recommended crop: {predicted_crop}")  # Debug logging
            
            # Generate crop-specific graph - ensure we're passing the correct crop name
            plot_path = generate_crop_graph(predicted_crop)
            
            # [Rest of your code...]
            
            return render_template('result.html',
                                crop=predicted_crop.title(),
                                plot_path=plot_path,
                                prob_plot_path=prob_plot_path,
                                probabilities=dict(zip(crop_names_sorted, probabilities_sorted)))

        except Exception as e:
            print(f"Recommendation error: {str(e)}", file=sys.stderr)
            return render_template('error.html',
                                error="Sorry, we couldn't process your request. Please try again."), 500

    return render_template('recommend.html')