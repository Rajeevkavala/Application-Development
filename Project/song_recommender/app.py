from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from models.recommender import get_recommendations, load_data, get_featured_songs

app = Flask(__name__)

# Load data once when the app starts
df, df_scaled, neigh = load_data()

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    recommendations = None
    
    # Get featured songs
    featured_songs = get_featured_songs(df)
    
    if request.method == 'POST':
        song_name = request.form['song_name']
        recommendations = get_recommendations(song_name, df, df_scaled, neigh)
        
        if recommendations is None:
            error = f"Sorry, couldn't find '{song_name}' in our database."
            return render_template('index.html', error=error, featured_songs=featured_songs)
        
        return render_template('results.html', recommendations=recommendations, song_name=song_name)
    
    return render_template('index.html', error=error, featured_songs=featured_songs)

if __name__ == '__main__':
    app.run(debug=True)
