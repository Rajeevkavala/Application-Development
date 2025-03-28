import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def load_data():
    df = pd.read_csv("data/data.csv")  # Update with actual dataset

    numeric_features = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms',
                        'energy', 'instrumentalness', 'key', 'liveness', 'loudness',
                        'mode', 'popularity', 'speechiness', 'tempo']

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_features]), columns=numeric_features)

    # Train Nearest Neighbors model
    neigh = NearestNeighbors(n_neighbors=6, metric='cosine')
    neigh.fit(df_scaled)

    return df, df_scaled, neigh


def get_recommendations(song_name, df, df_scaled, neigh, num_recommendations=5):
    song_idx = df[df["name"] == song_name].index

    if len(song_idx) == 0:
        return None

    song_idx = song_idx[0]
    song_features = df_scaled.iloc[song_idx].values.reshape(1, -1)
    distances, indices = neigh.kneighbors(song_features)

    recommended_indices = indices[0][1:num_recommendations + 1]
    recommendations = df.iloc[recommended_indices][["name", "artists", "year"]]
    recommendations["similarity"] = 1 - distances[0][1:num_recommendations + 1]

    return recommendations.to_dict(orient="records")


def get_featured_songs(df):
    try:
        # Define keywords for devotional and relaxation songs
        keywords = ['meditation', 'relaxation', 'spiritual', 'devotional', 'peaceful', 'calm', 'zen']
        
        # Filter songs based on keywords in name or low energy and high acousticness
        featured = df[
            (df['name'].str.lower().str.contains('|'.join(keywords), na=False)) |
            ((df['energy'] < 0.4) & (df['acousticness'] > 0.6))
        ]
        
        # If we don't get enough songs, just get some popular acoustic songs
        if len(featured) < 6:
            featured = df[
                (df['acousticness'] > 0.5)
            ].sort_values('popularity', ascending=False).head(6)
        
        # Sort by popularity and get top 6 songs
        featured = featured.sort_values('popularity', ascending=False).head(6)
        
        # Convert artists from string representation to actual string if needed
        if 'artists' in featured.columns:
            featured['artists'] = featured['artists'].apply(lambda x: x.strip("[]'").split(',')[0] if isinstance(x, str) else x)
        
        return featured[['name', 'artists']].to_dict(orient='records')
    except Exception as e:
        print(f"Error in get_featured_songs: {e}")
        return []
