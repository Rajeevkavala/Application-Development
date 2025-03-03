import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# Load and preprocess the dataset
@st.cache_data  # Cache the data to avoid reloading on every interaction
def load_data():
    df = pd.read_csv(r"C:\Users\rajee\Downloads\archive\data\data.csv")  # Update path as needed

    # Define numeric features for scaling
    numeric_features = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms',
                        'energy', 'instrumentalness', 'key', 'liveness', 'loudness',
                        'mode', 'popularity', 'speechiness', 'tempo']

    # Standardize numeric features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_features]), columns=numeric_features)

    # Fit NearestNeighbors model
    neigh = NearestNeighbors(n_neighbors=6, metric='cosine')  # 6 includes the input song
    neigh.fit(df_scaled)

    return df, df_scaled, neigh


# Function to get song recommendations
def get_recommendations(song_name, df, df_scaled, neigh, num_recommendations=5):
    song_idx = df[df["name"] == song_name].index

    if len(song_idx) == 0:
        return None

    song_idx = song_idx[0]
    song_features = df_scaled.iloc[song_idx].values.reshape(1, -1)

    # Find nearest neighbors
    distances, indices = neigh.kneighbors(song_features)

    # Get recommended songs (excluding the input song)
    recommended_indices = indices[0][1:num_recommendations + 1]
    recommendations = df.iloc[recommended_indices][["name", "artists", "year"]]
    recommendations["similarity"] = 1 - distances[0][1:num_recommendations + 1]

    return recommendations


# Streamlit UI
def main():
    # Load data and model
    df, df_scaled, neigh = load_data()

    # Set page title and header
    st.title("Song Recommendation System")
    st.markdown("Enter a song name to get similar song recommendations based on Spotify features.")

    # User input
    song_name = st.text_input("Enter Song Name", "").strip()

    # Button to get recommendations
    if st.button("Get Recommendations"):
        if song_name:
            recommendations = get_recommendations(song_name, df, df_scaled, neigh)

            if recommendations is not None:
                st.subheader(f"Recommendations for '{song_name}':")
                st.dataframe(recommendations.style.format({"similarity": "{:.4f}"}))
            else:
                st.error(f"Song '{song_name}' not found in the dataset.")
        else:
            st.warning("Please enter a song name.")

    # Exit option
    if st.button("Exit"):
        st.write("Thank you for using the Song Recommendation System!")
        st.stop()


if __name__ == "__main__":
    main()