import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors



@st.cache_data 
def load_data():
    df = pd.read_csv(r"C:\Users\rajee\Downloads\archive\data\data.csv")  # Update path as needed

    numeric_features = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms',
                        'energy', 'instrumentalness', 'key', 'liveness', 'loudness',
                        'mode', 'popularity', 'speechiness', 'tempo']

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_features]), columns=numeric_features)

    # Fit NearestNeighbors model
    neigh = NearestNeighbors(n_neighbors=6, metric='cosine')  # 6 includes the input song
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

    return recommendations


# Streamlit UI
def main():
    df, df_scaled, neigh = load_data()

    st.title("Song Recommendation System")
    st.markdown("Enter a song name to get similar song recommendations based on Spotify features.")

    song_name = st.text_input("Enter Song Name", "").strip()

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