import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import dotenv
import os
import requests
import base64
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np

dotenv.load_dotenv()

# Get your client ID and client secret from the Spotify Developer Dashboard
client_id = os.getenv("SPOTIFY_CLIENT")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

# Create a SpotifyClientCredentials object
credentials = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=credentials)

# Prepare the authentication request
url = "https://accounts.spotify.com/api/token"
auth_string = f"{client_id}:{client_secret}"
auth_bytes = auth_string.encode("utf-8")
auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")
headers = {
    "Authorization": "Basic " + auth_base64,
    "Content-Type": "application/x-www-form-urlencoded"
}
data = {"grant_type": "client_credentials"}

# Send the authentication request
response = requests.post(url, headers=headers, data=data)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    json_res = response.json()
    access_token = json_res["access_token"]

    # Set the headers for the API request
    api_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    playlist_id = "0ai3cXbvbrJfpXBSmYe4Pl"
    track_names = []
    tracks_data = []

    # Make the initial API request to retrieve the first set of tracks
    playlist_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    playlist_response = requests.get(playlist_url, headers=api_headers)

    # Check if the request was successful (status code 200)
    if playlist_response.status_code == 200:
        playlist_info = playlist_response.json()
        tracks = playlist_info.get("items", [])

        # Extract the track names from the initial response
        for track in tracks:
            track_names.append(track["track"]["name"])

        # Paginate through the tracks until all names are retrieved
        while "next" in playlist_info and playlist_info["next"]:
            playlist_url = playlist_info["next"]
            playlist_response = requests.get(playlist_url, headers=api_headers)

            if playlist_response.status_code == 200:
                playlist_info = playlist_response.json()
                tracks = playlist_info.get("items", [])

                for track in tracks:
                    track_names.append(track["track"]["name"])
            else:
                print(f"Error: {playlist_response.status_code} - {playlist_response.text}")
                break
        
        # print(len(track_names))
        track_ids = []
        for track_name in track_names:
            try:
                results = sp.search(q=track_name, type="track", limit=1)
                if results["tracks"]["items"]:
                    track_id = results["tracks"]["items"][0]["id"]
                    track_ids.append(track_id)
                else:
                    print(f"No track found for '{track_name}'")
            except Exception as e:
                print(f"Error occurred while searching for '{track_name}': {str(e)}")
        

        track_ids_str = ",".join(track_ids)

        max_tracks_per_request = 100

        # Fetch audio features in chunks
        audio_features = []
        for i in range(0, len(track_ids), max_tracks_per_request):
            chunk_ids = track_ids[i: i + max_tracks_per_request]
            chunk_ids_str = ",".join(chunk_ids)
            features_response = requests.get(
                "https://api.spotify.com/v1/audio-features",
                headers=api_headers,
                params={"ids": chunk_ids_str}  # Pass the track IDs as query parameters
            )
            if features_response.status_code == 200:
                features_info = features_response.json()
                features = features_info.get("audio_features", [])
                audio_features.extend(features)
            else:
                print(f"Error: {features_response.status_code} - {features_response.text}")

        

        # Process audio features data
        for i, feature in enumerate(audio_features):
            track_name = track_names[i]
            # Access different audio features for each track
            acousticness = feature["acousticness"]
            danceability = feature["danceability"]
            energy = feature["energy"]
            instrumentalness = feature["instrumentalness"]
            loudness = feature["loudness"]
            tempo = feature["tempo"]
            valence = feature["valence"]
            # ... and other features

            track_data = {
                "track_name": track_name,
                "acousticness": acousticness,
                "danceability": danceability,
                "energy": energy,
                "instrumentalness": instrumentalness,
                "loudness": loudness,
                "tempo": tempo,
                "valence": valence,
                # ... and other features
            }
            tracks_data.append(track_data)

        numerical_features = ["acousticness", "danceability", "tempo", "instrumentalness", "energy", "loudness", "valence"]
        data = [[track_data[feature] for feature in numerical_features] for track_data in tracks_data]
        
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
        for i, track_data in enumerate(tracks_data):
            for j, feature in enumerate(numerical_features):
                track_data[feature] = normalized_data[i][j]
        

        X = np.array([[track_data[feature] for feature in numerical_features] for track_data in tracks_data])

        # Define the number of clusters (emotions)
        num_clusters = 5

        emotion_labels = {
            0: "Happy",
            1: "Sad",
            2: "Energetic",
            3: "Love",
            4: "Calm"
        }

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)

        # Print the cluster labels for each song
        for i, track_name in enumerate(track_names):
            cluster_label = cluster_labels[i]
            emotion = emotion_labels.get(cluster_label, "Unknown")
            print(f"{track_name}: Cluster {cluster_label} -     {emotion}")
        

    else:
        print(f"Error: {playlist_response.status_code} - {playlist_response.text}")

# Print the response status code and error message if the request was not successful
else:
    print(f"Error: {response.status_code} - {response.text}")
