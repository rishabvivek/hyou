import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import dotenv
import os
import requests
import base64
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import csv
import lyricsgenius
import sys
import pandas as pd
csv.field_size_limit(sys.maxsize)



dotenv.load_dotenv()

# client_id = os.getenv("SPOTIFY_CLIENT")
# client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

# genius_token = os.getenv("GENIUS_ACCESS_TOKEN")

# genius = lyricsgenius.Genius(genius_token)

# # Create a SpotifyClientCredentials object
# credentials = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
# sp = spotipy.Spotify(client_credentials_manager=credentials)

# # Prepare the authentication request
# url = "https://accounts.spotify.com/api/token"
# auth_string = f"{client_id}:{client_secret}"
# auth_bytes = auth_string.encode("utf-8")
# auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")
# headers = {
#     "Authorization": "Basic " + auth_base64,
#     "Content-Type": "application/x-www-form-urlencoded"
# }
# data = {"grant_type": "client_credentials"}

# # Send the authentication request
# response = requests.post(url, headers=headers, data=data)

# if response.status_code == 200:
#     json_res = response.json()
#     access_token = json_res["access_token"]

#     # Set the headers for the API request
#     api_headers = {
#         "Authorization": f"Bearer {access_token}",
#         "Content-Type": "application/json"
#     }

#     playlist_id = "5itomqfhcuifgVISP1u4Vl"
#     track_names = []
#     tracks_data = []

#     playlist_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
#     playlist_response = requests.get(playlist_url, headers=api_headers)

#     if playlist_response.status_code == 200:
#         playlist_info = playlist_response.json()
#         tracks = playlist_info.get("items", [])

#         # Extract the track names from the initial response
#         for track in tracks:
#             track_names.append(track["track"]["name"])

#         # Get all tracks until all names are retrieved
#         while "next" in playlist_info and playlist_info["next"]:
#             playlist_url = playlist_info["next"]
#             playlist_response = requests.get(playlist_url, headers=api_headers)

#             if playlist_response.status_code == 200:
#                 playlist_info = playlist_response.json()
#                 tracks = playlist_info.get("items", [])

#                 for track in tracks:
#                     track_names.append(track["track"]["name"])
#             else:
#                 print(f"Error: {playlist_response.status_code} - {playlist_response.text}")
#                 break
        
#         # print(len(track_names))
#         track_ids = []
#         for track_name in track_names:
#             try:
#                 results = sp.search(q=track_name, type="track", limit=1)
#                 if results["tracks"]["items"]:
#                     track_id = results["tracks"]["items"][0]["id"]
#                     track_ids.append(track_id)
#                 else:
#                     print(f"No track found for '{track_name}'")
#             except Exception as e:
#                 print(f"Error occurred while searching for '{track_name}': {str(e)}")
        

#         track_ids_str = ",".join(track_ids)

#         max_tracks_per_request = 100

#         # Fetch audio features in chunks
#         audio_features = []
#         for i in range(0, len(track_ids), max_tracks_per_request):
#             chunk_ids = track_ids[i: i + max_tracks_per_request]
#             chunk_ids_str = ",".join(chunk_ids)
#             features_response = requests.get(
#                 "https://api.spotify.com/v1/audio-features",
#                 headers=api_headers,
#                 params={"ids": chunk_ids_str}  # Pass the track IDs as query parameters
#             )
#             if features_response.status_code == 200:
#                 features_info = features_response.json()
#                 features = features_info.get("audio_features", [])
#                 audio_features.extend(features)
#             else:
#                 print(f"Error: {features_response.status_code} - {features_response.text}")

        

#         # Process audio features data
#         for i, feature in enumerate(audio_features):
#             track_name = track_names[i]
            
#             acousticness = feature["acousticness"]
#             danceability = feature["danceability"]
#             energy = feature["energy"]
#             instrumentalness = feature["instrumentalness"]
#             loudness = feature["loudness"]
#             tempo = feature["tempo"]
#             valence = feature["valence"]
            

#             track_data = {
#                 "track_name": track_name,
#                 "acousticness": acousticness,
#                 "danceability": danceability,
#                 "energy": energy,
#                 "instrumentalness": instrumentalness,
#                 "loudness": loudness,
#                 "tempo": tempo,
#                 "valence": valence,
#             }
#             tracks_data.append(track_data)

#         numerical_features = ["acousticness", "danceability", "tempo", "instrumentalness", "energy", "loudness", "valence"]
#         data = [[track_data[feature] for feature in numerical_features] for track_data in tracks_data]

#         # Normalizing data 
#         scaler = MinMaxScaler()
#         normalized_data = scaler.fit_transform(data)
#         for i, track_data in enumerate(tracks_data):
#             for j, feature in enumerate(numerical_features):
#                 track_data[feature] = normalized_data[i][j]

        

#         # # Save data to CSV file
#         # filename = "test_set.csv"
#         # fieldnames = tracks_data[0].keys()
#         # # Write data to CSV file
#         # with open(filename, mode="w", newline="") as file:
#         #     writer = csv.DictWriter(file, fieldnames=fieldnames)
#         #     writer.writeheader()
#         #     for track_data in tracks_data:
#         #         writer.writerow(track_data)

#         # print(f"Data saved to {filename}")
        
#     else:
#         print(f"Error: {playlist_response.status_code} - {playlist_response.text}")

# else:
#     print(f"Error: {response.status_code} - {response.text}")



#NLP SECTION
lyrics_data = []

# for track_name in track_names:
#     try:
#         # Search for the track on Genius
#         song = genius.search_song(track_name)
#         if song:
#             track_lyrics = song.lyrics
#             lyrics_data.append({"track_name": track_name, "lyrics": track_lyrics})
#         else:
#             print(f"No lyrics found for '{track_name}'")
#     except Exception as e:
#         print(f"Error occurred while fetching lyrics for '{track_name}': {str(e)}")

# # Save data to CSV file
filename = "tracks_lyrics.csv"
# fieldnames = ["track_name", "lyrics"]

# with open(filename, mode="w", newline="") as file:
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(lyrics_data)

# with open(filename, mode="r") as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         lyrics_data.append(row)
# # Preprocess the lyrics data
# lemmatizer = WordNetLemmatizer()
# stopwords = set(stopwords.words("english"))

# for data in lyrics_data:
#     lyrics = data["lyrics"]

#     # Clean the text: remove special characters and symbols
#     lyrics = re.sub(r"[^A-Za-z0-9\s]", "", lyrics)

#     # Tokenize the lyrics: split into words/tokens
#     tokens = word_tokenize(lyrics)

#     # Normalize the text: convert to lowercase and remove stopwords
#     normalized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stopwords]

#     # Update the lyrics data with preprocessed lyrics
#     data["preprocessed_lyrics"] = " ".join(normalized_tokens)



train_data = pd.read_csv('training_set.csv')

X_train = train_data.drop(['track_name', 'happiness', 'sadness', 'love', 'calm', 'energetic'], axis=1)
y_train = train_data[['happiness', 'sadness', 'love', 'calm', 'energetic']]


X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Model
input_size = X_train.shape[1]
hidden_size = 512
output_size = y_train.shape[1]
model = MultiLabelClassifier(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Number of folds for cross-validation
num_folds = 25

# Perform k-fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True)

accuracies = []
precisions = []
recalls = []
f1_scores = []
threshold = 0.37
for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    # print(f"Fold {fold+1}/{num_folds}")

    X_train_fold, X_val_fold = X_train_tensor[train_index], X_train_tensor[val_index]
    y_train_fold, y_val_fold = y_train_tensor[train_index], y_train_tensor[val_index]

    # Training loop
    num_epochs = 100
    batch_size = 3
    for epoch in range(num_epochs):
        for i in range(0, X_train_fold.size(0), batch_size):
            inputs = X_train_fold[i:i+batch_size]
            labels = y_train_fold[i:i+batch_size]

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    with torch.no_grad():
        val_outputs = model(X_val_fold)
        val_predictions = (val_outputs > threshold).numpy().astype(int)
        val_accuracy = (val_predictions == y_val_fold.numpy()).mean()
        accuracies.append(val_accuracy)
        # print(f"Validation Accuracy: {val_accuracy}")
    
    val_predictions = val_predictions.astype(int)
    y_val_fold_numpy = y_val_fold.numpy().astype(int)

    # Precision, recall, and F1 score
    precision = precision_score(y_val_fold_numpy, val_predictions, average='micro')
    recall = recall_score(y_val_fold_numpy, val_predictions, average='micro')
    f1 = f1_score(y_val_fold_numpy, val_predictions, average='micro')

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")

# Average accuracy, precision, recall, and F1 score
average_accuracy = sum(accuracies) / num_folds
average_precision = sum(precisions) / num_folds
average_recall = sum(recalls) / num_folds
average_f1 = sum(f1_scores) / num_folds

print(f"Threshold : {threshold}")
print(f"Average Accuracy: {average_accuracy}")
print(f"Average Precision: {average_precision}")
print(f"Average Recall: {average_recall}")
print(f"Average F1 Score: {average_f1}")
