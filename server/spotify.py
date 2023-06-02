import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import dotenv
import os
import requests
import base64
import json

dotenv.load_dotenv()

# Get your client ID and client secret from the Spotify Developer Dashboard
client_id = os.getenv("SPOTIFY_CLIENT")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

# Create a SpotifyClientCredentials object
credentials = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=credentials)

# Search for the track
track_name = 'Blinding Lights'
results = sp.search(q=track_name, type='track', limit=1)
track_id = results['tracks']['items'][0]['id']

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
    
    # Make the API request to retrieve the track information
    track_url = f"https://api.spotify.com/v1/tracks/{track_id}"
    track_response = requests.get(track_url, headers=api_headers)
    
    # Check if the request was successful (status code 200)
    if track_response.status_code == 200:
        track_info = track_response.json()
        preview_url = track_info["preview_url"]
        print (preview_url)
        # Download the audio sample
        # if preview_url:
        #     response = requests.get(preview_url)
        #     with open("audio_sample.mp3", "wb") as file:
        #         file.write(response.content)
        #     print("Audio sample downloaded successfully.")
        # else:
        #     print("No audio sample available for this track.")
    else:
        print(f"Error: {track_response.status_code} - {track_response.text}")

# Print the response status code and error message if the request was not successful
else:
    print(f"Error: {response.status_code} - {response.text}")
