import os
import requests
from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv("PERSPECTIVE_API_KEY")

url = f"https://commentanalyzer.googleapis.com/v1/comments:analyze?key={KEY}"

payload = {
    "comment": {"text": "You are awesome!"},
    "languages": ["en"],
    "requestedAttributes": {"TOXICITY": {}}
}

resp = requests.post(url, json=payload)
print("Status:", resp.status_code)
print(resp.text)
