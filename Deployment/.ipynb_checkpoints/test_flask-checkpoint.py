import requests

url = "http://127.0.0.1:5000/predict"

data = {"features": [6,148,72,35,0,33.6,0.627,50]}

response = requests.post(url,json=data)

print(response.json())