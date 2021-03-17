import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'Type': 1, 'Torque [Nm]': 4.5})
print(r.json())