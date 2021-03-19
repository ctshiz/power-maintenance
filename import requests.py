import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'Torque [Nm]': 1.924952, 'Tool wear [minNm': 0.915434})
r_hdf = requests.post_hdf(url, json={'Type': 1, 'Dif temperature [K]': -8.4, 'Rotational speed [rpm]': 1363})
print(r.json())
print(r_hdf.json())
