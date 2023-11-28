import requests

data = {'file': open('/home/chg/Videos/ld/2012.mp4', 'rb')}

resp = requests.post("http://localhost:5000/predict", files=data)
print(resp.text)