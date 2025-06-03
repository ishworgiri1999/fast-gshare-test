import requests

# Open the file in binary mode
with open('en.wav', 'rb') as f:
    files = {'data': ('en.wav', f, 'audio/wav')}
    response = requests.post('http://localhost:5000/predict', files=files)

print('Status code:', response.status_code)
print('Response body:', response.text)
