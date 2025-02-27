import requests
import argparse

arg_p = argparse.ArgumentParser()
arg_p.add_argument("-i", "--image", required=True, 
                    help="Name of the image file to be classified") 

args = vars(arg_p.parse_args()) 

# REST_API_URL = "http://localhost:5000/predict"
#REST_API_URL = "http://10.244.0.119:8080/predict"
GATEWAY = "http://10.244.1.60:8080"
REST_API_URL = GATEWAY + "/function/test-fastpod/predict"
IMAGE_PATH = args["image"]

image = open(IMAGE_PATH, "rb")
payload = {"payload": image}

r = requests.post(REST_API_URL, files=payload).json()

if r["success"]:
    print(r)
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"],
            result["probability"]))
else:
    print(r)
    print("Failed request")
