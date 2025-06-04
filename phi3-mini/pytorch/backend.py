# Load model directly
import flask
from waitress import serve

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import threading




from flask import Flask , request ,jsonify
from waitress import serve
app = Flask(__name__)
import os

# Add a lock for thread-safe model access
model_lock = threading.Lock()

model_path = os.getenv("MODEL_PATH") if os.getenv("MODEL_PATH") else "/models/phi3-mini-4k-pytorch/"
max_token= int(os.getenv("MAX_NEW_TOKENS")) if os.getenv("MAX_NEW_TOKENS") else 4096

torch.random.manual_seed(0) 
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)



def generateMessage(message):
    
    return [ 
    {"role": "system", "content": "You are a helpful AI assistant. Always give answer in 10 word"}, 
    {"role": "user", "content": message }] 




pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 



@app.before_request
def fix_transfer_encoding():
    """
    Sets the "wsgi.input_terminated" environment flag, thus enabling
    Werkzeug to pass chunked requests as streams.  The gunicorn server
    should set this, but it's not yet been implemented.
    """
    transfer_encoding = request.headers.get("Transfer-Encoding", None)
    if transfer_encoding == "chunked":
        request.environ["wsgi.input_terminated"] = True





@app.route("/", defaults={"path": ""}, methods=["POST", "GET"])
@app.route("/<path:path>", methods=["POST", "GET"])
def main_route(path):
    if(path=='health'):
        return jsonify({"message": "success"}), 200
        
    question = request.args.get("question")

    if question == None:
        response = jsonify({"message": "invalid request"})
        response.status_code = 400
        return response
    
    generation_args = { 
    "max_new_tokens": max_token, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
    } 

    # Use lock to ensure only one request processes at a time
    with model_lock:
        output = pipe(generateMessage(question), **generation_args) 
        print('================\n')
        print(output[0]['generated_text'])

    
    response = jsonify(
        {"message": "success", "answer": output}
    )
    response.status_code = 200
    
    
    return response


if __name__ == "__main__":
    print("Server started at port 5000")
    # Reduce connection limit to prevent too many concurrent requests
    serve(app, host="0.0.0.0", port=5000, backlog=10, connection_limit=10)
