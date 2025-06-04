import onnxruntime_genai as og

from flask import Flask , request ,jsonify
from waitress import serve
app = Flask(__name__)
import threading

import os

# Add a lock for thread-safe model access
model_lock = threading.Lock()

model_path = os.getenv("MODEL_PATH") if os.getenv("MODEL_PATH") else "/models/phi3-mini-4k-onnx/"


def getPrompt(question):
    return f"<|system|>You are a helpful AI assistant.<|end|><|user|>{question}<|end|><|assistant|>"
    

model = og.Model(model_path)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()
search_options = {"max_length": 1024,"temperature":0.6}


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

    # Use lock to ensure only one request processes at a time
    with model_lock:
        params = og.GeneratorParams(model)
        params.try_use_cuda_graph_with_max_batch_size(0)
        params.set_search_options(**search_options)
        input_tokens = tokenizer.encode(getPrompt(question))
        params.input_ids = input_tokens

        generator = og.Generator(model, params)
        output = ""
        while not generator.is_done():
                        generator.compute_logits()
                        generator.generate_next_token()

                        new_token = generator.get_next_tokens()[0]
                        # print(tokenizer_stream.decode(new_token), end='', flush=True)
                        output += tokenizer_stream.decode(new_token)

    
    response = jsonify(
        {"message": "success", "answer": output}
    )
    response.status_code = 200
    
    
    return response


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000, backlog=10, connection_limit=10)
    print("Server started at port 5000")
