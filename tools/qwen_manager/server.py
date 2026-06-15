import argparse
import logging
import pickle
import time
import os

from http.server import BaseHTTPRequestHandler, HTTPServer

# IMPORTANT: import your vLLM wrapper
from qwen_mm_wrapper import QwenMultimodalWrapper


# =========================
# HTTP Handler
# =========================

class ModelServerHandler(BaseHTTPRequestHandler):
    server: "ModelServer"

    def do_POST(self):
        start = time.time()

        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)

        # Expect: (args, kwargs)
        args, kwargs = pickle.loads(post_data)

        try:
            result = self.server.handle_model_request(self.path, *args, **kwargs)

            self.send_response(200)
            self.end_headers()
            self.wfile.write(pickle.dumps(result))

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(pickle.dumps({"error": str(e)}))

        self.server.logger.info(
            f"{self.path} elapsed: {time.time() - start:.3f}s"
        )


# =========================
# Model Server
# =========================

class ModelServer(HTTPServer):
    def __init__(self, server_address):
        print(f"Starting Qwen vLLM server...server address: {server_address}")
        super().__init__(server_address, ModelServerHandler)

        # Logging
        self.logger = logging.getLogger("qwen_server")
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        self.logger.addHandler(console_handler)

        # Load model once
        print("Loading Qwen vLLM model...")
        self.models = {}
        self.models["qwen_mm"] = QwenMultimodalWrapper(
            model_id=os.getenv(
                "QWEN_VL_MODEL_ID",
                "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
            )
        )
        print("Model loaded.")

    # Core processing logic
    def handle_model_request(self, url, *args, **kwargs):
        """
        URL format:
            /qwen_mm
        """
        path = url.strip("/")

        if path not in self.models:
            raise ValueError(f"Unknown model endpoint: {path}")

        model = self.models[path]

        if path == "qwen_mm":
            # texts, images, sampling_params
            return model.chat_mm(*args, **kwargs)

        raise ValueError(f"Unsupported endpoint: {path}")


# =========================
# Main
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    server = ModelServer((args.host, args.port))
    print(f"Server running at http://{args.host}:{args.port}")
    server.serve_forever()