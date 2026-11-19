import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"   # must be set before importing torch/fairseq

import json, warnings
warnings.filterwarnings("ignore")
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from ai4bharat.transliteration import XlitEngine

engine = XlitEngine("hi", beam_width=10, src_script_type="en")
print("IndicXlit ready on http://127.0.0.1:5005")

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        word = parse_qs(urlparse(self.path).query).get("w", [""])[0]
        try:
            res = engine.translit_word(word, topk=1)["hi"][0]
        except Exception:
            res = ""
        body = json.dumps({"hi": res}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

HTTPServer(("127.0.0.1", 5005), H).serve_forever()
