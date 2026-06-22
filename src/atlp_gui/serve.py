#!/usr/bin/env python3
"""
serve.py  –  Serves the frontend on http://localhost:5500

Run with:
    python serve.py

The server adds CORS headers so the frontend can also reach your
Python backend at a different port (default assumed: 8000).
"""

import http.server
import socketserver
import os

PORT = 5500   # ← EDIT: change frontend port if needed

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} – {fmt % args}")


os.chdir(os.path.dirname(os.path.abspath(__file__)))

socketserver.TCPServer.allow_reuse_address = True

with socketserver.TCPServer(("", PORT), CORSHandler) as httpd:
    print(f"\n  Video Inspector frontend")
    print(f"  ➜  http://localhost:{PORT}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
