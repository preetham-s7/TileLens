import json
from http.server import BaseHTTPRequestHandler
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from tilelens.hardware.database import HARDWARE_DATABASE
    chips_list = list(HARDWARE_DATABASE.keys())
except Exception:
    chips_list = [
        "apple_a17_pro", "snapdragon_8_gen3", "google_tensor_g4",
        "mediatek_dimensity_9300", "apple_m4", "apple_m3_max",
        "intel_lunar_lake_288v", "amd_ryzen_ai_9_hx370",
        "nvidia_rtx_4060_mobile", "nvidia_h100_sxm", "nvidia_b200"
    ]


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        data = {
            "status": "online",
            "app": "TileLens Hardware-Software Co-Design Engine",
            "version": "0.1.0",
            "total_chips": len(chips_list),
            "chips": chips_list
        }
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))
