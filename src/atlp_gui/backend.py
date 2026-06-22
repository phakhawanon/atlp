from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any
import os

# import importlib.util
# import sys
import json

from .. import atlp
# def import_from_path(module_name, absolute_path):
#     spec = importlib.util.spec_from_file_location(module_name, absolute_path)
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[module_name] = module  # optional: register it
#     spec.loader.exec_module(module)
#     return module

# atlp = import_from_path("atlp", "/home/o25141/auto-task-labelling-pipeline/src/atlp/__init__.py")

app = FastAPI()

# app.mount("/static", StaticFiles(directory="."), name="static")

# 1. Get the absolute path of the directory where main.py lives
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Use absolute paths for the static files and index.html
# app.mount("/static", StaticFiles(directory=CURRENT_DIR), name="static")

# Allow the frontend (localhost:5500) to call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve video files as static files from a folder called "media"
# http://localhost:8000/media/video1.mp4  etc.
# os.makedirs("media", exist_ok=True)
# app.mount("/media", StaticFiles(directory="media"), name="media")


# ── REQUEST MODELS ────────────────────────────────────────────
class DirectoryRequest(BaseModel):
    path: str
    filterOptions: dict

class DatapointRequest(BaseModel):
    path: str
    datapoint: str

class SaveJsonRequest(BaseModel):
    model_data: dict[str, Any]
    actual_data: dict[str, Any]
    datapoint: str


# ── ENDPOINT 1: Load directory ────────────────────────────────
@app.post("/load-datapoint")
async def load_datapoint(req: DatapointRequest):
    path = req.path
    datapoint = req.datapoint

    # TODO: replace this block with your real logic
    # e.g. scan the directory, find video files, load JSON sidecar files
    # files = os.listdir(path) if os.path.isdir(path) else []

    data = atlp.load_data()
    # model_json = json.dumps(data["datapoints"][datapoint]["model"]) 
    # actual_json = json.dumps(data["datapoints"][datapoint]["actual"]) 
    # print(model_json)
    # print(actual_json)
    return {
        # "output": [f"Found: {f}" for f in files],   # list of strings → Output log
        "video1": f"{path}/{datapoint}/camera_left_wrist.mp4",               # path served by /media
        "video2": f"{path}/{datapoint}/camera_front_head_rgb.mp4",
        "video3": f"{path}/{datapoint}/camera_right_wrist.mp4",
        "json1":  atlp.get_datapoint(datapoint, use_dict=data, get_all=True, from_actual=False),  # any dict
        "json2":  atlp.get_datapoint(datapoint, use_dict=data, get_all=True, from_actual=True),
    }

@app.post("/load-directory")
async def load_directory(req: DirectoryRequest):
    path = req.path
    filter = req.filterOptions
    atlp.set_root(path)
    atlp.populate_header()
    return {
        "datapoints": atlp.filter(**filter),
        "model_tag_list": atlp.tag_get(),
        "actual_tag_list": atlp.tag_get(from_actual=True),
    }

# ── ENDPOINT 2: Save JSON ─────────────────────────────────────
@app.post("/save-json")
async def save_json(req: SaveJsonRequest):
    print("Saving JSON:")
    print(req.actual_data)

    # TODO: persist req.data however you need
    # e.g. write to a file, update a database, etc.

    model_data = req.model_data
    actual_data = req.actual_data
    datapoint = req.datapoint

    atlp.modify_datapoint(datapoint,
                          labels=model_data["label"],
                          visions=model_data["vision"],
                          motions=model_data["motion"],
                          )
    atlp.modify_datapoint(datapoint,
                          labels=actual_data["label"],
                          visions=actual_data["vision"],
                          motions=actual_data["motion"],
                          modify_actual=True,
                          )


    return {"status": "ok"}

from fastapi.responses import FileResponse

@app.get("/video")
async def get_video(path: str):
    # path is the absolute path on your machine, e.g. /home/user/data/clip.mp4
    if not os.path.isfile(path):
        return {"error": f"File not found at path {path}"}
    return FileResponse(path, media_type="video/mp4")
