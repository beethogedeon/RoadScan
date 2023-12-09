from os.path import join

import json
import cv2
from PIL import Image
from loguru import logger
import sys

#  from fastapi.middleware.cors import CORSMiddleware

from io import BytesIO

import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, WebSocket, HTTPException, status
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel
from detector import video_detector, image_detector, webcam_detector

from util import bytes_to_image, image_to_bytes

####################################### logger #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI Setup #############################

# title
app = FastAPI(
    title="RoadScan",
    description="""Road damage detection and tracking""",
    version="0.0.1",
)

@app.on_event("startup")
def save_openapi_json():
    """Save the OpenAPI documentation
    data of the FastAPI application to a JSON file."""
    openapi_data = app.openapi()

    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)


# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


class DetectRequest(BaseModel):
    source: UploadFile


@app.get("/", include_in_schema=False, tags=["General"])
async def index(request: Request):
    return RedirectResponse(join(request.url.path, "docs"))


@app.post("/detect_from_image", tags=["Detection"])
async def run(image: bytes = File(...)):
    request_image = bytes_to_image(image)

    detected_image = image_detector(request_image)

    return StreamingResponse(content=image_to_bytes(detected_image), media_type="image/jpeg")


@app.post("/detect_from_video", tags=["Detection"])
async def run(video: UploadFile = File(...)):
    video_url = video.file.read()

    return StreamingResponse(video_detector(video_url), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/detect_from_webcam", tags=["Detection"])
async def run(index: int = 0):
    return StreamingResponse(webcam_detector(index), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/detect_from_video_ws")
async def run(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            video = await websocket.receive_bytes()
            arr = np.frombuffer(video, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            frame = video_detector(video)
            await websocket.send(frame)
        except Exception as e:
            logger.error(e)
            break

    await websocket.close()


@app.websocket("/detect_from_webcam_ws")
async def run(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            frame = webcam_detector(0)
            await websocket.send(frame)
        except Exception as e:
            logger.error(e)
            break

    await websocket.close()
