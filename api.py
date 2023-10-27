from os.path import join

import json
from PIL import Image
from loguru import logger
import sys

from fastapi.middleware.cors import CORSMiddleware

from io import BytesIO

import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, WebSocket, HTTPException, status
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel
from detector import video_detector, image_detector, webcam_detector

from util import bytes_to_image, image_to_bytes

api = FastAPI(title="Road Scan API")

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
    title="Object Detection FastAPI Template",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="2023.1.31",
)

# This function is needed if you want to allow client requests
# from specific domains (specified in the origins argument)
# to access resources from the FastAPI server,
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def save_openapi_json():
    """This function is used to save the OpenAPI documentation
    data of the FastAPI application to a JSON file.
    The purpose of saving the OpenAPI documentation data is to have
    a permanent and offline record of the API specification,
    which can be used for documentation purposes or
    to generate client libraries. It is not necessarily needed,
    but can be helpful in certain scenarios."""
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)


# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    """
    It basically sends a GET request to the route & hopes to get a "200"
    response code. Failing to return a 200 response code just enables
    the GitHub Actions to rollback to the last version the project was
    found in a "working condition". It acts as a last line of defense in
    case something goes south.
    Additionally, it also returns a JSON response in the form of:
    {
        'healtcheck': 'Everything OK!'
    }
    """
    return {'healthcheck': 'Everything OK!'}


class DetectRequest(BaseModel):
    source: UploadFile


@api.get("/", include_in_schema=False, tags=["General"])
async def index(request: Request):
    return RedirectResponse(join(request.url.path, "docs"))


@api.post("/detect_from_image", tags=["Detection"])
async def run(image: bytes = File(...)):
    request_image = bytes_to_image(image)

    detected_image = image_detector(request_image)

    return StreamingResponse(content=image_to_bytes(detected_image), media_type="image/jpeg")
