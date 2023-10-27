import cv2

from imread_from_url import imread_from_url
from cap_from_youtube import cap_from_youtube

from yolov8 import YOLOv8

import numpy as np

from PIL import Image

# Initialize yolov8 object detector
model_path = "weights/detector.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)


# Read image

def image_detector(image):
    try:
        try:
            if str(image).startswith("http"):
                img = imread_from_url(image)
            else:
                img = np.asarray(image)
        except Exception as e:
            raise ValueError("Error while reading Image", e)

        # Detect Objects
        try:
            boxes, scores, class_ids = yolov8_detector(img)
        except Exception as e:
            raise ValueError("Error while detecting", e)

        # Draw detections
        try:
            combined_img = yolov8_detector.draw_detections(img)
        except Exception as e:
            raise ValueError("Error while drawing")
        #    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
        #    cv2.imshow("Detected Objects", combined_img)
        image_outpath = "doc/img/detected_image.jpg"
        detected_image = Image.fromarray(combined_img, 'RGB')
        # cv2.imwrite(image_outpath, combined_img)
        detected_image.save(image_outpath)

    except Exception as e:
        raise ValueError("Error while detecting on image :", e)

    return image_outpath


#    cv2.waitKey(0)


def video_detector(video_url):
    if "https://youtu" in video_url:
        cap = cap_from_youtube(video_url, resolution='720p')
    else:
        cap = cv2.VideoCapture(video_url)

    start_time = 1  # skip first {start_time} seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

    # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (3840, 2160))

    #    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    while cap.isOpened():

        # Press key q to stop
        #        if cv2.waitKey(1) == ord('q'):
        #            break

        try:
            # Read frame from the video
            ret, frame = cap.read()
            if not ret:
                break
        except Exception as e:
            print(e)
            continue

        # Update object localizer
        boxes, scores, class_ids = yolov8_detector(frame)

        combined_img = yolov8_detector.draw_detections(frame)
        # cv2.imshow("Detected Objects", combined_img)
        # out.write(combined_img)
        _, buffer = cv2.imencode(".jpg", combined_img)

        IMAGE_HEADER = (b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n")

        video = IMAGE_HEADER + buffer.tobytes() + b"\r\n"

        yield video


def webcam_detector(index):
    cap = cv2.VideoCapture(index)

    start_time = 1  # skip first {start_time} seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

    # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (3840, 2160))

    #    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    while cap.isOpened():

        # Press key q to stop
        #        if cv2.waitKey(1) == ord('q'):
        #            break

        try:
            # Read frame from the video
            ret, frame = cap.read()
            if not ret:
                break
        except Exception as e:
            print(e)
            continue

        # Update object localizer
        boxes, scores, class_ids = yolov8_detector(frame)

        combined_img = yolov8_detector.draw_detections(frame)
        # cv2.imshow("Detected Objects", combined_img)
        # out.write(combined_img)

        _, buffer = cv2.imencode(".jpg", combined_img)

        IMAGE_HEADER = (b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n")

        video = IMAGE_HEADER + buffer.tobytes() + b"\r\n"

        yield video
