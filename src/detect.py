import cv2
import numpy as np

from ultralytics import YOLOv10


def detect_helmet(image):
	PATH_MODEL_DETECT = "runs/detect/train2/weights/best.pt"
	model = YOLOv10(PATH_MODEL_DETECT)
	results = model.predict(image)
	annotated_image = results[0].plot()

	has_helmet = any(result.cls.item() == 1 for result in results[0].boxes)
	return annotated_image, has_helmet


def process_image(image):
	img_array = np.frombuffer(image.read(), np.uint8)
	img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
	img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	return img_rgb
