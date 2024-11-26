from ultralytics import YOLO
import cv2

# Load the YOLO model from Ultralytics (Make sure to have the correct model path)
model = YOLO('yolov3.pt')

# Load the image
img = cv2.imread('cats.jpg')
if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# Perform detection
results = model(img)  # Automatically detects and labels objects

# Access the first result and render it
annotated_img = results[0].plot()  # 'plot()' adds bounding boxes and labels to the image

# Display the annotated image
cv2.imshow("Detected Image", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
