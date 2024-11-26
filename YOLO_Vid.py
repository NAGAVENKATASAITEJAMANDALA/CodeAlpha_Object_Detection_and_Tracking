from ultralytics import YOLO
import cv2

# Define the path to the video file
video_path = r"D:\sai\code alpha\task3\car-detection.mp4"  # Use a raw string to handle backslashes

# Load the YOLO model from Ultralytics
model = YOLO('yolov3.pt')  # Ensure the model file is in the correct directory or provide the full path

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    raise FileNotFoundError("Video file not found. Check the file path.")

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break  # Exit loop if there are no frames left to process

    # Perform detection on the current frame
    results = model(frame)  # YOLO model performs detection

    # Annotate the frame with detected objects
    annotated_frame = results[0].plot()  # Add bounding boxes and labels

    # Display the annotated frame
    cv2.imshow("Detected Video", annotated_frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Video display interrupted by user.")
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
