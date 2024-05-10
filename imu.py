import cv2

def detect_and_identify_objects(video_path):
  """
  This function detects moving objects in a video and optionally identifies them using contours.

  Args:
      video_path: Path to the video file.

  Returns:
      None
  """

  # Capture video
  cap = cv2.VideoCapture(video_path)

  # Background subtraction using MOG algorithm
#   bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
  car_cascade = cv2.CascadeClassifier("haarcascade_car.xml")
  while True:
    ret, frame = cap.read()
    
    if not ret:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect cars using the cascade classifier
    cars = car_cascade.detectMultiScale(gray, 1.1, 4)

    #--------------------------------------------------------------------    

    # Process only one car (if detected)
    # if len(cars) == 1:
    #   for (x, y, w, h) in cars:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw bounding box
    #     # break  # Exit the inner loop after finding one car
      
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Draw bounding box
        break
    

    cv2.imshow("Frame", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Release resources
  cap.release()
  cv2.destroyAllWindows()

# Example usage
video_path = "car.mp4"  # Replace with your video filename
detect_and_identify_objects(video_path)
