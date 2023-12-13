import cv2
import dlib
import os
import time

# Load the dlib shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

glips_dataset_path = "Path-to-dataset"
filename = "filename"
split = "train-test-val" # the split of data you want to check
video_filename = "video-file-name.mp4" 
video_path = os.path.join(glips_dataset_path,filename,split,video_filename)

# Load the video
cap = cv2.VideoCapture(video_path)

ct = 0
while True:
    # Capture the current frame
    ret, frame = cap.read()
    ct+=1

    # Check if the end of the video has been reached
    if not ret:
        cap.release()
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray, 1)
    print(str(ct) + "-> "+str(len(faces)))

    for face in faces:
        # Get the landmarks for each face
        landmarks = predictor(frame, face)

        # Draw landmarks on the frame
        for landmark in landmarks.parts():
            cv2.circle(frame, (landmark.x, landmark.y), 5, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('Frame', frame)
    time.sleep(0.2)

    # Check if the ESC key was pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
