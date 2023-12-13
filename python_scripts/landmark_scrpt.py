import cv2
import dlib
import pickle
import os
import numpy as np

# Load the dlib shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


glips_dataset_path = "Path-to-dataset"
landmarks_output_path = "Path-to-landmark-folder-inside-parent-directory-of-this-script"

for filename in os.listdir(glips_dataset_path):
    if os.path.exists(os.path.join(landmarks_output_path,filename)):
        print("+++++++"+filename+"++++++")
        continue
    for split in ["test","train","val"]:
        for video_filename in os.listdir(os.path.join(glips_dataset_path,filename,split)):
            if not video_filename.endswith('.mp4'):
                continue

            video_path = os.path.join(glips_dataset_path, filename,split,video_filename)

            output_dir = os.path.join(landmarks_output_path, filename, split)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            landmarks_array = []
           
            cap = cv2.VideoCapture(video_path)

            while True:
                # Capture the current frame
                ret, frame = cap.read()

                # Check if the end of the video has been reached
                if not ret:
                    cap.release()
                    break
                    

                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = detector(gray, 1)

                for face in faces:
                    # Get the landmarks for each face
                    landmarks = predictor(frame, face)

                    # Convert landmarks to a list of coordinates
                    landmark_coordinates = np.array([(landmark.x, landmark.y) for landmark in landmarks.parts()], dtype=np.float32)

                    # Reshape the landmark coordinates to the desired format
                    landmark_coordinates = landmark_coordinates.reshape((68, 2))

                    # Append the landmark coordinates to the array
                    landmarks_array.append(landmark_coordinates)

            # Save the landmarks for this video to the appropriate pickle file
            pickle_filename = f'{video_filename[:-4]}.pkl'
            pickle_filepath = os.path.join(output_dir, pickle_filename)
            with open(pickle_filepath, 'wb') as f:
                pickle.dump(landmarks_array, f)
    print("--------"+filename+"----------")