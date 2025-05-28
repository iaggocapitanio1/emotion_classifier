import pickle
import cv2
import mediapipe as mp
import settings
from libs.process import get_face_landmarks


MODEL_PATH = './model.pkl'

def main():
    # Load trained RandomForest model
    with open(MODEL_PATH, 'rb') as f:
        rf_classifier = pickle.load(f)

    # Initialize MediaPipe FaceMesh
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            print("Failed to open webcam.")
            return

        print("🎥 Press 'q' to quit.")

        while True:
            ret, frame = webcam.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break


            face_landmarks = get_face_landmarks(frame, face_mesh,  draw=True)

            # Proceed only if a face was detected with expected features
            if len(face_landmarks) == 1404:
                prediction = rf_classifier.predict([face_landmarks])[0]
                label = str(settings.EMOTIONS[int(prediction)])

                # Display emotion on frame
                cv2.putText(
                    frame,
                    label,
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

            cv2.imshow('Emotion Recognition', frame, )

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
