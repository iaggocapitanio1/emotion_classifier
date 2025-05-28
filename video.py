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

    # Input and output paths
    input_video_path = settings.BASE_DIR / 'media' / 'input.mp4'
    output_video_path = settings.BASE_DIR / 'media' / 'output.mp4'

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"Failed to open video file: {input_video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0 or fps is None:
        fps = 25.0  # fallback FPS

    # Use a codec that's more compatible with Windows Media Player
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Initialize MediaPipe FaceMesh
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        print("🎥 Processing video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_landmarks = get_face_landmarks(frame, face_mesh, draw=True)

            if len(face_landmarks) == 1404:
                prediction = rf_classifier.predict([face_landmarks])[0]
                label = str(settings.EMOTIONS[int(prediction)])

                cv2.putText(
                    frame,
                    label,
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

            out.write(frame)

    cap.release()
    out.release()
    print("✅ Done. Output saved to:", output_video_path)

if __name__ == '__main__':
    main()
