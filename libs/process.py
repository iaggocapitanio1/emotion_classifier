import cv2
import mediapipe as mp


def get_face_landmarks(image, face_mesh, draw=False):
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_input_rgb)

    image_landmarks = []

    if results.multi_face_landmarks:
        if draw:
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec,
            )

        landmarks = results.multi_face_landmarks[0].landmark
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        zs = [lm.z for lm in landmarks]

        for j in range(len(xs)):
            image_landmarks.extend([
                xs[j] - min(xs),
                ys[j] - min(ys),
                zs[j] - min(zs)
            ])

    return image_landmarks
