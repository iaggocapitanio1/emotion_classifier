import gc
import math
import random
import shutil
from pathlib import Path
from typing import List

import cv2
import mediapipe as mp

import settings
from libs.process import get_face_landmarks
from libs.system import get_files


def move_percentage_to_test(data_dir: Path, keys: List[str], percentage: float = 0.2) -> None:
    """
    Moves a percentage of images from train/key to test/key based on filename keys.

    :param data_dir: Path to the base data folder containing 'train' and 'test'
    :param keys: List of keys like ['cloudy', 'rain', 'shine']
    :param percentage: Float between 0 and 1 indicating how many to move per class
    """
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'

    for key in keys:
        train_key_dir = train_dir / key
        test_key_dir = test_dir / key
        test_key_dir.mkdir(parents=True, exist_ok=True)

        if not train_key_dir.exists():
            print(f"Skipping {key}: folder not found in train/")
            continue

        images = [p for p in train_key_dir.iterdir() if p.is_file()]
        if not images:
            print(f"No images found in {train_key_dir}")
            continue

        n_to_move = math.ceil(len(images) * percentage)
        selected_images = random.sample(images, n_to_move)

        for image in selected_images:
            shutil.move(image, test_key_dir / image.name)
            print(f"Moved {image.name} -> {test_key_dir}")

def prepare_data() -> None:

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:


        file_path = settings.DATA_DIR /  'data.txt'

        with open(file_path.as_posix(), 'a') as f:
            for index, emotion in enumerate(settings.EMOTIONS):
                for img_index, image_path in enumerate(get_files(settings.DATA_DIR  / emotion)):
                    image = cv2.imread(image_path.as_posix())
                    if image is None:
                        continue

                    face_landmarks = get_face_landmarks(image, face_mesh)
                    if img_index % 200 == 0:
                        print(f'image_index: {img_index}, emotion: {emotion}')

                    if len(face_landmarks) == 1404:
                        face_landmarks.append(index)
                        line = ' '.join(map(str, face_landmarks))
                        f.write(line + '\n')

                    del image
                    del face_landmarks
                    gc.collect()

def get_max_value(dir: Path) -> int:
    max_index = 0
    for img_path in get_files(dir):
        name = img_path.stem.replace('im', '')
        value = int(name)
        if value > max_index:
            max_index = value
    return max_index

def move_emotions():
    for emotion in settings.EMOTIONS:
        raw_folder = settings.RAW_DATA / emotion
        data_folder = settings.DATA_DIR / emotion
        if raw_folder.exists():
            max_index = get_max_value(data_folder)
            for index, img_path in enumerate(get_files(raw_folder)):
                new_index: int = max_index + index + 1
                new_image_path = data_folder / f'im{new_index}.png'
                shutil.move(img_path.as_posix(), new_image_path.as_posix())

            if raw_folder.exists() and get_files(raw_folder).__len__() == 0:
                shutil.rmtree(raw_folder.as_posix())