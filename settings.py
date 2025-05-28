from pathlib import  Path

from typing import List, Literal

Emotion = Literal['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / 'data'

RAW_DATA = BASE_DIR / 'raw'

EMOTIONS: List[Emotion] = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']