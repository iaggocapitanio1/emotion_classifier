# 🎭 Emotion Classifier

This project is a facial expression classification system that detects emotions using facial landmarks and machine learning.

## 📁 Project Structure

- **`treat_data.py`**  
  Prepares the data for training by extracting facial landmarks from images.  
  > 📂 Input images must be stored in the `data/` folder, organized by emotion (e.g., `data/angry`, `data/sad`, etc.).

- **`train.py`**  
  Trains a classification model using the extracted landmark features.

- **`main.py`**  
  Runs real-time emotion detection from a webcam using the trained model.

## 🧠 Dataset

The dataset used to train and test the model can be downloaded from the following link:  
📥 [Emotion Dataset on Google Drive](https://drive.google.com/drive/folders/1wCHAiGV3Q0eFqc15ImgISbQQ82Y6IGPf?usp=drive_link)

Make sure to place the unzipped dataset in the `data/` directory.

## 👨‍💻 Author

This project was developed by **Iaggo Capitanio**.
