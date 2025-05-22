from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

import os
import sys
import uuid
import numpy as np
import librosa
import torch

from project_name.models.baseline_rnn import RnnClassifier
from project_name.models.baseline_cnn import CNN
from project_name.models.main_combined_classifier import CombinedClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from project_name.data.preprocessing import Preprocessing

from pydantic import BaseModel, validator, ValidationError

app = FastAPI()

class ModelTypeChosen(BaseModel):
    model_type: str 

    @validator("model_type")
    def validate_model_type(cls, value):
        allowed = {"RNN", "CNN", "Combined"}
        if value not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return value

p = Preprocessing(0.7, 0.15, 0.15, 48000)

SAVE_DIR = "Applied-ML-Group-7/project_name/models/user_uploads"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/predict-audio")
async def predict_audio(model_type: str = Form(...), file: UploadFile = File(...)):
    LABEL_MAP = {
    0: "Airplane",
    1: "Bics",
    2: "Cars",
    3: "Helicopter",
    4: "Motocycles",
    5: "Train",
    6: "Truck",
    7: "Bus"
    }
    try:
        model_chosen = ModelTypeChosen(model_type=model_type)
    except (ValidationError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    try:
        uid = str(uuid.uuid4())
        save_path = os.path.join(SAVE_DIR, f"{uid}.wav")
        with open(save_path, "wb") as f:
            f.write(await file.read())

        if model_type == "RNN":
                features = p.extract_sequential_manual_features(
                    audio_path=save_path, sr=48000, n_mfcc=13, hop_length=512, frame_length=2048
                )
                features = features[:100] if features.shape[0] > 100 else np.pad(features, ((0, 100 - features.shape[0]), (0, 0)))
                x = torch.tensor([features], dtype=torch.float32)
                model = RnnClassifier(input_dim=features.shape[1], hidden_dim=128, output_dim=8, num_layers=2)
                model.load("project_name/models/model_weights/RNN_best_model.pt")
        
        elif model_type == "CNN":
                y, _ = librosa.load(save_path, sr=48000)
                mel = librosa.feature.melspectrogram(y=y, sr=48000, n_mels=128)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_norm = librosa.util.normalize(mel_db)
                x = torch.tensor([[mel_norm]], dtype=torch.float32)
                model = CNN(no_channels=1, no_classes=8, input_h=mel_norm.shape[0], input_w=mel_norm.shape[1])
                model.load("project_name/models/model_weights/CNN_best_model.pt")

        elif model_type == "Combined":
                y, _ = librosa.load(save_path, sr=48000)
                mel = librosa.feature.melspectrogram(y=y, sr=48000, n_mels=128)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_norm = librosa.util.normalize(mel_db)
                cnn_tensor = torch.tensor([[mel_norm]], dtype=torch.float32)

                features = p.extract_sequential_manual_features(
                    audio_path=save_path, sr=48000, n_mfcc=13, hop_length=512, frame_length=2048
                )
                features = features[:100] if features.shape[0] > 100 else np.pad(features, ((0, 100 - features.shape[0]), (0, 0)))
                rnn_tensor = torch.tensor([features], dtype=torch.float32)

                x = (cnn_tensor, rnn_tensor)
                model = CombinedClassifier(
                    no_channels=1,
                    no_classes=8,
                    input_cnn_h=mel_norm.shape[0],
                    input_cnn_w=mel_norm.shape[1],
                    input_rnn_dim=features.shape[1],
                    hidden_rnn_dim=128
                )
                model.load("project_name/models/model_weights/combined_best_model.pt")
        
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type")

        model.eval()
        with torch.no_grad():
            pred = model(*x).argmax(dim=1).item() if isinstance(x, tuple) else model(x).argmax(dim=1).item()

        return JSONResponse(content={
            "prediction_index": pred,
            "prediction_label": LABEL_MAP.get(pred, "Unknown"),
            "model": model_type
        })

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)




