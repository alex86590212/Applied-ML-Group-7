from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

from project_name.models.config import Config

import os
import sys
import uuid
import numpy as np
import librosa
import torch

from project_name.models.baseline_main_models.baseline_rnn import RnnClassifier
from project_name.models.baseline_main_models.baseline_cnn import CNN
from project_name.models.baseline_main_models.main_combined_classifier import CombinedClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from project_name.data.preprocessing import Preprocessing, pad_or_crop_seq, pad_or_crop_spec
from project_name.models.train_predict_pipeline import predict

from pydantic import BaseModel, validator, ValidationError

app = FastAPI()

def pca(new_rnn_tensor, pca_dir):
    T, D = new_rnn_tensor.shape
    n_components = 15
    rnn_tensor_pca = np.zeros((T, n_components))
    for t in range(T):
        comp_path = os.path.join(pca_dir, f"pca_components_t{t}.npy")
        mean_path = os.path.join(pca_dir, f"pca_mean_t{t}.npy")
        
        components = np.load(comp_path)
        mean = np.load(mean_path)
        
        rnn_tensor_pca[t] = (new_rnn_tensor[t] - mean) @ components.T
    
    return rnn_tensor_pca

class ModelTypeChosen(BaseModel):
    model_type: str 

    @validator("model_type")
    def validate_model_type(cls, value):
        allowed = {"RNN", "CNN", "Combined"}
        if value not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return value

p = Preprocessing(0.7, 0.15, 0.15, 48000)

config = Config()
os.makedirs(config.user_uplods, exist_ok=True)

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
        save_path = os.path.join(config.user_uplods, f"{uid}.wav")
        with open(save_path, "wb") as f:
            f.write(await file.read())

        if model_type == "RNN":
                rnn_tensor = p.extract_sequential_manual_features(
                    audio_path=save_path, sr=48000, n_mfcc=13, hop_length=512, frame_length=2048
                )
                new_rnn_tensor = pad_or_crop_seq(rnn_tensor, target_frames=300)
                rnn_tensor_pca = pca(new_rnn_tensor, config.pca_components)
                x_tensor = torch.tensor(rnn_tensor_pca, dtype=torch.float32).unsqueeze(0)
                y_dummy = torch.tensor([0])
                test_loader = DataLoader(TensorDataset(x_tensor, y_dummy))

                model_class = RnnClassifier
                model_args = {
                    "input_dim": rnn_tensor_pca.shape[1],
                    "hidden_dim": 128,
                    "output_dim": 8,
                    "num_layers": 2
                }
                save_path_model=config.RNN_best_model_weights
        
        elif model_type == "CNN":
                cnn_tensor = p.spectograms_extraction(audio_path=save_path, n_mfcc=128, n_mels=128)
                new_tensor = pad_or_crop_spec(cnn_tensor, target_frames=300)
                x_tensor = torch.tensor(new_tensor, dtype=torch.float32).unsqueeze(0)
                y_dummy = torch.tensor([0])
                test_loader = DataLoader(TensorDataset(x_tensor, y_dummy))

                model_class = CNN
                model_args = {
                    "no_channels": new_tensor.shape[0],
                    "no_classes": 8,
                    "input_h": new_tensor.shape[1],
                    "input_w": new_tensor.shape[2]
                }
                save_path_model=config.CNN_best_model_weights

        elif model_type == "Combined":
                cnn_tensor = p.spectograms_extraction(audio_path=save_path, n_mfcc=128, n_mels=128)
                new_cnn_tensor = pad_or_crop_spec(cnn_tensor, target_frames=300)
            
                rnn_tensor = p.extract_sequential_manual_features(
                    audio_path=save_path, sr=48000, n_mfcc=13, hop_length=512, frame_length=2048
                )
                new_rnn_tensor = pad_or_crop_seq(rnn_tensor, target_frames=300)
                pca_dir = "project_name/data/pca_components"
                rnn_tensor_pca = pca(new_rnn_tensor, pca_dir)

                cnn_tensor_t = torch.tensor(new_cnn_tensor, dtype=torch.float32).unsqueeze(0)
                rnn_tensor_t = torch.tensor(rnn_tensor_pca, dtype=torch.float32).unsqueeze(0)
                y_dummy = torch.tensor([0])
                test_loader = DataLoader(list(zip(cnn_tensor_t, rnn_tensor_t, y_dummy)))

                model_class = CombinedClassifier
                model_args = {
                    "no_channels": new_cnn_tensor.shape[0],
                    "no_classes": 8,
                    "input_cnn_h": new_cnn_tensor.shape[1],
                    "input_cnn_w": new_cnn_tensor.shape[2],
                    "input_rnn_dim": rnn_tensor_pca.shape[1],
                    "hidden_rnn_dim": 128
                }
                save_path_model=config.Combined_best_model_weights
        
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type")

        pred = predict(model_class, model_args, save_path_model, test_loader, inference=True)

        return JSONResponse(content={
            "prediction_index": pred,
            "prediction_label": LABEL_MAP.get(pred, "Unknown"),
            "model": model_type
        })

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)




