#!/usr/bin/env python3
"""
Standalone tracing script: given an input WAV file, preprocesses it for each of RNN, CNN, and Combined models,
traces each with TorchScript, and saves the traced artifacts alongside the original weights.
"""
import os
import sys
import argparse
import uuid

import torch
import numpy as np


from project.models.config import Config
from project.data.preprocessing import Preprocessing, pad_or_crop_seq, pad_or_crop_spec
from project.models.baseline_main_models.baseline_rnn import RnnClassifier
from project.models.baseline_main_models.baseline_cnn import CNN
from project.models.baseline_main_models.main_combined_classifier import CombinedClassifier


def load_and_pca(rnn_tensor: np.ndarray, pca_dir: str) -> np.ndarray:
    """
    Apply per-frame PCA using saved components and means in `pca_dir`.
    Assumes files named pca_components_t{t}.npy and pca_mean_t{t}.npy.
    """
    T, D = rnn_tensor.shape
    n_components = np.load(os.path.join(pca_dir, "pca_components_t0.npy")).shape[0]
    out = np.zeros((T, n_components), dtype=np.float32)
    for t in range(T):
        comps = np.load(os.path.join(pca_dir, f"pca_components_t{t}.npy"))  # [n_components, D]
        mean = np.load(os.path.join(pca_dir, f"pca_mean_t{t}.npy"))         # [D]
        out[t] = (rnn_tensor[t] - mean) @ comps.T
    return out


def main():
    parser = argparse.ArgumentParser(description="Trace RNN, CNN, and Combined models to TorchScript")
    parser.add_argument("wav", help="Path to input .wav file for example tracing")
    args = parser.parse_args()

    wav_path = args.wav
    if not wav_path.endswith(".wav"):
        sys.exit("Error: input must be a .wav file")

    # Initialize config and preprocessing
    config = Config()
    p = Preprocessing(0.7, 0.15, 0.15, 48000)

    # Common: temporary storage for trace inputs
    # 1) RNN pipeline
    rnn_tensor = p.extract_sequential_manual_features(
        audio_path=wav_path,
        sr=48000,
        n_mfcc=13,
        hop_length=512,
        frame_length=2048
    )
    rnn_tensor = pad_or_crop_seq(rnn_tensor, target_frames=300)
    rnn_pca = load_and_pca(rnn_tensor, config.pca_components)
    x_rnn = torch.tensor(rnn_pca, dtype=torch.float32).unsqueeze(0)  # [1, T, D']

    # 2) CNN pipeline
    cnn_tensor = p.spectograms_extraction(audio_path=wav_path, n_mfcc=128, n_mels=128)
    cnn_tensor = pad_or_crop_spec(cnn_tensor, target_frames=300)
    x_cnn = torch.tensor(cnn_tensor, dtype=torch.float32).unsqueeze(0)  # [1, C, H, W]

    # 3) Combined pipeline uses both
    x_combined = (x_cnn, x_rnn)

    # Define models for tracing
    tasks = [
        ("RNN", RnnClassifier, config.RNN_best_model_weights,
            {"input_dim": rnn_pca.shape[1], "hidden_dim": 128, "output_dim": 8, "num_layers": 2}, x_rnn),
        ("CNN", CNN, config.CNN_best_model_weights,
            {"no_channels": cnn_tensor.shape[0], "no_classes": 8, "input_h": cnn_tensor.shape[1], "input_w": cnn_tensor.shape[2]}, x_cnn),
        ("Combined", CombinedClassifier, config.Combined_best_model_weights,
            {"no_channels": cnn_tensor.shape[0], "no_classes": 8, "input_cnn_h": cnn_tensor.shape[1], "input_cnn_w": cnn_tensor.shape[2], "input_rnn_dim": rnn_pca.shape[1], "hidden_rnn_dim": 128}, x_combined),
    ]

    for name, ModelClass, weight_path, init_args, example_input in tasks:
        if not os.path.isfile(weight_path):
            print(f"Warning: weight file for {name} not found at {weight_path}, skipping.")
            continue

        print(f"Tracing {name} model...")
        # Load weights
        sd = torch.load(weight_path, weights_only=True)
        model = ModelClass(**init_args)
        model.load_state_dict(sd)
        model.eval()

        # Trace
        traced = torch.jit.trace(model, example_input)
        traced_path = weight_path.replace(".pt", f"_{name.lower()}_traced.pt")
        traced.save(traced_path)
        print(f"Saved traced {name} model to {traced_path}\n")

    print("Tracing complete.")


if __name__ == "__main__":
    main()
