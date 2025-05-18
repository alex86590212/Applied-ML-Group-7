import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from project_name.data.preprocessing import Preprocessing

import numpy as np
from baseline_rnn import RnnClassifier
from baseline_cnn import CNN
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight


if __name__ == "__main__":
    p = Preprocessing(0.7, 0.15, 0.15, 48000)
    Model = {1: "RNN", 2: "CNN"}
    K = 5
    NUMBER = 1

    train_spec = "Applied-ML-Group-7/project_name/data/spectograms/train"
    train_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/train"
    X_spec_train, X_manual_train, y_spec_train, y_manual_train = p.load_dual_inputs(train_spec, train_manual)

    valid_spec = "Applied-ML-Group-7/project_name/data/spectograms/valid"
    valid_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/valid"
    X_spec_valid, X_manual_valid, y_spec_valid, y_manual_valid = p.load_dual_inputs(valid_spec, valid_manual)

    test_spec = "Applied-ML-Group-7/project_name/data/spectograms/test"
    test_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/test"
    X_spec_test, X_manual_test, y_spec_test, y_manual_test = p.load_dual_inputs(test_spec, test_manual) 

    p.print_dataset_summary(X_spec_train, y_spec_train, X_spec_valid, y_spec_valid, X_spec_test, y_spec_test)
    p.print_dataset_summary(X_manual_train, y_manual_train, X_manual_valid, y_manual_valid, X_manual_test, y_manual_test)

    if Model[NUMBER] == "RNN":
        data_X = np.concatenate((X_manual_train, X_manual_valid), axis=0)
        data_y = np.concatenate((y_manual_train, y_manual_valid), axis=0)
    else:
        data_X = np.concatenate((X_spec_train, X_spec_valid), axis=0)
        data_y = np.concatenate((y_spec_train, y_spec_valid), axis=0)

    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_X)):
        print(f"\nFold {fold + 1}/{K}")

        if Model[NUMBER] == "RNN":
            input_dim = data_X.shape[2]
            hidden_dim = 64
            output_dim = len(np.unique(data_y))

            model = RnnClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=2)

        elif Model[NUMBER] == "CNN":
            input_channels = data_X.shape[1]
            input_h = data_X.shape[2]
            input_w = data_X.shape[3]
            output_dim = len(np.unique(data_y))

            model = CNN(no_channels=input_channels, 
                        no_classes=output_dim, 
                        input_h=input_h, 
                        input_w=input_w)

        class_weights_np = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(data_y[train_idx]),
            y=data_y[train_idx]
        )
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

        loss_fn = CrossEntropyLoss(weight=class_weights)
        optimizer = Adam(model.parameters(), lr=1e-3)

        x_train_tensor = torch.tensor(data_X[train_idx], dtype=torch.float32)
        y_train_tensor = torch.tensor(data_y[train_idx], dtype=torch.long)
        x_valid_tensor = torch.tensor(data_X[val_idx], dtype=torch.float32)
        y_valid_tensor = torch.tensor(data_y[val_idx], dtype=torch.long)

        train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
        valid_loader = DataLoader(TensorDataset(x_valid_tensor, y_valid_tensor), batch_size=32)

        epochs = 10
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch in train_loader:
                loss = model.train_step(x_batch, y_batch, optimizer, loss_fn)
                total_loss += loss

            val_loss, val_acc = model.evaluate(valid_loader, loss_fn)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")