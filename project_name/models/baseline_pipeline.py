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
from sklearn.utils.class_weight import compute_class_weight

if __name__ == "__main__":
    p = Preprocessing(0.7, 0.15, 0.15, 48000)
    Model = {1: "RNN", 2: "CNN"}

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if Model[NUMBER] == "RNN":
        input_dim = X_manual_train.shape[2]
        hidden_dim = 64
        output_dim = len(np.unique(y_manual_train))

        model = RnnClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=2).to(device)

        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=np.unique(y_manual_train),
                                             y=y_manual_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

        optimizer = Adam(model.parameters(), lr=1e-3)

        x_train_tensor = torch.tensor(X_manual_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_manual_train, dtype=torch.long).to(device)
        x_valid_tensor = torch.tensor(X_manual_valid, dtype=torch.float32).to(device)
        y_valid_tensor = torch.tensor(y_manual_valid, dtype=torch.long).to(device)

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
    
    NUMBER = 2
    if Model[NUMBER] == "CNN":
        input_channels = X_spec_train.shape[1]
        input_h = X_spec_train.shape[2]
        input_w = X_spec_train.shape[3]
        output_dim = len(np.unique(y_spec_train))

        model = CNN(no_channels=input_channels, 
                    no_classes=output_dim, 
                    input_h=input_h, 
                    input_w=input_w).to(device)

        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=np.unique(y_spec_train),
                                             y=y_spec_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

        optimizer = Adam(model.parameters(), lr=1e-3)

        x_train_tensor = torch.tensor(X_spec_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_spec_train, dtype=torch.long).to(device)
        x_valid_tensor = torch.tensor(X_spec_valid, dtype=torch.float32).to(device)
        y_valid_tensor = torch.tensor(y_spec_valid, dtype=torch.long).to(device)

        train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
        valid_loader = DataLoader(TensorDataset(x_valid_tensor, y_valid_tensor), batch_size=32)

        epochs = 10
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch in train_loader:
                loss = model.train_step(x_batch, y_batch, optimizer, loss_fn)
                total_loss += loss

            val_loss, val_acc = model.evaluate(valid_loader, loss_fn)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_acc:.2%}")
