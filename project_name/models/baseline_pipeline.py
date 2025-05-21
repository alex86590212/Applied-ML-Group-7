import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from project_name.data.preprocessing import Preprocessing
from sklearn.metrics import classification_report


import matplotlib.pyplot as plt
import numpy as np
from baseline_rnn import RnnClassifier
from baseline_cnn import CNN
from main_combined_classifier import CombinedClassifier
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight


def run(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, save_path, name):
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(30):
        model.train()
        total_loss = 0
        total_samples = 0

        for batch in train_loader:
            if isinstance(batch, list) or isinstance(batch, tuple):
                x_batch, y_batch = batch[:-1], batch[-1]
                loss = model.train_step(*x_batch, y_batch, optimizer, loss_fn)
                batch_size = y_batch.size(0)
            else:
                x_batch, y_batch = batch
                loss = model.train_step(x_batch, y_batch, optimizer, loss_fn)
                batch_size = y_batch.size(0)

            total_loss += loss * batch_size
            total_samples += batch_size

        avg_train_loss = total_loss / total_samples

        val_loss, val_acc = model.evaluate(valid_loader, loss_fn)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/30 | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

    model.save(save_path)
    model.load(save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f"Training and Validation Loss of the {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_curve.png")
    plt.close()

    y_true, y_pred = [], []
    for batch in test_loader:
        with torch.no_grad():
            if isinstance(batch, list) or isinstance(batch, tuple):
                preds = model(*batch[:-1]).argmax(dim=1)
                y_batch = batch[-1]
            else:
                x_batch, y_batch = batch
                preds = model(x_batch).argmax(dim=1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.numpy())
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    p = Preprocessing(0.7, 0.15, 0.15, 48000)
    Model = {1: "RNN", 2: "CNN", 3: "Combined Model"}
    NUMBER = 3

    print("Loading train data...")
    train_spec = "Applied-ML-Group-7/project_name/data/spectograms/train"
    train_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/train"
    X_spec_train, X_manual_train, y_spec_train, y_manual_train = p.load_dual_inputs(train_spec, train_manual)
    print("Train data loaded.")

    valid_spec = "Applied-ML-Group-7/project_name/data/spectograms/valid"
    valid_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/valid"
    X_spec_valid, X_manual_valid, y_spec_valid, y_manual_valid = p.load_dual_inputs(valid_spec, valid_manual)

    test_spec = "Applied-ML-Group-7/project_name/data/spectograms/test"
    test_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/test"
    X_spec_test, X_manual_test, y_spec_test, y_manual_test = p.load_dual_inputs(test_spec, test_manual) 

    p.print_dataset_summary(X_spec_train, y_spec_train, X_spec_valid, y_spec_valid, X_spec_test, y_spec_test)
    p.print_dataset_summary(X_manual_train, y_manual_train, X_manual_valid, y_manual_valid, X_manual_test, y_manual_test)


    if Model[NUMBER] == "RNN":
        model = RnnClassifier(
            input_dim=X_manual_train.shape[2],
            hidden_dim=128,
            output_dim=len(np.unique(y_manual_train)),
            num_layers=2
        )
        loss_fn = CrossEntropyLoss(weight=torch.tensor(compute_class_weight("balanced", classes=np.unique(y_manual_train), y=y_manual_train), dtype=torch.float32))
        optimizer = Adam(model.parameters(), lr=1e-3)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_manual_train, dtype=torch.float32), torch.tensor(y_manual_train, dtype=torch.long)), batch_size=64, shuffle=True)
        valid_loader = DataLoader(TensorDataset(torch.tensor(X_manual_valid, dtype=torch.float32), torch.tensor(y_manual_valid, dtype=torch.long)), batch_size=64)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_manual_test, dtype=torch.float32), torch.tensor(y_manual_test, dtype=torch.long)), batch_size=64)

        run(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, save_path="Applied-ML-Group-7/project_name/models/model_weights/RNN_best_model.pt", name="RNN")

    elif Model[NUMBER] == "CNN":
        model = CNN(
            no_channels=X_spec_train.shape[1],
            no_classes=len(np.unique(y_spec_train)),
            input_h=X_spec_train.shape[2],
            input_w=X_spec_train.shape[3]
        )
        loss_fn = CrossEntropyLoss(weight=torch.tensor(compute_class_weight("balanced", classes=np.unique(y_spec_train), y=y_spec_train), dtype=torch.float32))
        optimizer = Adam(model.parameters(), lr=1e-3)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_spec_train, dtype=torch.float32), torch.tensor(y_spec_train, dtype=torch.long)), batch_size=64, shuffle=True)
        valid_loader = DataLoader(TensorDataset(torch.tensor(X_spec_valid, dtype=torch.float32), torch.tensor(y_spec_valid, dtype=torch.long)), batch_size=64)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_spec_test, dtype=torch.float32), torch.tensor(y_spec_test, dtype=torch.long)), batch_size=64)

        run(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, save_path="Applied-ML-Group-7/project_name/models/model_weights/CNN_best_model.pt", name="CNN")
    else:
        model = CombinedClassifier(
            no_channels=X_spec_train.shape[1],
            no_classes=len(np.unique(y_spec_train)),
            input_cnn_h=X_spec_train.shape[2],
            input_cnn_w=X_spec_train.shape[3],
            input_rnn_dim=X_manual_train.shape[2],
            hidden_rnn_dim=128
        )
        loss_fn = CrossEntropyLoss(weight=torch.tensor(compute_class_weight("balanced", classes=np.unique(y_spec_train), y=y_spec_train), dtype=torch.float32))
        optimizer = Adam(model.parameters(), lr=1e-3)

        train_loader = DataLoader(list(zip(
            torch.tensor(X_spec_train, dtype=torch.float32),
            torch.tensor(X_manual_train, dtype=torch.float32),
            torch.tensor(y_spec_train, dtype=torch.long)
        )), batch_size=64, shuffle=True)

        valid_loader = DataLoader(list(zip(
            torch.tensor(X_spec_valid, dtype=torch.float32),
            torch.tensor(X_manual_valid, dtype=torch.float32),
            torch.tensor(y_spec_valid, dtype=torch.long)
        )), batch_size=64)

        test_loader = DataLoader(list(zip(
            torch.tensor(X_spec_test, dtype=torch.float32),
            torch.tensor(X_manual_test, dtype=torch.float32),
            torch.tensor(y_spec_test, dtype=torch.long)
        )), batch_size=64)

        run(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, save_path="Applied-ML-Group-7/project_name/models/model_weights/combined_best_model.pt", name="Combined Model")
