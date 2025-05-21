import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from project_name.data.preprocessing import Preprocessing
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


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

def predict(model_class, model_args, save_path, test_loader, name: str):
    model = model_class(**model_args)
    model.load(save_path)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch[:-1], batch[-1] if isinstance(batch, (list, tuple)) else batch
            preds = model(*x_batch).argmax(dim=1) if isinstance(batch, (list, tuple)) else model(x_batch).argmax(dim=1)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n=== Test Set Classification Report ===")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', xticks_rotation=45)

    output_dir = "Applied-ML-Group-7/project_name/models/confusion_matrix"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"confusion_matrix_{name.replace(' ', '_').lower()}.png")

    plt.title(f"Confusion Matrix of {name}")
    plt.tight_layout()
    plt.savefig(output_path)  
    plt.show()


def train(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, save_path, name, predict_after_training=True):
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
        print(f"Epoch {epoch+1}/30 | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

    model.save(save_path)

    plot_dir = "Applied-ML-Group-7/project_name/models/loss_plots"
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f"Training and Validation Loss of the {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"loss_curve_{name.lower()}.png"))
    plt.close()

    if predict_after_training:
        predict(model.__class__, model.get_model_args(), save_path, test_loader, name)


if __name__ == "__main__":
    p = Preprocessing(0.7, 0.15, 0.15, 48000)
    Model = {1: "RNN", 2: "CNN", 3: "Combined Model"}
    NUMBER = 1
    MODE = "predict"

    model_type = Model[NUMBER]
    print(f"Model type: {model_type}")

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

    X_train_pca, X_valid_pca, X_test_pca = p.apply_pca(X_manual_train, X_manual_valid, X_manual_test)
    print(X_train_pca[0].shape)
    print(X_valid_pca[0].shape)
    print(X_test_pca[0].shape)

    if MODE == "train":

        if model_type == "RNN":
            model = RnnClassifier(
                input_dim=X_train_pca.shape[2],
                hidden_dim=128,
                output_dim=len(np.unique(y_manual_train)),
                num_layers=2
            )
            loss_fn = CrossEntropyLoss(weight=torch.tensor(compute_class_weight("balanced", classes=np.unique(y_manual_train), y=y_manual_train), dtype=torch.float32))
            optimizer = Adam(model.parameters(), lr=1e-3)

            train_loader = DataLoader(TensorDataset(torch.tensor(X_train_pca, dtype=torch.float32), torch.tensor(y_manual_train, dtype=torch.long)), batch_size=64, shuffle=True)
            valid_loader = DataLoader(TensorDataset(torch.tensor(X_valid_pca, dtype=torch.float32), torch.tensor(y_manual_valid, dtype=torch.long)), batch_size=64)
            test_loader = DataLoader(TensorDataset(torch.tensor(X_test_pca, dtype=torch.float32), torch.tensor(y_manual_test, dtype=torch.long)), batch_size=64)

            train(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, save_path="Applied-ML-Group-7/project_name/models/model_weights/RNN_best_model.pt", name="RNN")

        elif model_type == "CNN":
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

            train(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, save_path="Applied-ML-Group-7/project_name/models/model_weights/CNN_best_model.pt", name="CNN")
        
        elif model_type == "Combined Model":
            model = CombinedClassifier(
                no_channels=X_spec_train.shape[1],
                no_classes=len(np.unique(y_spec_train)),
                input_cnn_h=X_spec_train.shape[2],
                input_cnn_w=X_spec_train.shape[3],
                input_rnn_dim=X_train_pca.shape[2],
                hidden_rnn_dim=128
            )
            loss_fn = CrossEntropyLoss(weight=torch.tensor(compute_class_weight("balanced", classes=np.unique(y_spec_train), y=y_spec_train), dtype=torch.float32))
            optimizer = Adam(model.parameters(), lr=1e-3)

            train_loader = DataLoader(list(zip(
                torch.tensor(X_spec_train, dtype=torch.float32),
                torch.tensor(X_train_pca, dtype=torch.float32),
                torch.tensor(y_spec_train, dtype=torch.long)
            )), batch_size=64, shuffle=True)

            valid_loader = DataLoader(list(zip(
                torch.tensor(X_spec_valid, dtype=torch.float32),
                torch.tensor(X_valid_pca, dtype=torch.float32),
                torch.tensor(y_spec_valid, dtype=torch.long)
            )), batch_size=64)

            test_loader = DataLoader(list(zip(
                torch.tensor(X_spec_test, dtype=torch.float32),
                torch.tensor(X_test_pca, dtype=torch.float32),
                torch.tensor(y_spec_test, dtype=torch.long)
            )), batch_size=64)

            train(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, save_path="Applied-ML-Group-7/project_name/models/model_weights/combined_best_model.pt", name="Combined Model")
        
    elif MODE == "predict":
        if model_type == "RNN":
            model_args = dict(
                input_dim=X_train_pca.shape[2],
                hidden_dim=128,
                output_dim=len(np.unique(y_manual_train)),
                num_layers=2
            )
            model_class = RnnClassifier
            test_loader = DataLoader(TensorDataset(
                torch.tensor(X_test_pca, dtype=torch.float32),
                torch.tensor(y_manual_test, dtype=torch.long)
            ), batch_size=64)
            save_path = "Applied-ML-Group-7/project_name/models/model_weights/RNN_best_model.pt"
            predict(model_class, model_args, save_path, test_loader, "RNN")

        elif model_type == "CNN":
            model_args = dict(
                no_channels=X_spec_train.shape[1],
                no_classes=len(np.unique(y_spec_train)),
                input_h=X_spec_train.shape[2],
                input_w=X_spec_train.shape[3]
            )
            model_class = CNN
            test_loader = DataLoader(TensorDataset(
                torch.tensor(X_spec_test, dtype=torch.float32),
                torch.tensor(y_spec_test, dtype=torch.long)
            ), batch_size=64)
            save_path = "Applied-ML-Group-7/project_name/models/model_weights/CNN_best_model.pt"
            predict(model_class, model_args, save_path, test_loader, "CNN")

        elif model_type == "Combined Model":
            model_args = dict(
                no_channels=X_spec_train.shape[1],
                no_classes=len(np.unique(y_spec_train)),
                input_cnn_h=X_spec_train.shape[2],
                input_cnn_w=X_spec_train.shape[3],
                input_rnn_dim=X_train_pca.shape[2],
                hidden_rnn_dim=128
            )
            model_class = CombinedClassifier
            test_loader = DataLoader(list(zip(
                torch.tensor(X_spec_test, dtype=torch.float32),
                torch.tensor(X_test_pca, dtype=torch.float32),
                torch.tensor(y_spec_test, dtype=torch.long)
            )), batch_size=64)
            save_path = "Applied-ML-Group-7/project_name/models/model_weights/combined_best_model.pt"
            predict(model_class, model_args, save_path, test_loader, "Combined Model")

