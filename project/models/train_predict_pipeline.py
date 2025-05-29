import os

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from .config import Config

config = Config()

def predict(model_class, model_args, save_path, test_loader, name: str = None, inference=False):
    model = model_class(**model_args)
    model.load(save_path)
    model.eval()

    if inference == True:
        y_pred = None

        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch[:-1], batch[-1] if isinstance(batch, (list, tuple)) else batch
                preds = model(*x_batch).argmax(dim=1) if isinstance(batch, (list, tuple)) else model(x_batch).argmax(dim=1)
                y_pred = preds.item()

        return y_pred

    elif inference == False:
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch[:-1], batch[-1] if isinstance(batch, (list, tuple)) else batch
                preds = model(*x_batch).argmax(dim=1) if isinstance(batch, (list, tuple)) else model(x_batch).argmax(dim=1)

                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        print("\nTest Set Classification Report")
        print(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', xticks_rotation=45)

        os.makedirs(config.confusion_matrix, exist_ok=True)
        output_path = os.path.join(config.confusion_matrix, f"confusion_matrix_{name.replace(' ', '_').lower()}.png")

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
    os.makedirs(config.loss_plots, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f"Training and Validation Loss of the {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.loss_plots, f"loss_curve_{name.lower()}.png"))
    plt.close()

    if predict_after_training:
        predict(model.__class__, model.get_model_args(), save_path, test_loader, name)


