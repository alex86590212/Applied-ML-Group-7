import numpy as np
from project.models.baseline_main_models.baseline_rnn import RnnClassifier
from project.models.baseline_main_models.baseline_cnn import CNN
from project.models.baseline_main_models.main_combined_classifier import CombinedClassifier
from torch.utils.data import DataLoader, TensorDataset
from project.models.config import Config

import os
import gdown, zipfile
from pathlib import Path

from project.models.train_predict_pipeline import train, predict

import torch
from project.data.preprocessing import Preprocessing

from project.features.correlation_matrix import plot_train_classwise_pca_correlation

import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

def extract_train_valid_test(zip_path: Path, out_dir: Path):
    """
    From a zip whose structure is
        some_root_folder/
            train/...
            valid/...
            test/...
    extract *only* the train, valid, and test trees into out_dir/, dropping that some_root_folder layer.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = [n for n in zf.namelist() if n.strip()]
        prefix = Path(all_names[0]).parts[0] + "/"

        for member in zf.infolist():
            rel = member.filename[len(prefix):]
            if not (rel.startswith("train/") or
                    rel.startswith("valid/") or
                    rel.startswith("test/")):
                continue

            target = out_dir / rel
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

if __name__ == "__main__":
    config = Config()
    p = Preprocessing(0.7, 0.15, 0.15, 48000)
    label_map = {
        "Airplane": 0,
        "Bics": 1,
        "Cars": 2,
        "Helicopter": 3,
        "Motocycles": 4,
        "Train": 5,
        "Truck": 6,
        "bus": 7
    }
    Model = {1: "RNN", 2: "CNN", 3: "Combined Model"}
    run_local = False
    # If you want to run local on your computer:
    if run_local:
        NUMBER = int(input("To choose the model that you want to train/predict, input number 1: RNN, 2: CNN, 3: Combined Model"))
        MODE = str(input("To choose the mode to train/predict, input: predict, train"))
        TRAIN_FROM_SCRATCH = bool(input("To chose if you want to train from scratch input: True, else input: False"))
    # If you want ro run it on habrok or other computer
    else:
        NUMBER = 3
        MODE = "train"
        #if you already downloaded the dataset, set this to None
        TRAIN_FROM_SCRATCH = True

    ZIPS = {
    config.data_audio_samples_split:    config.drive_url_splits,
    config.spectograms:                 config.drive_url_spectograms,
    config.manually_extracted_features: config.drive_url_manual_feats,
    config.pca_components:              config.drive_url_pca,
    }

    if TRAIN_FROM_SCRATCH == True:

        p.split_the_data()
        p.verify_split()
        p.find_max_sample_rate_per_class(config.data_audio_samples_split)
        p.resample_audio(config.data_audio_samples_split)
        p.noise_reduction(config.data_audio_samples_split)
        p.spectograms(128, 128, config.data_audio_samples_split, config.spectograms)
        p.sequential_manual_features(config.data_audio_samples_split, config.manually_extracted_features)

    elif TRAIN_FROM_SCRATCH == False:
        DATA_ROOT = Path(__file__).parent.parent / "data"
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        print("DATA_ROOT is:", DATA_ROOT.resolve())

        for cfg_path, share_url in ZIPS.items():
            folder_name = Path(cfg_path).name
            out_dir = DATA_ROOT / folder_name
            out_dir.mkdir(parents=True, exist_ok=True)

            zip_path = out_dir / f"{folder_name}.zip"
            gdown.download(share_url, str(zip_path), quiet=False, fuzzy=True)

            extract_train_valid_test(zip_path, out_dir)

            zip_path.unlink()
            print(f"`{folder_name}` ready at {out_dir}\n")

    X_spec_train, X_manual_train, y_spec_train, y_manual_train = p.load_dual_inputs(config.train_spec, config.train_manual)

    X_spec_valid, X_manual_valid, y_spec_valid, y_manual_valid = p.load_dual_inputs(config.valid_spec, config.valid_manual)

    X_spec_test, X_manual_test, y_spec_test, y_manual_test = p.load_dual_inputs(config.test_spec, config.test_manual) 

    p.print_dataset_summary(X_spec_train, y_spec_train, X_spec_valid, y_spec_valid, X_spec_test, y_spec_test)
    p.print_dataset_summary(X_manual_train, y_manual_train, X_manual_valid, y_manual_valid, X_manual_test, y_manual_test)

    X_train_pca, X_valid_pca, X_test_pca = p.apply_pca(X_manual_train, X_manual_valid, X_manual_test)
    plot_train_classwise_pca_correlation(X_train_pca, y_manual_train, label_map)

    #correlatino matrix of the manually_exrtacted_features after PCA
    fig = plt.gcf()
    fig.savefig(
        os.path.join(config.correlation, "train_classwise_pca_correlation.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(X_train_pca[0].shape)
    print(X_valid_pca[0].shape)
    print(X_test_pca[0].shape)

    model_type = Model[NUMBER]
    print(f"Model type: {model_type}")

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

            train(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, save_path=config.RNN_best_model_weights, name="RNN")

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

            train(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, save_path=config.CNN_best_model_weights, name="CNN")
        
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

            train(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, save_path=config.Combined_best_model_weights, name="Combined Model")
        
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
            save_path = config.RNN_best_model_weights
            if not Path(save_path).is_file():
                raise FileNotFoundError(
                    f"No RNN weights found at {save_path!r}; please train first "
                    "or point CONFIG.RNN_best_model_weights to an existing checkpoint."
                )
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
            save_path = config.CNN_best_model_weights

            if not Path(save_path).is_file():
                raise FileNotFoundError(
                    f"No CNN weights found at {save_path!r}; please train first "
                    "or point CONFIG.CNN_best_model_weights to an existing checkpoint."
                )
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
            save_path = config.Combined_best_model_weights
            if not Path(save_path).is_file():
                raise FileNotFoundError(
                    f"No Combined Model weights found at {save_path!r}; please train first "
                    "or point CONFIG.Combined_best_model_weights to an existing checkpoint."
                )
            predict(model_class, model_args, save_path, test_loader, "Combined Model")