import os
import random
import shutil
from collections import defaultdict
import librosa
import soundfile as sf
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


import kagglehub

# Download latest version
path = kagglehub.dataset_download("janboubiabderrahim/vehicle-sounds-dataset")
dataset_path = path

print("Path to dataset files:", path)


class Preprocessing:
    def __init__(self, train_ratio: float, valid_ratio: float, test_ratio: float, sampling_rate: int | None):
        """
        Initializes the Preprocessing object with split ratios and target sample rate.

        Args:
            train_ratio (float): Ratio of training data.
            valid_ratio (float): Ratio of validation data.
            test_ratio (float): Ratio of test data.
            sampling_rate (int | None): Desired sample rate for all audio files.
        """
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.sampling_rate = sampling_rate

    def split_the_data(self):
        """
        Splits the audio dataset into training, validation, and test sets based on specified ratios.
        Copies files from the source directory into corresponding subdirectories.
        """
        source_dir = dataset_path
        target_dir = "Applied-ML-Group-7/project_name/data/data_audio_samples_split"
        split_ratio = {"train": self.train_ratio, "valid": self.valid_ratio, "test": self.test_ratio}

        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        for split in split_ratio:
            for cls in os.listdir(source_dir):
                cls_path = os.path.join(target_dir, split, cls)
                os.makedirs(cls_path, exist_ok=True)

        for cls in os.listdir(source_dir):
            cls_dir = os.path.join(source_dir, cls)
            if not os.path.isdir(cls_dir):
                continue

            files = [file for file in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, file))]
            random.shuffle(files)

            total = len(files)
            train_end = int(split_ratio["train"] * total)
            valid_end = train_end + int(split_ratio["valid"] * total)

            splits = {
                "train": files[:train_end],
                "valid": files[train_end:valid_end],
                "test": files[valid_end:]
            }

            for split, split_files in splits.items():
                for file in split_files:
                    source_path = os.path.join(cls_dir, file)
                    destination_path = os.path.join(target_dir, split, cls, file)
                    shutil.copy2(source_path, destination_path)

    def verify_split(self):
        """
        Verifies the data split by printing the number of files in each split for each class.
        Helps ensure the dataset has been divided according to the intended ratios.
        """
        source_dir = dataset_path
        target_dir = "Applied-ML-Group-7/project_name/data/data_audio_samples_split"
        expected_splits = {"train": self.train_ratio, "valid": self.valid_ratio, "test": self.test_ratio}

        class_names = [cls for cls in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, cls))]

        for cls in class_names:
            original_path = os.path.join(source_dir, cls)
            original_count = len([
                file for file in os.listdir(original_path)
                if os.path.isfile(os.path.join(original_path, file))
            ])

            print(f"Class: {cls} with a total {original_count} files")

            for split in expected_splits:
                split_path = os.path.join(target_dir, split, cls)
                if not os.path.exists(split_path):
                    print(f"Missing {split}/{cls}")
                    continue

                split_count = len([
                    file for file in os.listdir(split_path)
                    if os.path.isfile(os.path.join(split_path, file))
                ])

                percent = (split_count / original_count) * 100 if original_count else 0
                print(f"{split:<5}: {split_count} files ({percent:.1f}%)")
            print("\n")

    def find_max_sample_rate_per_class(self, root_dir):
        """
        Finds and prints the maximum sample rate found in the dataset for each class.

        Args:
            root_dir (str): Path to the root directory of the split dataset.

        Returns:
            dict: Maximum sample rate per class.
        """
        max_sample_rate_per_class = defaultdict(int)

        for split in os.listdir(root_dir):
            split_path = os.path.join(root_dir, split)
            if not os.path.isdir(split_path):
                continue
            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                if not os.path.isdir(cls_path):
                    continue
                for file in os.listdir(cls_path):
                    if file.lower().endswith("wav"):
                        file_path = os.path.join(cls_path, file)
                        _, sample_rate = librosa.load(file_path, sr=None)
                        if sample_rate > max_sample_rate_per_class[cls]:
                            max_sample_rate_per_class[cls] = sample_rate

        for cls, sr in max_sample_rate_per_class.items():
            print(f"{cls}: {sr} Hz")
        return max_sample_rate_per_class

    def resample_audio(self, root_dir):
        """
        Resamples all .wav audio files in the dataset to the target sampling rate.

        Args:
            root_dir (str): Path to the dataset directory (train/valid/test structure).
        """

        summary = {}

        for split in ["train", "valid", "test"]:
            split_path = os.path.join(root_dir, split)
            if not os.path.isdir(split_path):
                continue

            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                if not os.path.isdir(cls_path):
                    continue

                resampled = 0 
                skipped = 0

                for file in os.listdir(cls_path):
                    if file.lower().endswith("wav"):
                        file_path = os.path.join(cls_path, file)
                        audio, sr = librosa.load(file_path, sr=None)
                        if sr != self.sampling_rate:
                            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
                            sf.write(file_path, audio_resampled, self.sampling_rate)
                            resampled += 1
                        else:
                            skipped += 1
                
                if cls not in summary:
                    summary[cls] = {"resampled": 0, "skipped": 0}
                summary[cls]["resampled"] += resampled
                summary[cls]["skipped"] += skipped

        for cls, counts in summary.items():
            print(f"{cls}: Resampled = {counts['resampled']}, Skipped = {counts['skipped']}")

    def noise_reduction(self, input_dir):
        """
        Applies silence trimming to each audio file to remove leading and trailing silence.

        Args:
            input_dir (str): Path to the dataset directory (train/valid/test structure).
        """
        for split in ["train", "valid", "test"]:
            split_path = os.path.join(input_dir, split)
            if not os.path.isdir(split_path):
                continue

            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                if not os.path.isdir(cls_path):
                    continue

                for file in os.listdir(cls_path):
                    if file.lower().endswith(".wav"):
                        input_path = os.path.join(cls_path, file)
                        try:
                            y, sr = librosa.load(input_path, sr=self.sampling_rate)
                            yt, _ = librosa.effects.trim(y)
                            sf.write(input_path, yt, sr)
                        except Exception as e:
                            print(f"Failed to trim {file}: {e}")

    def spectograms_extraction(self, n_mfcc, n_mels, input_root, output_root):
        """
        Extracts mel spectrograms, MFCCs, and delta MFCCs from audio files and saves them as .npy files.

        Args:
            n_mfcc (int): Number of MFCC coefficients to extract.
            n_mels (int): Number of mel bands.
            input_root (str): Root directory of input dataset.
            output_root (str): Directory to save extracted feature arrays.
        """
        for split in ["train", "valid", "test"]:
            split_path = os.path.join(input_root, split)
            if not os.path.isdir(split_path):
                continue

            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                if not os.path.isdir(cls_path):
                    continue

                output_cls_dir = os.path.join(output_root, split, cls)
                if os.path.exists(output_cls_dir):
                    shutil.rmtree(output_cls_dir)
                os.makedirs(output_cls_dir, exist_ok=True)

                for file in os.listdir(cls_path):
                    if file.lower().endswith(".wav"):
                        file_path = os.path.join(cls_path, file)
                        base_name = os.path.splitext(file)[0]
                        y, _ = librosa.load(file_path, sr=self.sampling_rate)

                        mel = librosa.feature.melspectrogram(y=y, sr=self.sampling_rate, n_mels=n_mels)
                        mel_db = librosa.power_to_db(mel, ref=np.max)
                        mel_norm = librosa.util.normalize(mel_db)

                        mfccs = librosa.feature.mfcc(S=mel_db, sr=self.sampling_rate, n_mfcc=n_mfcc)
                        mfccs_norm = librosa.util.normalize(mfccs)

                        delta_mfcc = librosa.feature.delta(mfccs_norm)
                        delta_mfcc_norm = librosa.util.normalize(delta_mfcc)

                        tensor = np.stack([mel_norm, mfccs_norm, delta_mfcc_norm], axis=0)

                        np.save(os.path.join(output_cls_dir, f"{base_name}_tensor.npy"), tensor)

    def manually_extracted_features(self, n_mfcc, input_root, output_root):
        """
        Extracts a vector of handcrafted statistical features (e.g., ZCR, RMS, spectral properties, MFCCs, chroma).
        Saves features as normalized .npy files.

        Args:
            n_mfcc (int): Number of MFCC coefficients.
            input_root (str): Path to input dataset.
            output_root (str): Path to save feature files.
        """
        def extract_manual_features(audio_path, sr, n_mfcc):
            y, sr = librosa.load(audio_path, sr=sr)

            if y.ndim > 1:
                y = librosa.to_mono(y)

            zcr = librosa.feature.zero_crossing_rate(y)[0]
            rms = librosa.feature.rms(y=y)[0]
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)[0]
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            features = np.array([
                np.mean(zcr), np.std(zcr),
                np.mean(rms), np.std(rms),
                np.mean(centroid), np.std(centroid),
                np.mean(bandwidth), np.std(bandwidth),
                np.mean(rolloff), np.std(rolloff),
                *np.mean(mfcc, axis=1),
                *np.std(mfcc, axis=1),
                *np.mean(chroma, axis=1)
            ])

            features = librosa.util.normalize(features)
            return features

        for split in ["train", "valid", "test"]:
            split_path = os.path.join(input_root, split)
            if not os.path.isdir(split_path):
                continue

            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                if not os.path.isdir(cls_path):
                    continue

                output_cls_dir = os.path.join(output_root, split, cls)
                if os.path.exists(output_cls_dir):
                    shutil.rmtree(output_cls_dir)
                os.makedirs(output_cls_dir, exist_ok=True)

                for file in os.listdir(cls_path):
                    if file.lower().endswith("wav"):
                        file_path = os.path.join(cls_path, file)
                        base_name = os.path.splitext(file)[0]
                        features = extract_manual_features(file_path, sr=self.sampling_rate, n_mfcc=n_mfcc)
                        np.save(os.path.join(output_cls_dir, f"{base_name}_manual.npy"), features)

    def load_dual_inputs(self, spectrogram_dir, manual_dir, target_frames=300):
        """
        Loads spectrogram and manual features for each class, pads/crops spectrograms to a fixed number of frames,
        and assigns numeric labels based on class names.

        Args:
            spectrogram_dir (str): Path to spectrograms (organized by class folders).
            manual_dir (str): Path to manually extracted features (same structure).
            target_frames (int): Fixed number of time frames for spectrogram tensors.

        Returns:
            Tuple: (X_spec, X_manual, y_spec, y_manual), each as a NumPy array.
        """
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

        X_spec = []
        X_manual = []
        y_spec = []
        y_manual = []

        def pad_or_crop(tensor, target_frames):
            _, height, width = tensor.shape
            if width < target_frames:
                pad_width = target_frames - width
                return np.pad(tensor, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
            else:
                return tensor[:, :, :target_frames]

        for cls in os.listdir(spectrogram_dir):
            spec_cls_dir = os.path.join(spectrogram_dir, cls)
            manual_cls_dir = os.path.join(manual_dir, cls)

            if not os.path.isdir(spec_cls_dir) or not os.path.isdir(manual_cls_dir):
                continue

            label = label_map[cls]

            for file in os.listdir(spec_cls_dir):
                base_name = file.replace("_tensor.npy", "")
                spec_path = os.path.join(spec_cls_dir, file)
                manual_path = os.path.join(manual_cls_dir, f"{base_name}_manual.npy")

                spec_tensor = np.load(spec_path)
                spec_tensor = pad_or_crop(spec_tensor, target_frames=target_frames)

                X_spec.append(spec_tensor)        
                X_manual.append(np.load(manual_path))     
                y_spec.append(label)
                y_manual.append(label)

        return np.array(X_spec), np.array(X_manual), np.array(y_spec), np.array(y_manual)

    def apply_pca(self, X_train, X_valid, X_test, n_components=50):
        """
        Applies PCA for dimensionality reduction on the dataset.

        Args:
            X_train (np.ndarray): Training feature array.
            X_valid (np.ndarray): Validation feature array.
            X_test (np.ndarray): Test feature array.
            n_components (float or int): Number of components or variance ratio to retain.

        Returns:
            Tuple: Transformed (X_train_pca, X_valid_pca, X_test_pca)
        """
        X_train_flat = np.vstack(X_train)
        X_valid_flat = np.vstack(X_valid)
        X_test_flat = np.vstack(X_test)

        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_flat)
        X_valid_pca = pca.transform(X_valid_flat)
        X_test_pca = pca.transform(X_test_flat)

        print(f"PCA reduced feature size from {X_train_flat.shape[1]} to {X_train_pca.shape[1]}")
        return X_train_pca, X_valid_pca, X_test_pca
    
    def plot_spectrogram(self, tensor, title="Mel Spectrogram"):
        """
        Plots only the mel spectrogram from a 3-channel spectrogram tensor.

        Args:
            tensor (np.ndarray): A 3xHxW tensor where tensor[0] is the mel spectrogram.
            title (str): Title of the plot.
        """

        mel = tensor[0]  

        plt.figure(figsize=(8, 4))
        plt.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        plt.title(title)
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bins")
        plt.colorbar(label="dB")
        plt.tight_layout()
        plt.show()
    
    def print_dataset_summary(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
        """
        Prints summary information about the loaded datasets.
        """
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
        print("\n--- Dataset Summary ---")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_valid)}")
        print(f"Test samples: {len(X_test)}")

        print("\nSample shapes:")
        print(f"X_train[0] shape: {np.array(X_train[0]).shape}")
        print(f"X_valid[0] shape: {np.array(X_valid[0]).shape}")
        print(f"X_test[0] shape: {np.array(X_test[0]).shape}")
        print(f"y_train[0]: {y_train[0]}")

        print("\nClass label mapping:")
        for cls, idx in label_map.items():
            print(f"  {cls}: {idx}")    


if __name__ == "__main__":
    p = Preprocessing(0.7, 0.15, 0.15, 48000)

    #p.split_the_data()
    p.verify_split()
    root_dir = "Applied-ML-Group-7/project_name/data/data_audio_samples_split"
    output_root_spectograms = "Applied-ML-Group-7/project_name/data/spectograms"
    output_root_manually_extracted_features = "Applied-ML-Group-7/project_name/data/manually_extracted_features"
    #p.find_max_sample_rate_per_class(root_dir)
    #p.resample_audio(root_dir)
    #p.noise_reduction(root_dir)
    #p.spectograms_extraction(128, 128, root_dir, output_root_spectograms)
    #p.manually_extracted_features(128, root_dir, output_root_manually_extracted_features)
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
    p.plot_spectrogram(X_spec_train[0])