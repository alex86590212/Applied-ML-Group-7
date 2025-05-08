import os
import random
import shutil
from collections import defaultdict
import librosa
import soundfile as sf
import numpy as np
from sklearn.decomposition import PCA


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
        source_dir = "Applied-ML-Group-7/project_name/data/data_audio_samples"
        target_dir = "Applied-ML-Group-7/project_name/data/data_audio_samples_split"
        split_ratio = {"train": self.train_ratio, "valid": self.valid_ratio, "test": self.test_ratio}

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
        source_dir = "Applied-ML-Group-7/project_name/data/data_audio_samples"
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
        for split in ["train", "valid", "test"]:
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
                        try:
                            audio, sr = librosa.load(file_path, sr=None)
                            if sr != self.sampling_rate:
                                audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
                                sf.write(file_path, audio_resampled, self.sampling_rate)
                        except Exception as e:
                            print(f"Failed to resample {file}: {e}")

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

                        np.save(os.path.join(output_cls_dir, f"{base_name}_mel.npy"), mel_norm)
                        np.save(os.path.join(output_cls_dir, f"{base_name}_mfcc.npy"), mfccs_norm)
                        np.save(os.path.join(output_cls_dir, f"{base_name}_delta.npy"), delta_mfcc_norm)

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
                os.makedirs(output_cls_dir, exist_ok=True)

                for file in os.listdir(cls_path):
                    if file.lower().endswith("wav"):
                        file_path = os.path.join(cls_path, file)
                        base_name = os.path.splitext(file)[0]
                        features = extract_manual_features(file_path, sr=self.sampling_rate, n_mfcc=n_mfcc)
                        np.save(os.path.join(output_cls_dir, f"{base_name}_manual.npy"), features)

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
    
    def print_dataset_summary(self, X_train, y_train, X_valid, y_valid, X_test, y_test, label_map):
        """
        Prints summary information about the loaded datasets.
        """
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
