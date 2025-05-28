import unittest
import os
import sys
import librosa
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_name.data.preprocessing import Preprocessing


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Preprocessing(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, sampling_rate=48000)
        cls.dataset_dir = "Applied-ML-Group-7/project_name/data/data_audio_samples_split"
        cls.manual_dir = "Applied-ML-Group-7/project_name/data/manually_extracted_features"
        cls.spectogram_dir = "Applied-ML-Group-7/project_name/data/spectograms"

    def test_init(self):
        self.assertEqual(self.p.train_ratio, 0.7)
        self.assertEqual(self.p.valid_ratio, 0.15)
        self.assertEqual(self.p.test_ratio, 0.15)
        self.assertEqual(self.p.sampling_rate, 48000)
    
    def test_verify_split_runs(self):
        try:
            self.p.verify_split()
        except Exception as e:
            self.fail(f"verify_split() raised an exception: {e}")
    
    def test_find_max_sample_rate(self):
        result = self.p.find_max_sample_rate_per_class(self.dataset_dir)
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(v, int)
    
    def test_resample_audio_changes_sr(self):
        self.p.resample_audio(self.dataset_dir)
        for split in ["train", "valid", "test"]:
            split_path = os.path.join(self.dataset_dir, split)
            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                for file in os.listdir(cls_path):
                    if file.endswith(".wav"):
                        file_path = os.path.join(cls_path, file)
                        sr = librosa.get_samplerate(file_path)
                        self.assertEqual(sr, self.p.sampling_rate)
                        return

    def test_all_manual_features(self):
        for split in ["train", "valid", "test"]:
            split_path = os.path.join(self.manual_dir, split)
            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                for file in os.listdir(cls_path):
                    if file.endswith("_manual_seq.npy"):
                        full_path = os.path.join(cls_path, file)
                        self.assertTrue(os.path.exists(full_path))
                        data = np.load(full_path)
                        self.assertEqual(len(data.shape), 2, f"{file} should be 2D")
                        self.assertLessEqual(np.max(data), 1.0, f"{file} not normalized")
                        self.assertGreaterEqual(np.min(data), -1.0, f"{file} not normalized")

    def test_all_spectograms_tensor(self):
        for split in ["train", "valid", "test"]:
            split_path = os.path.join(self.spectogram_dir, split)
            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                for file in os.listdir(cls_path):
                    if file.endswith("_tensor.npy"):
                        full_path = os.path.join(cls_path, file)
                        self.assertTrue(os.path.exists(full_path))
                        tensor = np.load(full_path)
                        self.assertEqual(len(tensor.shape), 3, f"{file} should be 3D")
                        self.assertEqual(tensor.shape[0], 3, f"{file} first dim should be 3 (mel, mfcc, delta)")
                        self.assertLessEqual(np.max(tensor), 1.0, f"{file} not normalized")
                        self.assertGreaterEqual(np.min(tensor), -1.0, f"{file} not normalized")

    def test_all_matching_spec_and_manual_alignment(self):
        for split in ["train", "valid", "test"]:
            spec_split = os.path.join(self.spectogram_dir, split)
            manual_split = os.path.join(self.manual_dir, split)
            for cls in os.listdir(spec_split):
                spec_cls_dir = os.path.join(spec_split, cls)
                manual_cls_dir = os.path.join(manual_split, cls)
                for file in os.listdir(spec_cls_dir):
                    if file.endswith("_tensor.npy"):
                        base_name = file.replace("_tensor.npy", "")
                        spec_path = os.path.join(spec_cls_dir, file)
                        manual_path = os.path.join(manual_cls_dir, base_name + "_manual_seq.npy")
                        if not os.path.exists(manual_path):
                            continue

                        spec_tensor = np.load(spec_path)
                        manual_tensor = np.load(manual_path)

                        time_frames_spec = spec_tensor.shape[2]
                        time_frames_manual = manual_tensor.shape[0]

                        self.assertTrue(
                            abs(time_frames_spec - time_frames_manual) < 20,
                            f"{base_name}: Time mismatch (spec: {time_frames_spec}, manual: {time_frames_manual})"
                        )
    
    def test_expected_classes_exist(self):
        expected_classes = {"Airplane", "Bics", "bus", "Cars", "Helicopter", "Motocycles", "Train", "Truck"}
        for split in ["train", "valid", "test"]:
            split_path = os.path.join(self.dataset_dir, split)
            classes_found = set(os.listdir(split_path))
            self.assertTrue(expected_classes.issubset(classes_found),
                            f"Missing class folders in {split}: {expected_classes - classes_found}")

    def test_all_npy_files_non_empty(self):
        base_dirs = [self.manual_dir, self.spectogram_dir]
        for base in base_dirs:
            for split in ["train", "valid", "test"]:
                split_path = os.path.join(base, split)
                for cls in os.listdir(split_path):
                    cls_path = os.path.join(split_path, cls)
                    for file in os.listdir(cls_path):
                        if file.endswith(".npy"):
                            arr = np.load(os.path.join(cls_path, file))
                            self.assertGreater(arr.size, 0, f"{file} is empty")

if __name__ == "__main__":
    unittest.main()

