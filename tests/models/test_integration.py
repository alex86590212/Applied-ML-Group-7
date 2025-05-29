import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from io import BytesIO
import numpy as np

from main_api import app

client = TestClient(app)

class TestAudioPredictionAPI(unittest.TestCase):

    @patch("main_api.predict_single")
    @patch("main_api.Preprocessing.extract_sequential_manual_features")
    @patch("main_api.pca")
    def test_rnn_prediction_success(self, mock_pca, mock_extract, mock_predict):
        mock_extract.return_value = np.random.rand(250, 13)
        mock_pca.return_value = np.random.rand(300, 15)
        mock_predict.return_value = 2 

        wav_file = BytesIO(b"fake wav content")
        response = client.post(
            "/predict-audio",
            files={"file": ("audio.wav", wav_file, "audio/wav")},
            data={"model_type": "RNN"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction_index"], 2)
        self.assertEqual(response.json()["prediction_label"], "Cars")
        self.assertEqual(response.json()["model"], "RNN")

        mock_extract.assert_called_once()
        mock_pca.assert_called_once()
        mock_predict.assert_called_once()

    def test_rejects_non_wav_file(self):
        fake_file = BytesIO(b"not-audio")
        response = client.post(
            "/predict-audio",
            files={"file": ("bad.mp3", fake_file, "audio/mpeg")},
            data={"model_type": "RNN"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only .wav files", response.json()["detail"])

    def test_invalid_model_type(self):
        wav_file = BytesIO(b"fake audio")
        response = client.post(
            "/predict-audio",
            files={"file": ("audio.wav", wav_file, "audio/wav")},
            data={"model_type": "INVALID"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("model_type must be one of", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
