import unittest
import torch
import os
import tempfile
import sys
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from project_name.models.baseline_cnn import CNN


class TestCNN(unittest.TestCase):
    def setUp(self):
        self.model = CNN()
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 3, 128, 300)
        self.labels = torch.randint(0, 8, (self.batch_size,))
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters())

    def test_cnn_initialization(self):
        self.assertIsInstance(self.model, CNN)

    def test_cnn_forward_pass(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 8))

    def test_cnn_train_step(self):
        loss = self.model.train_step(self.input_tensor, self.labels, self.optimizer, self.loss_fn)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_get_model_args(self):
        args = self.model.get_model_args()
        expected_keys = {"no_channels", "no_classes", "input_h", "input_w"}
        self.assertTrue(expected_keys.issubset(args.keys()))

    def test_cnn_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cnn_model.pt")
            self.model.save(path)
            self.assertTrue(os.path.exists(path))

            new_model = CNN()
            new_model.load(path)

            for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

if __name__ == "__main__":
    unittest.main()
