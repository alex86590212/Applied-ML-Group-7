import unittest
import torch
import os
import sys
import tempfile
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from project_name.models.baseline_rnn import RnnClassifier


class TestRnnClassifier(unittest.TestCase):
    def setUp(self):
        self.input_dim = 13
        self.hidden_dim = 32
        self.output_dim = 8
        self.num_layers = 2
        self.batch_size = 4
        self.seq_len = 100

        self.model = RnnClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers
        )

        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        self.labels = torch.randint(0, self.output_dim, (self.batch_size,))
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.01)

    def test_initialization(self):
        self.assertIsInstance(self.model, RnnClassifier)

    def test_forward_pass_output_shape(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_train_step_runs(self):
        loss = self.model.train_step(self.input_tensor, self.labels, self.optimizer, self.loss_fn)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_get_model_args_keys(self):
        args = self.model.get_model_args()
        expected_keys = {"input_dim", "hidden_dim", "output_dim", "num_layers"}
        self.assertTrue(expected_keys.issubset(args.keys()))

    def test_save_and_load_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "rnn_model.pt")
            self.model.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            new_model = RnnClassifier(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers)
            new_model.load(save_path)

            for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()