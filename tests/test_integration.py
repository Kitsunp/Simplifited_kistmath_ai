import unittest
from src.models.kistmat_ai import Kistmat_AI
from src.utils.utils import generate_dataset, evaluate_model

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.model = Kistmat_AI(input_shape=(50,), output_shape=2)

    def test_training_flow(self):
        problems = generate_dataset(100, 'elementary1', 1.0)
        history = self.model.train(problems, epochs=1)
        self.assertTrue(len(history.history['loss']) > 0)

    def test_evaluation_flow(self):
        problems = generate_dataset(100, 'elementary1', 1.0)
        mse, r2 = evaluate_model(self.model, problems, 'elementary1')
        self.assertTrue(mse >= 0)
        self.assertTrue(-1 <= r2 <= 1)

if __name__ == '__main__':
    unittest.main()