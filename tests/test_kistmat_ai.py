import unittest
from src.models.kistmat_ai import Kistmat_AI
from src.utils.utils import tokenize_problem

class TestKistmatAI(unittest.TestCase):
    def setUp(self):
        self.model = Kistmat_AI(input_shape=(50,), output_shape=2)

    def test_call(self):
        inputs = [[1] * 50]
        result = self.model(inputs)
        self.assertEqual(result.shape, (1, 2))

    def test_learning_stage(self):
        self.model.set_learning_stage('junior_high1')
        self.assertEqual(self.model.get_learning_stage(), 'junior_high1')

    def test_tokenize_problem(self):
        problem = "2 + 2"
        tokens = tokenize_problem(problem)
        self.assertEqual(len(tokens), 50)
        self.assertTrue(all(isinstance(token, int) for token in tokens))

    def test_model_training(self):
        problems = [
            {"problem": "2 + 2", "solution": 4},
            {"problem": "3 + 5", "solution": 8}
        ]
        tokenized_problems = [tokenize_problem(p["problem"]) for p in problems]
        solutions = [p["solution"] for p in problems]
        history = self.model.fit(tokenized_problems, solutions, epochs=1)
        self.assertTrue(len(history.history['loss']) > 0)

if __name__ == '__main__':
    unittest.main()