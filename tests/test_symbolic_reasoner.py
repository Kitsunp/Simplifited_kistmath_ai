import unittest
from src.models.symbolic_reasoner import SymbolicReasoner
import sympy as sp

class TestSymbolicReasoner(unittest.TestCase):
    def setUp(self):
        self.reasoner = SymbolicReasoner()

    def test_add_symbol(self):
        self.reasoner.add_symbol('x')
        self.assertIn('x', self.reasoner.symbols)
        self.assertIsInstance(self.reasoner.symbols['x'], sp.Symbol)

    def test_add_rule(self):
        rule = (sp.Symbol('x') + 1, sp.Symbol('y'))
        self.reasoner.add_rule(rule)
        self.assertIn(rule, self.reasoner.rules)

    def test_apply_rules(self):
        self.reasoner.add_symbol('x')
        self.reasoner.add_symbol('y')
        rule = (sp.Symbol('x') + 1, sp.Symbol('y'))
        self.reasoner.add_rule(rule)
        expression = sp.Symbol('x') + 1
        result = self.reasoner.apply_rules(expression)
        self.assertEqual(result, sp.Symbol('y'))
        self.reasoner.add_symbol('y')
        rule = (sp.Symbol('x') + 1, sp.Symbol('y'))
        self.reasoner.add_rule(rule)
        expression = sp.Symbol('x') + 1
        result = self.reasoner.apply_rules(expression)
        self.assertEqual(result, sp.Symbol('y'))

    def test_simplify(self):
        expression = sp.sin(sp.Symbol('x'))**2 + sp.cos(sp.Symbol('x'))**2
        simplified_expression = self.reasoner.simplify(expression)
        self.assertEqual(simplified_expression, 1)

    def test_symbolic_loss(self):
        predicted = sp.Symbol('x') + 1
        actual = sp.Symbol('x') + 2
        loss = self.reasoner.symbolic_loss(predicted, actual)
        self.assertGreater(loss, 0)

if __name__ == '__main__':
    unittest.main()