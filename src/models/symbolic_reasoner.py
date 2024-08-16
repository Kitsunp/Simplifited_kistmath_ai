import sympy as sp

class SymbolicReasoner:
    def __init__(self):
        self.symbols = {}
        self.rules = []

    def add_symbol(self, name):
        self.symbols[name] = sp.Symbol(name)

    def add_rule(self, rule):
        self.rules.append(rule)

    def apply_rules(self, expression):
        for rule in self.rules:
            expression = expression.replace(rule)
        return expression

    def simplify(self, expression):
        return sp.simplify(expression)

    def symbolic_loss(self, predicted, actual):
        return tf.reduce_mean(tf.square(predicted - actual))