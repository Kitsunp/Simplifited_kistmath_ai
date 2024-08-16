import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from src.models.math_problem import MathProblem

VOCAB_SIZE = 1000
MAX_LENGTH = 50
MAX_TERMS = 50

def tokenize_problem(problem, vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH):
    tokens = problem.lower().split()
    tokens = [hash(token) % vocab_size for token in tokens]
    tokens = tokens[:max_length]
    tokens += [0] * (max_length - len(tokens))
    return tokens

def tokenize_calculus_problem(problem, max_terms=MAX_TERMS):
    func_str = problem.split("d/dx ")[1].strip("()")
    terms = func_str.replace("-", "+-").split("+")

    coeffs = np.zeros(max_terms)
    exponents = np.zeros(max_terms)

    for i, term in enumerate(terms[:max_terms]):
        if 'x^' in term:
            coeff, exp = term.split('x^')
        elif 'x' in term:
            coeff, exp = term.split('x')
            exp = '1'
        else:
            coeff, exp = term, '0'

        coeffs[i] = float(coeff) if coeff else 1
        exponents[i] = float(exp)

    coeffs = coeffs / np.max(np.abs(coeffs)) if np.max(np.abs(coeffs)) > 0 else coeffs
    exponents = exponents / np.max(exponents) if np.max(exponents) > 0 else exponents

    return np.pad(np.concatenate([coeffs, exponents]), (0, MAX_LENGTH - 2*max_terms))

def generate_dataset(num_problems, stage, difficulty):
    problems = []
    for _ in range(num_problems):
        if stage == 'elementary1':
            a, b = np.random.randint(1, int(10 * difficulty) + 1, size=2)
            op = np.random.choice(['+', '-'])
            problem = f"{a} {op} {b}"
            solution = complex(eval(problem))
            problems.append(MathProblem(problem, solution, difficulty, op))
        elif stage == 'elementary2':
            a, b = np.random.randint(1, int(20 * difficulty) + 1, size=2)
            op = np.random.choice(['+', '-', '*'])
            problem = f"{a} {op} {b}"
            solution = complex(eval(problem))
            problems.append(MathProblem(problem, solution, difficulty, op))
        elif stage == 'elementary3':
            a, b = np.random.randint(1, int(30 * difficulty) + 1, size=2)
            op = np.random.choice(['+', '-', '*', '/'])
            problem = f"{a} {op} {b}"
            solution = complex(eval(problem))
            problems.append(MathProblem(problem, solution, difficulty, op))
        elif stage == 'junior_high1':
            a, b, c = np.random.randint(-int(10 * difficulty), int(10 * difficulty) + 1, size=3)
            if a == 0:
                a = 1
            problem = f"{a}x + {b} = {c}"
            solution = complex((c - b) / a)
            problems.append(MathProblem(problem, solution, difficulty, 'linear_equation'))
        elif stage == 'junior_high2':
            a, b, c = np.random.randint(-int(5 * difficulty), int(5 * difficulty) + 1, size=3)
            if a == 0:
                a = 1
            problem = f"{a}x^2 + {b}x + {c} = 0"
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                solution = (-b + np.sqrt(discriminant)) / (2*a)
            else:
                solution = complex(-b / (2*a), np.sqrt(-discriminant) / (2*a))
            problems.append(MathProblem(problem, solution, difficulty, 'quadratic'))
        elif stage == 'high_school1':
            base = np.random.randint(2, 5)
            exponent = np.random.randint(1, int(5 * difficulty) + 1)
            problem = f"log_{base}(x) = {exponent}"
            solution = base ** exponent
            problems.append(MathProblem(problem, solution, difficulty, 'logarithm'))
        elif stage == 'high_school2':
            angle = np.random.randint(0, 360)
            func = np.random.choice(['sin', 'cos', 'tan'])
            problem = f"{func}({angle}°)"
            if func == 'sin':
                solution = np.sin(np.radians(angle))
            elif func == 'cos':
                solution = np.cos(np.radians(angle))
            else:
                solution = np.tan(np.radians(angle))
            problems.append(MathProblem(problem, complex(solution), difficulty, 'trigonometry'))
        elif stage == 'high_school3':
            a = np.random.randint(1, int(3 * difficulty) + 1)
            problem = f"lim(x->0) (sin({a}x) / x)"
            solution = a
            problems.append(MathProblem(problem, solution, difficulty, 'limits'))
        elif stage == 'university':
            max_degree = max(1, int(3 * difficulty))  # Ensure max_degree is at least 1
            num_terms = np.random.randint(1, max_degree + 1)
            coeffs = np.random.randint(1, int(5 * difficulty) + 1, size=num_terms)
            exponents = np.random.randint(0, max_degree + 1, size=num_terms)

            problem_str = "d/dx ("
            solution = 0
            for coeff, exp in zip(coeffs, exponents):
                if exp == 0:
                    problem_str += f"{coeff} + "
                elif exp == 1:
                    problem_str += f"{coeff}x + "
                    solution += coeff
                else:
                    problem_str += f"{coeff}x^{exp} + "
                    solution += coeff * exp * (exp - 1)
            problem_str = problem_str.rstrip(" + ") + ")"

            problems.append(MathProblem(problem_str, solution, difficulty, 'derivatives'))
    return problems

def evaluate_readiness(model, problems, threshold):
    if model.get_learning_stage() == 'university':
        X = np.array([tokenize_calculus_problem(p.problem) for p in problems])
        y = np.array([p.solution for p in problems])
        if y.ndim == 1:
            y = np.column_stack((y, np.zeros_like(y)))
    else:
        X = np.array([tokenize_problem(p.problem) for p in problems])
        y_real = np.array([p.solution.real for p in problems])
        y_imag = np.array([p.solution.imag for p in problems])
        y = np.column_stack((y_real, y_imag))

    predictions = model.predict(X)
    mse = np.mean(np.square(y - predictions))
    r2 = r2_score(y, predictions)

    print(f"Evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")

    return r2

def evaluate_model(model, problems, stage):
    if stage == 'university':
        X = np.array([tokenize_calculus_problem(p.problem) for p in problems])
        y = np.array([p.solution for p in problems])
    else:
        X = np.array([tokenize_problem(p.problem) for p in problems])
        y_real = np.array([p.solution.real for p in problems])
        y_imag = np.array([p.solution.imag for p in problems])
        y = np.column_stack((y_real, y_imag))

    predictions = model.predict(X)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    if y.shape[1] == 1 and predictions.shape[1] == 2:
        # If actual output is real but prediction is complex
        y = np.column_stack((y, np.zeros_like(y)))
    elif y.shape[1] == 2 and predictions.shape[1] == 1:
        # If actual output is complex but prediction is real
        predictions = np.column_stack((predictions, np.zeros_like(predictions)))

    mse = np.mean(np.square(y - predictions))

    if y.shape[1] == 2:
        r2_real = r2_score(y[:, 0], predictions[:, 0])
        r2_imag = r2_score(y[:, 1], predictions[:, 1])
        r2 = (r2_real + r2_imag) / 2
    else:
        r2 = r2_score(y, predictions)

    return mse, r2