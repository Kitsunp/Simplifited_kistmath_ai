import os
import numpy as np
import tensorflow as tf
from src.models.kistmat_ai import Kistmat_AI
from src.utils.utils import generate_dataset, evaluate_model, plot_learning_curves, smooth_curriculum_learning

def main():
    print("Initializing Kistmat_AI model...")
    model = Kistmat_AI(input_shape=(50,), output_shape=2)

    stages = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 'junior_high2',
              'high_school1', 'high_school2', 'high_school3', 'university']

    print("Starting curriculum learning process...")
    all_history, evaluation_results = smooth_curriculum_learning(
        model,
        stages,
        initial_problems=1000,
        max_problems=2000,
        initial_difficulty=0.2,
        max_difficulty=5.0,
        difficulty_increase_rate=0.5,
        difficulty_decrease_rate=0.2,
        readiness_threshold=0.8,
        max_attempts_per_stage=5
    )

    print("\nGenerating final test problems...")
    test_problems = generate_dataset(100, 'university', difficulty=5.0)

    print("Evaluating model on final test set...")
    test_mse, test_r2 = evaluate_model(model, test_problems, 'university')

    print("\nFinal Test Results:")
    print(f"Mean Squared Error: {test_mse:.4f}")
    print(f"R-squared: {test_r2:.4f}")

    print("\nSample predictions:")
    sample_size = 5
    for i in range(sample_size):
        problem = test_problems[i]
        X = np.array([tokenize_calculus_problem(problem.problem)])
        prediction = model.predict(X)[0][0]
        print(f"Problem: {problem.problem}")
        print(f"Prediction: {prediction:.4f}")
        print(f"Actual solution: {problem.solution:.4f}")
        print()

    print("\nDetailed Evaluation Results:")
    for stage, results in evaluation_results.items():
        print(f"\n{'='*30}")
        print(f"{stage.upper()}:")
        print(f"{'='*30}")
        print(f"Readiness Score: {results['readiness_score']:.4f}")
        print(f"Transfer Ratio: {results['transfer_ratio']:.4f}")
        print(f"Consistency Score: {results['consistency_score']:.4f}")
        print(f"Memory Score: {results['memory_score']:.4f}")
        print(f"Generalization Score: {results['generalization_score']:.4f}")

    print("\nSaving final model...")
    model.save('final_kistmat_ai_model.keras')
    print("Final model saved as 'final_kistmat_ai_model.keras'")

    print("\nGenerating learning curve plots...")
    for stage_data in all_history:
        plot_learning_curves(stage_data, stage_data['stage'])

    print("\nKistmat_AI training and evaluation complete.")

if __name__ == "__main__":
    main()