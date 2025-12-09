
"""
Evaluation utilities for HODL Final Project.
All teammates should use these functions for consistent evaluation.
"""

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd

def evaluate_model(y_true, y_pred, y_prob):
    """
    Standardized evaluation function.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probability of class 1
    
    Returns:
        dict with accuracy, f1, auc_roc
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob)
    }

def print_results(results, model_name="Model"):
    """Pretty print results."""
    print(f"{model_name}: {results['accuracy']:.2%} acc, {results['f1']:.3f} F1, {results['auc_roc']:.3f} AUC")

def save_predictions(y_true, y_pred, y_prob, model_name, output_path):
    """Save predictions in standardized format."""
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    })
    df.to_csv(f'{output_path}/{model_name}_predictions.csv', index=False)
    print(f"Saved predictions to {output_path}/{model_name}_predictions.csv")
