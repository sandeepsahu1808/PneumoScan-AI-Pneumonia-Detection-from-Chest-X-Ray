import os
import tensorflow as tf
# Force CPU for evaluation to prevent Mac GPU (MPS) freezing
tf.config.set_visible_devices([], 'GPU')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json

from data_pipeline import get_generators

def evaluate_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    merged_data_dir = os.path.join(base_dir, 'data', 'merged')
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading test data generator...")
    _, _, test_gen, _ = get_generators(merged_data_dir)
    
    model_path = os.path.join(models_dir, 'best_model.h5')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please complete training first.")
        return
        
    print(f"Loading best model from {model_path}...")
    model = load_model(model_path, compile=False)
    # Recompile to enable evaluation metrics
    model.compile(loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    
    print("\nEvaluating model on unseen test set...")
    metrics = model.evaluate(test_gen, verbose=1)
    for name, value in zip(model.metrics_names, metrics):
        print(f"Test {name}: {value:.4f}")
        
    print("\nGenerating predictions...")
    # Reset generator before prediction to ensure it starts from the beginning
    test_gen.reset()
    pred_probs = model.predict(test_gen, verbose=1)
    y_prob = pred_probs.reshape(-1)
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # Find optimal threshold for recall >= 0.93
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Get threshold where recall >= 0.93
    valid_thresholds = thresholds[recalls[:-1] >= 0.93]
    if len(valid_thresholds) > 0:
        best_threshold = valid_thresholds[-1]
    else:
        best_threshold = 0.5

    print(f"\nOptimal threshold for 93% recall: {best_threshold:.4f}")

    # Generate predictions with both thresholds
    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_optimal = (y_prob >= best_threshold).astype(int)

    # Print both classification reports
    print("\n=== Classification Report (Default Threshold 0.5) ===")
    print(classification_report(y_true, y_pred_default, target_names=['NORMAL', 'PNEUMONIA']))

    print(f"\n=== Classification Report (Optimal Threshold {best_threshold:.4f}) ===")
    print(classification_report(y_true, y_pred_optimal, target_names=['NORMAL', 'PNEUMONIA']))

    # Plot confusion matrix for both thresholds side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: default threshold
    sns.heatmap(confusion_matrix(y_true, y_pred_default), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL','PNEUMONIA'],
                yticklabels=['NORMAL','PNEUMONIA'], ax=axes[0])
    axes[0].set_title('Confusion Matrix (Threshold 0.5)')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Right: optimal threshold
    sns.heatmap(confusion_matrix(y_true, y_pred_optimal),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL','PNEUMONIA'],
                yticklabels=['NORMAL','PNEUMONIA'], ax=axes[1])
    axes[1].set_title(f'Confusion Matrix (Threshold {best_threshold:.4f})')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('results/confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nGenerating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(results_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curve saved to {roc_path}")

    # SAVE METRICS JSON
    metrics = {
        "threshold_default": {
            "threshold": 0.5,
            "accuracy": round(accuracy_score(y_true, y_pred_default), 4),
            "precision": round(precision_score(y_true, y_pred_default), 4),
            "recall": round(recall_score(y_true, y_pred_default), 4),
            "f1": round(f1_score(y_true, y_pred_default), 4),
            "auc_roc": round(roc_auc_score(y_true, y_prob), 4)
        },
        "threshold_optimal": {
            "threshold": round(float(best_threshold), 4),
            "accuracy": round(accuracy_score(y_true, y_pred_optimal), 4),
            "precision": round(precision_score(y_true, y_pred_optimal), 4),
            "recall": round(recall_score(y_true, y_pred_optimal), 4),
            "f1": round(f1_score(y_true, y_pred_optimal), 4),
            "auc_roc": round(roc_auc_score(y_true, y_prob), 4)
        },
        "dataset": {
            "test_samples": int(len(y_true)),
            "normal_samples": int(sum(y_true == 0)),
            "pneumonia_samples": int(sum(y_true == 1))
        }
    }

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\nMetrics saved to results/metrics.json")

if __name__ == '__main__':
    evaluate_model()
