import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(model, test_ds, label_encoder):
    """Evaluate model on test set and return metrics."""
    all_preds, all_true = [], []
    for x_batch, y_batch in test_ds:
        logits = model(x_batch, training=False)
        all_preds.extend(tf.argmax(logits, axis=1).numpy())
        all_true.extend(y_batch.numpy())
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    class_names = list(label_encoder.classes_)
    eval_results = {
        'accuracy': float((all_preds == all_true).mean()),
        'f1_macro': float(f1_score(all_true, all_preds, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(all_true, all_preds, average='weighted', zero_division=0)),
        'precision': float(precision_score(all_true, all_preds, average='macro', zero_division=0)),
        'recall': float(recall_score(all_true, all_preds, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(all_true, all_preds),
        'class_report': classification_report(all_true, all_preds, target_names=class_names, output_dict=True, zero_division=0),
        'class_names': class_names,
        'predictions': all_preds,
        'labels': all_true,
    }
    print(f"Test Accuracy: {eval_results['accuracy']:.4f}  F1-macro: {eval_results['f1_macro']:.4f}")
    return eval_results


def plot_training_history(history, output_dir):
    """Plot training history curves."""
    current_history = history.history

    if not current_history or 'loss' not in current_history:
        print("Warning: No training history found to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(current_history['loss']) + 1)

    axes[0].plot(epochs_range, current_history['loss'], label='Train')
    axes[0].plot(epochs_range, current_history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, current_history['accuracy'], label='Train')
    axes[1].plot(epochs_range, current_history['val_accuracy'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    print(f"Training plot saved to {output_dir / 'training_history.png'}")


def plot_confusion_matrix(eval_results, output_dir):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        eval_results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=eval_results['class_names'],
        yticklabels=eval_results['class_names'],
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    print(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")


def save_evaluation_results(eval_results, output_dir):
    """Save evaluation metrics to JSON and numpy files."""
    eval_summary = {
        'accuracy': float(eval_results['accuracy']),
        'f1_macro': float(eval_results['f1_macro']),
        'f1_weighted': float(eval_results['f1_weighted']),
        'precision': float(eval_results['precision']),
        'recall': float(eval_results['recall']),
        'per_class': {cls: eval_results['class_report'][cls] for cls in eval_results['class_names']},
        'confusion_matrix': eval_results['confusion_matrix'].tolist(),
    }

    with open(output_dir / 'eval_results.json', 'w') as f:
        json.dump(eval_summary, f, indent=2)

    np.save(output_dir / 'confusion_matrix.npy', eval_results['confusion_matrix'])
    np.save(output_dir / 'predictions.npy', eval_results['predictions'])

    print(f"Evaluation results saved to {output_dir}")