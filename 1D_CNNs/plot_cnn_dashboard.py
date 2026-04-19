import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

LABEL_ORDER = ["Genuine", "Fraud"]

def main():
    df = pd.read_csv('dataset_preparado.csv', engine='python', encoding='utf-8')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    X_seq = tokenizer.texts_to_sequences(df['X_text'].astype(str))
    X_pad = pad_sequences(X_seq, maxlen=400, padding='post', truncating='post')
    Y_label = df['Y_label'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_pad, Y_label, test_size=0.20, 
        random_state=42, stratify=Y_label
    )
    
    model = load_model('modelo_fraude_final.keras')
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=LABEL_ORDER, output_dict=True)
    
    correct_per_class = np.diag(cm)
    support_per_class = cm.sum(axis=1)
    incorrect_per_class = support_per_class - correct_per_class
    total_correct = int(correct_per_class.sum())
    total_examples = int(support_per_class.sum())
    total_incorrect = total_examples - total_correct

    precision_values = [report[label]["precision"] for label in LABEL_ORDER]
    recall_values = [report[label]["recall"] for label in LABEL_ORDER]
    f1_values = [report[label]["f1-score"] for label in LABEL_ORDER]
    
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    label_positions = np.arange(len(LABEL_ORDER))

    axes[0, 0].bar(
        ["Correct", "Incorrect"],
        [total_correct, total_incorrect],
        color=["#2ca02c", "#d62728"],
        width=0.6,
    )
    axes[0, 0].set_title("Overall Correct vs Incorrect")
    axes[0, 0].set_ylabel("Examples")
    axes[0, 0].set_ylim(0, max(total_examples, total_correct) * 1.1)
    axes[0, 0].text(0, total_correct + total_examples * 0.02, f"{total_correct} ({total_correct / total_examples:.1%})", ha="center")
    axes[0, 0].text(1, total_incorrect + total_examples * 0.02, f"{total_incorrect} ({total_incorrect / total_examples:.1%})", ha="center")

    axes[0, 1].bar(label_positions, correct_per_class, color="#2ca02c", label="Correct")
    axes[0, 1].bar(label_positions, incorrect_per_class, bottom=correct_per_class, color="#d62728", label="Incorrect")
    axes[0, 1].set_xticks(label_positions, LABEL_ORDER)
    axes[0, 1].set_title("Per-Class Outcomes")
    axes[0, 1].set_ylabel("Examples")
    axes[0, 1].legend()
    for index, (correct_value, incorrect_value) in enumerate(zip(correct_per_class, incorrect_per_class)):
        axes[0, 1].text(index, correct_value / 2, str(int(correct_value)), ha="center", va="center", color="white", fontsize=9)
        if incorrect_value > 0:
            axes[0, 1].text(index, correct_value + incorrect_value / 2, str(int(incorrect_value)), ha="center", va="center", color="black", fontsize=9)

    width = 0.24
    axes[1, 0].bar(label_positions - width, precision_values, width=width, color="#1f77b4", label="Precision")
    axes[1, 0].bar(label_positions, recall_values, width=width, color="#ff7f0e", label="Recall")
    axes[1, 0].bar(label_positions + width, f1_values, width=width, color="#2ca02c", label="F1")
    axes[1, 0].set_xticks(label_positions, LABEL_ORDER)
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("Per-Class Metrics")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()

    image = axes[1, 1].imshow(cm, cmap="Blues")
    axes[1, 1].set_xticks(label_positions, LABEL_ORDER)
    axes[1, 1].set_yticks(label_positions, LABEL_ORDER)
    axes[1, 1].set_title("Confusion Matrix")
    axes[1, 1].set_xlabel("Predicted label")
    axes[1, 1].set_ylabel("True label")
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)
    threshold = cm.max() / 2 if cm.size else 0
    for row_index in range(cm.shape[0]):
        for column_index in range(cm.shape[1]):
            value = cm[row_index, column_index]
            color = "white" if value > threshold else "black"
            axes[1, 1].text(column_index, row_index, str(value), ha="center", va="center", color=color, fontsize=10)

    acc = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    figure.suptitle(f"1D CNN Validation Dashboard | Accuracy={acc:.4f} | Macro F1={macro_f1:.4f}", fontsize=14)
    figure.tight_layout()
    output_path = "cnn_validation_dashboard.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved dashboard to: {output_path}")

if __name__ == "__main__":
    main()
