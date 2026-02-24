import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def histogram_label(df):
    label = (df.groupby('label').count().sort_values(by='message', ascending=False).reset_index())
    group = (df.groupby('group').count().sort_values(by='message', ascending=False).reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    # dataset - Single Label
    axes[0, 0].bar(label['label'], label['message'])
    axes[0, 0].set_title("Training dataset - Single Label Count")
    axes[0, 0].set_xlabel("GitHub Single Label")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis='x', rotation=-45)
    # dataset - Group Label
    axes[0, 1].bar(group['group'], group['message'])
    axes[0, 1].set_title("Training dataset - Group Label Count")
    axes[0, 1].set_xlabel("GitHub Group Label")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].tick_params(axis='x', rotation=-45)
    plt.tight_layout()
    plt.show()


def plot_metrics(df, label):
    displayed_labels = df[label].unique()
    label_true = df[label]
    predicted = df['predicted_' + label]
    conf_matrix = confusion_matrix(label_true, predicted, labels=displayed_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=displayed_labels)
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', values_format='d')
    plt.title("Confusion Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_true, predicted, average='macro')
    print("Macro scores")
    print(f"Precision macro: {precision_macro:.4f}")
    print(f"Recall macro: {recall_macro:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    precision_cls, recall_cls, f1_cls, support_cls = precision_recall_fscore_support(label_true, predicted, labels=displayed_labels, average=None)
    print("\nClass Scores")
    for label, precision, recall, f1_score, support in zip(displayed_labels, precision_cls, recall_cls, f1_cls, support_cls):
        print(f"Class: {label}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1_score:.4f}")
        print(f"Support: {support}")
        print('-' * 30)


def plot_fine_tuning(json_path: str):
    os.makedirs("./plots", exist_ok=True)
    with open(json_path, "r") as f:
        trainer_state = json.load(f)

    log_history = trainer_state["log_history"]
    train_steps, train_loss = [], []
    eval_steps, eval_loss, eval_f1 = [], [], []

    for entry in log_history:
        step = entry.get("step")
        if "loss" in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])
            eval_f1.append(entry["eval_f1"])

    fig1 = plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_loss, linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Steps", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./plots/training_loss.png", dpi=300)
    plt.close(fig1)

    fig2, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(eval_steps, eval_loss, color='orange', linewidth=2, label="Eval Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Eval Loss", color='orange')
    ax1.tick_params(axis="y", color='orange')
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(eval_steps, eval_f1, color='green', linewidth=2, label="F1 Score")
    ax2.set_ylabel("F1 Score", color='green')
    ax2.tick_params(axis="y", color='green')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.title("Eval Loss & F1 Score over Steps", fontsize=14)
    plt.tight_layout()
    plt.savefig("./plots/eval_loss_f1.png", dpi=300)
    plt.close(fig2)