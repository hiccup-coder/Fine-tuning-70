# fine_tuning.py

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import evaluate

# -----------------------------
# 1. Load dataset (CSV)
# -----------------------------
# Your CSV should have columns: 'statement', 'excerpt', 'label'
dataset = load_dataset("csv", data_files={"train": "dataset.csv", "test": "dataset.csv"})
dataset = dataset.rename_column("relate", "label")

# -----------------------------
# 2. Load tokenizer
# -----------------------------
# Use slow tokenizer to avoid tiktoken/SentencePiece errors
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)

# Tokenize text pairs
def preprocess(examples):
    return tokenizer(
        examples["statement"],
        examples["excerpt"],
        # truncation=True,
        # padding="max_length",
        # max_length=256
        padding="longest"
    )

encoded_dataset = dataset.map(preprocess, batched=True)

# -----------------------------
# 3. Load model
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2,  # Option A / Option B
    id2label={0: "support", 1: "unrelated"},
    label2id={"support": 0, "unrelated": 1}
)

# -----------------------------
# 4. Define metrics
# -----------------------------
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy.compute(predictions=preds, references=labels)

# -----------------------------
# 5. Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"  # disable external logging like wandb
)

# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -----------------------------
# 7. Train
# -----------------------------

print("Start Train")
trainer.train()

# -----------------------------
# 8. Save model
# -----------------------------
print("Save model")
trainer.save_model("./deberta-finetuned-classifier")

# # -----------------------------
# # 9. Inference example
# # -----------------------------
# classifier = pipeline(
#     "text-classification",
#     model="./deberta-finetuned-classifier",
#     tokenizer=tokenizer,
#     device=0 if torch.cuda.is_available() else -1,  # GPU if available
#     return_all_scores=True  # get probabilities
# )

# examples = [
#     {
#         "text": "The first board game was created in ancient Egypt and was called Senet.",
#         "text_pair": "Backgammon also originated in Mesopotamia about 5,000 years ago."
#     },
#     {
#         "text": "Einstein developed the theory of relativity.",
#         "text_pair": "Newton formulated the laws of motion."
#     }
# ]

# results = classifier(examples, batch_size=8)

# # Print human-readable results
# id2label = {0: "Option A", 1: "Option B"}
# for i, res in enumerate(results):
#     # res contains list of dicts with 'label' and 'score'
#     sorted_res = sorted(res, key=lambda x: x["score"], reverse=True)
#     print(f"Example {i+1}:")
#     for r in sorted_res:
#         label_name = id2label[int(r["label"].split("_")[-1])]
#         print(f"  {label_name}: {r['score']:.4f}")
