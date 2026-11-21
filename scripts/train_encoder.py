import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback)

# ============ CONFIG ============
MODEL     = "microsoft/deberta-v3-base"
OUT       = r"D:\experiment\models\deberta-v3-base"
TEXT_COL  = "text"
LABEL_COL = "label"
SEED      = 42
BATCH     = 16   # OOM? -> 8 (and add gradient_accumulation_steps=2)
# ================================

ds = load_dataset("csv", data_files={
    "train":      r"D:\experiment\data\train.csv",
    "validation": r"D:\experiment\data\val.csv",
    "test":       r"D:\experiment\data\test.csv"})

labels = sorted(ds["train"].unique(LABEL_COL))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
print("Classes:", label2id)

tok = AutoTokenizer.from_pretrained(MODEL)

def prep(ex):
    return {**tok(ex[TEXT_COL], truncation=True, max_length=256),
            "labels": [label2id[l] for l in ex[LABEL_COL]]}

ds = ds.map(prep, batched=True, remove_columns=ds["train"].column_names)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(labels), id2label=id2label, label2id=label2id)

def metrics(p):
    pred = np.argmax(p.predictions, axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, pred),
            "f1_macro": f1_score(p.label_ids, pred, average="macro")}

args = TrainingArguments(
    output_dir=OUT, num_train_epochs=5, learning_rate=2e-5,
    per_device_train_batch_size=BATCH, per_device_eval_batch_size=32,
    weight_decay=0.01, warmup_ratio=0.1,
    eval_strategy="epoch", save_strategy="epoch",
    save_total_limit=1, load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=True, logging_steps=50, report_to="none", seed=SEED)

trainer = Trainer(model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

trainer.train()
print("TEST:", trainer.evaluate(ds["test"]))
trainer.save_model(OUT + r"\best")
