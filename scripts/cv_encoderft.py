import gc, os
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

# ---------------- CONFIG ----------------
MODEL = "microsoft/deberta-v3-base"
NAME  = "DeBERTaV3 in-domain"
BATCH, EPOCHS, LR = 16, 4, 2e-5
# ----------------------------------------
FOLD_DIR = r"D:\experiment\data-ai"
RES_CSV  = r"D:\experiment\results_cv.csv"
LABELS = ["negative", "positive"]
EVAL_B = 32

done = set()
if os.path.exists(RES_CSV):
    done = {(r["name"], int(r["fold"])) for _, r in pd.read_csv(RES_CSV).iterrows()}

for k in range(5):
    if (NAME, k) in done:
        print("skip", NAME, k); continue
    tr = pd.read_csv(rf"{FOLD_DIR}\fold{k}_train.csv")
    te = pd.read_csv(rf"{FOLD_DIR}\fold{k}_test.csv")
    tok = AutoTokenizer.from_pretrained(MODEL)
    def prep(ex):
        return {**tok(ex["text"], truncation=True, max_length=256),
                "labels": [LABELS.index(l) for l in ex["label"]]}
    tr_ds = HFDataset.from_pandas(tr[["text","label"]], preserve_index=False) \
                     .map(prep, batched=True, remove_columns=["text","label"])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    args = TrainingArguments(output_dir=r"D:\experiment\models\_cv_tmp",
        num_train_epochs=EPOCHS, learning_rate=LR,
        per_device_train_batch_size=BATCH, weight_decay=0.01, warmup_ratio=0.1,
        fp16=True, eval_strategy="no", save_strategy="no", report_to="none", seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=tr_ds, tokenizer=tok,
                      data_collator=DataCollatorWithPadding(tok))
    trainer.train(); model.eval()
    preds = []
    with torch.no_grad():
        texts = te["text"].astype(str).tolist()
        for i in range(0, len(texts), EVAL_B):
            enc = tok(texts[i:i+EVAL_B], truncation=True, max_length=256,
                      padding=True, return_tensors="pt").to("cuda")
            preds += [LABELS[j] for j in model(**enc).logits.argmax(-1).tolist()]
    acc = accuracy_score(te["label"], preds)
    f1 = f1_score(te["label"], preds, labels=LABELS, average="macro")
    pd.DataFrame([{"name": NAME, "fold": k, "acc": round(acc,4), "f1": round(f1,4)}]
        ).to_csv(RES_CSV, mode="a", header=not os.path.exists(RES_CSV), index=False)
    print(f"  -> fold {k}: acc={acc:.4f}  f1={f1:.4f}")
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
print("DONE")
