import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- CONFIG ----------------
MODEL = r"D:\experiment\models\deberta-v3-base\best"
NAME  = "DeBERTaV3 transfer"
# ----------------------------------------
FOLD_DIR = r"D:\experiment\data-ai"
RES_CSV  = r"D:\experiment\results_cv.csv"
LABELS = ["negative", "positive"]
BATCH = 32

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).cuda().eval()
id2lab = {int(i): s for i, s in model.config.id2label.items()}
print("classes:", id2lab)

done = set()
if os.path.exists(RES_CSV):
    done = {(r["name"], int(r["fold"])) for _, r in pd.read_csv(RES_CSV).iterrows()}

@torch.no_grad()
def predict(texts):
    preds = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i+BATCH], truncation=True, max_length=256,
                  padding=True, return_tensors="pt").to("cuda")
        preds += [id2lab[j] for j in model(**enc).logits.argmax(-1).tolist()]
    return preds

for k in range(5):
    if (NAME, k) in done:
        print("skip", NAME, k); continue
    df = pd.read_csv(rf"{FOLD_DIR}\fold{k}_test.csv")
    p = predict(df["text"].astype(str).tolist())
    acc = accuracy_score(df["label"], p)
    f1 = f1_score(df["label"], p, labels=LABELS, average="macro")
    pd.DataFrame([{"name": NAME, "fold": k, "acc": round(acc,4), "f1": round(f1,4)}]
        ).to_csv(RES_CSV, mode="a", header=not os.path.exists(RES_CSV), index=False)
    print(f"  -> fold {k}: acc={acc:.4f}  f1={f1:.4f}")
print("DONE")
