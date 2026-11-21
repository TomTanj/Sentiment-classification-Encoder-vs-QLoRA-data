import time
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ================= CONFIG =================
MODEL = "Qwen/Qwen2-1.5B-Instruct"
NAME  = "qwen2-1.5b-zero-shot"
BATCH = 16
LIMIT = 0
# ==========================================

LABELS = ["negative", "positive"]

q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_use_double_quant=True,
                       bnb_4bit_compute_dtype=torch.float16)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=q, device_map="cuda",
    torch_dtype=torch.float16)
model.eval()

# sanity check: each label must be ONE token (needed for the scoring trick)
tid = {}
for lab in LABELS:
    ids = tok(lab, add_special_tokens=False).input_ids
    print(f"label '{lab}' -> token ids {ids}")
    assert len(ids) == 1, f"'{lab}' is multi-token - stop and tell me"
    tid[lab] = ids[0]

tok.padding_side = "left"   # required for batched decoder scoring

PROMPT = ("Classify the sentiment of the following text as positive or negative. "
          "Reply with exactly one word: positive or negative.\n\nText: {t}")

@torch.no_grad()
def classify(texts):
    preds = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i+BATCH]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(t=t)}],
            tokenize=False, add_generation_prompt=True) for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        logits = model(**enc).logits[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        scores = torch.stack([logp[:, tid[l]] for l in LABELS], dim=1)
        preds += [LABELS[j] for j in scores.argmax(1).tolist()]
        if (i // BATCH) % 10 == 0:
            print(f"  {i+len(chunk)}/{len(texts)}", flush=True)
    return preds

df = pd.read_csv(r"D:\experiment\data\test.csv")
if LIMIT: df = df.iloc[:LIMIT].copy()
texts = df["text"].astype(str).tolist()

t0 = time.time()
preds = classify(texts)
print(f"\n=== {NAME} | {MODEL} | {len(df)} rows ===")
print("accuracy :", round(accuracy_score(df["label"], preds), 4))
print("f1_macro :", round(f1_score(df["label"], preds, average="macro"), 4))
print("confusion (rows=true, cols=pred):", LABELS)
print(confusion_matrix(df["label"], preds, labels=LABELS))
print("runtime  :", round(time.time()-t0), "s")
