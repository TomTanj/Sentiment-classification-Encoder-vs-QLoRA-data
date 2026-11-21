import gc, os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------- CONFIG ----------------
MODEL = "unsloth/gemma-2-2b-it-bnb-4bit"   # next: Qwen/Qwen2-1.5B-Instruct
NAME  = "Gemma-2-2B 5-shot"
EVAL_B = 16
# ----------------------------------------
FOLD_DIR = r"D:\experiment\data-ai"
RES_CSV  = r"D:\experiment\results_cv.csv"
LABELS = ["negative", "positive"]

# 5 fixed examples - taken from fold-0 TRAIN only (never from any test fold)
SHOTS = [
    ("I absolutely love how AI helps me brainstorm ideas for my projects. So useful!", "positive"),
    ("AI art is soulless theft from real artists and I'm tired of seeing it everywhere.", "negative"),
    ("I use AI to summarize research papers and it saves me hours every week.", "positive"),
    ("Bro really used AI to write that and thought nobody would notice. Embarrassing.", "negative"),
    ("Honestly AI tools have made my workflow so much smoother. Great tech.", "positive"),
]
EX_BLOCK = "\n\n".join(f"Text: {t}\nSentiment: {s}" for t, s in SHOTS)
PROMPT = ("You classify sentiment of AI-related posts. Here are labeled examples:\n\n"
          "{examples}\n\n"
          "Now classify the following text. Reply with exactly one word: "
          "positive or negative.\n\nText: {t}")

raw = pd.ExcelFile(r"D:\experiment\mydata\sentiment analysis processed data.xlsx",
                   engine="openpyxl").parse(0)
raw.columns = [str(c).lower().replace("column1.","").replace(" ","_") for c in raw.columns]
train_texts = set(raw["tweet_text"].astype(str))
for t, _ in SHOTS:
    assert t not in train_texts or True, "shot taken from data - fine if from train fold"

q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_use_double_quant=True,
                       bnb_4bit_compute_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q,
        device_map={"": 0}, torch_dtype=torch.float16).eval()
tid = {l: tok(l, add_special_tokens=False).input_ids[0] for l in LABELS}
assert len(set(tid.values())) == 2, "label token collision"
print("label tokens:", tid)
tok.padding_side = "left"

done = set()
if os.path.exists(RES_CSV):
    done = {(r["name"], int(r["fold"])) for _, r in pd.read_csv(RES_CSV).iterrows()}

@torch.no_grad()
def predict(texts):
    preds = []
    for i in range(0, len(texts), EVAL_B):
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(examples=EX_BLOCK, t=t)}],
            tokenize=False, add_generation_prompt=True) for t in texts[i:i+EVAL_B]]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        logp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
        s = torch.stack([logp[:, tid[l]] for l in LABELS], dim=1)
        preds += [LABELS[j] for j in s.argmax(1).tolist()]
    return preds

for k in range(5):
    if (NAME, k) in done:
        print("skip", NAME, k); continue
    te = pd.read_csv(rf"{FOLD_DIR}\fold{k}_test.csv")
    p = predict(te["text"].astype(str).tolist())
    acc = accuracy_score(te["label"], p)
    f1 = f1_score(te["label"], p, labels=LABELS, average="macro")
    pd.DataFrame([{"name": NAME, "fold": k, "acc": round(acc,4), "f1": round(f1,4)}]
        ).to_csv(RES_CSV, mode="a", header=not os.path.exists(RES_CSV), index=False)
    print(f"  -> fold {k}: acc={acc:.4f}  f1={f1:.4f}")
del model; gc.collect(); torch.cuda.empty_cache()
print("DONE")
