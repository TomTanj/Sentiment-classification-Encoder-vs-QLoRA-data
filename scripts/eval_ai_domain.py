import gc
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForCausalLM, BitsAndBytesConfig)
from peft import PeftModel

DF    = pd.read_csv(r"D:\experiment\data\test.csv")
TEXTS = DF["text"].astype(str).tolist()
GOLD  = DF["label"].tolist()
LABELS = ["negative", "positive"]
HAS_LEN = "length" in DF.columns
print("rows:", len(DF), "| labels:", pd.Series(GOLD).value_counts().to_dict())

PROMPT = ("Classify the sentiment of the following text as positive or negative. "
          "Reply with exactly one word: positive or negative.\n\nText: {t}")
results = {}

def report(name, preds):
    acc = accuracy_score(GOLD, preds)
    f1m = f1_score(GOLD, preds, labels=LABELS, average="macro")
    print(f"\n=== {name} ===")
    print(f"accuracy : {acc:.4f}   f1_macro : {f1m:.4f}")
    print(confusion_matrix(GOLD, preds, labels=LABELS))
    if HAS_LEN:
        for cat in ["long", "short"]:
            mask = (DF["length"] == cat).tolist()
            sp = [p for p, m in zip(preds, mask) if m]
            sg = [g for g, m in zip(GOLD, mask) if m]
            if sp:
                print(f"   {cat:5s} n={len(sp):4d}  acc={accuracy_score(sg, sp):.4f}")
    results[name] = (acc, f1m)

def free():
    gc.collect(); torch.cuda.empty_cache()

# ---------- 1-2) fine-tuned encoders ----------
for name, ckpt in [("ELECTRA-base", r"D:\experiment\models\electra-base\best"),
                   ("DeBERTaV3-base", r"D:\experiment\models\deberta-v3-base\best")]:
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).cuda().eval()
    id2lab = {int(k): v for k, v in model.config.id2label.items()}
    preds = []
    with torch.no_grad():
        for i in range(0, len(TEXTS), 32):
            enc = tok(TEXTS[i:i+32], truncation=True, max_length=256,
                      padding=True, return_tensors="pt").to("cuda")
            preds += [id2lab[j] for j in model(**enc).logits.argmax(-1).tolist()]
    report(name + " (fine-tuned TweetEval)", preds)
    del model; free()

# ---------- 3) LLMs ----------
def tid_of(tok):
    tid = {}
    for lab in LABELS:
        ids = tok(lab, add_special_tokens=False).input_ids
        assert len(ids) == 1, f"{lab} multi-token"
        tid[lab] = ids[0]
    return tid

def llm_scores(model, tok, tid, batch=16):
    tok.padding_side = "left"
    preds = []
    with torch.no_grad():
        for i in range(0, len(TEXTS), batch):
            chunk = TEXTS[i:i+batch]
            prompts = [tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(t=t)}],
                tokenize=False, add_generation_prompt=True) for t in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            logp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
            sc = torch.stack([logp[:, tid[l]] for l in LABELS], dim=1)
            preds += [LABELS[j] for j in sc.argmax(1).tolist()]
    return preds

Q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_use_double_quant=True,
                       bnb_4bit_compute_dtype=torch.float16)

# Qwen zero-shot
M = "Qwen/Qwen2-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, quantization_config=Q,
        device_map={"": 0}, torch_dtype=torch.float16).eval()
report("Qwen2-1.5B (zero-shot)", llm_scores(model, tok, tid_of(tok)))
del model; free()

# Gemma: zero-shot + QLoRA (load base ONCE)
M = "unsloth/gemma-2-2b-it-bnb-4bit"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, quantization_config=Q,
        device_map={"": 0}, torch_dtype=torch.float16).eval()
report("Gemma-2-2B (zero-shot)", llm_scores(model, tok, tid_of(tok)))

model = PeftModel.from_pretrained(model, r"D:\experiment\models\gemma-2-2b-qlora").eval()
report("Gemma-2-2B (QLoRA, TweetEval-trained)", llm_scores(model, tok, tid_of(tok)))
del model; free()

# ---------- final table ----------
print("\n" + "=" * 68)
print(f"AI-DOMAIN TEST SET (n={len(DF)}) - domain transfer from TweetEval")
print("=" * 68)
print(f"{'model':42s}{'acc':>8s}{'f1_macro':>10s}")
for name, (acc, f1m) in results.items():
    print(f"{name:42s}{acc:8.4f}{f1m:10.4f}")
