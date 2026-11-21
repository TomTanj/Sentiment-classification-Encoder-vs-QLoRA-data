import gc, os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------- CONFIG ----------------
MODEL = "Qwen/Qwen2-1.5B-Instruct"
NAME  = "Qwen2-1.5B QLoRA in-domain"
BATCH, ACCUM, EPOCHS, LR, MAXLEN = 4, 8, 3, 2e-4, 512
EVAL_B = 16
# ----------------------------------------
FOLD_DIR = r"D:\experiment\data-ai"
RES_CSV  = r"D:\experiment\results_cv.csv"
LABELS = ["negative", "positive"]
PROMPT = ("Classify the sentiment of the following text as positive or negative. "
          "Reply with exactly one word: positive or negative.\n\nText: {t}")

q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_use_double_quant=True,
                       bnb_4bit_compute_dtype=torch.float16)

class ClsDataset(Dataset):
    def __init__(self, df):
        self.items = []
        for t, l in zip(df["text"].astype(str), df["label"]):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(t=t[:2000])}],
                tokenize=True, add_generation_prompt=True)[:MAXLEN]
            ans = [tid[l], tok.eos_token_id]
            self.items.append({"input_ids": ids + ans,
                               "labels": [-100]*len(ids) + ans})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return {"input_ids": torch.tensor([b["input_ids"] + [pad]*(n-len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100]*(n-len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor([[1]*len(b["input_ids"]) + [0]*(n-len(b["input_ids"])) for b in batch])}

done = set()
if os.path.exists(RES_CSV):
    done = {(r["name"], int(r["fold"])) for _, r in pd.read_csv(RES_CSV).iterrows()}

def log_result(k, acc, f1):
    pd.DataFrame([{"name": NAME, "fold": k, "acc": round(acc,4), "f1": round(f1,4)}]
        ).to_csv(RES_CSV, mode="a", header=not os.path.exists(RES_CSV), index=False)
    print(f"  -> fold {k}: acc={acc:.4f}  f1={f1:.4f}")

for k in range(5):
    if (NAME, k) in done:
        print("skip", NAME, k); continue
    tok = AutoTokenizer.from_pretrained(MODEL)
    tid = {l: tok(l, add_special_tokens=False).input_ids[0] for l in LABELS}
    assert len(set(tid.values())) == 2, "label token collision - aborting"
    print(f"fold {k} | label tokens: {tid}")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q,
            device_map={"": 0}, torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    model.print_trainable_parameters()
    tr = pd.read_csv(rf"{FOLD_DIR}\fold{k}_train.csv")
    te = pd.read_csv(rf"{FOLD_DIR}\fold{k}_test.csv")
    print(f"fold {k}: training on {len(tr)} rows")
    args = TrainingArguments(output_dir=r"D:\experiment\models\_cv_tmp",
        num_train_epochs=EPOCHS, learning_rate=LR,
        per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUM,
        warmup_ratio=0.05, weight_decay=0.01, lr_scheduler_type="cosine",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit", fp16=True, logging_steps=10,
        save_strategy="no", report_to="none", seed=42)
    trainer = Trainer(model=model, args=args,
                      train_dataset=ClsDataset(tr), data_collator=collate)
    trainer.train()
    model.eval(); tok.padding_side = "left"
    preds = []
    with torch.no_grad():
        texts = te["text"].astype(str).tolist()
        for i in range(0, len(texts), EVAL_B):
            prompts = [tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(t=t)}],
                tokenize=False, add_generation_prompt=True) for t in texts[i:i+EVAL_B]]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            logp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
            s = torch.stack([logp[:, tid[l]] for l in LABELS], dim=1)
            preds += [LABELS[j] for j in s.argmax(1).tolist()]
    log_result(k, accuracy_score(te["label"], preds),
                  f1_score(te["label"], preds, labels=LABELS, average="macro"))
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
print("DONE")
