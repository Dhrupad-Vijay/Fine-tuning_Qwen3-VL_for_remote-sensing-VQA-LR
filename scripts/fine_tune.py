import argparse
import json
from pathlib import Path
import torch
from PIL import Image
import yaml
import random
import shutil
import math

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup
from utils import load_splits, answer_options, build_prompt, model_pth

PTH = Path(__file__).resolve().parent.parent

def make_batch(rows, processor, options):
    full_texts, prompt_lens, images = [], [], []
    for r in rows:
        image = Image.open(r["img"]).convert("RGB")
        
        # User prompt
        content = [{'type': 'image'}, {'type': 'text', 'text': build_prompt(r['question'], options[r['type']])}]
        user_prompt = {'role': 'user', 'content': content}

        # Assistant output
        assist_content = [{'type': 'text', 'text': r['answer']}]
        assistant = {'role': 'assistant', 'content': assist_content}

        prompt = processor.apply_chat_template([user_prompt], tokenize=False, add_generation_prompt=True)
        full_out = processor.apply_chat_template([user_prompt, assistant], tokenize=False)
        assert full_out.startswith(prompt), "Prompt template need to be followed"

        full_texts.append(full_out)
        prompt_lens.append(processor(text=[prompt], images=[image], return_tensors="pt").input_ids.shape[1])
        images.append(image)

    # Create '-100' labels for user prompt tokens and padding
    batch = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
    labels = batch.input_ids.clone()
    for i,j in enumerate(prompt_lens):
        labels[i, :j] = -100
    labels[batch.attention_mask == 0] = -100
    batch["labels"] = labels

    return batch

@torch.no_grad()
def val_loss(model, rows, processor, options, bs):
    model.eval()
    total = 0.0
    for i in range(0, len(rows), bs):
        chunk = rows[i : i + bs]
        total += model(**make_batch(chunk, processor, options).to("cuda")).loss.item() * len(chunk)
    model.train()
    return total / len(rows)

@torch.no_grad()
def show_predictions(model, rows, processor, options):
    model.eval()
    for r in rows:
        content = [{'type': 'image'}, {'type': 'text', 'text': build_prompt(r['question'], options[r['type']])}]
        user_prompt = {'role': 'user', 'content': content}
        text = processor.apply_chat_template([user_prompt], tokenize=False, add_generation_prompt=True)
        image = Image.open(r["img"]).convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        pred = processor.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  [{r['type']}] {r['question']} | truth: {r['answer']} | model prediction: {pred}")
    model.train()

def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--config", required=True)
    args = argp.parse_args()
    cfg = yaml.safe_load(open(args.config))
    
    seed_ = cfg["seed"]
    random.seed(seed_)
    torch.manual_seed(seed_)
    out_dir = PTH / "results/runs" / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")

    # Get the train and val data rows
    train = load_splits("train")
    if cfg["limit_train"] != "None":
        train = train[:cfg["limit_train"]]
    options = answer_options(load_splits("train"))
    val = load_splits("val")
    val = val[:cfg["val_loss_n"]]    # Reduce the val evaluation footprint for now!
    random.Random(0).shuffle(val)

    # Set model & processor + LoRA 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = model_pth(cfg["model"])
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = 'right'
    lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    assert not any("lora" in n and "visual" in n for n, _ in model.named_parameters()), "LoRA leaked into vision"

    # Test for batching logic
    # first = make_batch(train[:1], processor, options)
    # supervised = first.input_ids
    # print(processor.batch_decode(supervised, skip_special_tokens=True))

    # Optimizer and LR scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    bs, accum = cfg["batch_size"], cfg["grad_accum"]
    total_steps = cfg["epochs"] * math.ceil(len(train) / (bs * accum))
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)
    print(f"{len(train)} train questions, {total_steps} optimizer steps")

    # Train loop
    model.train()
    step, running = 0, []
    log = open(out_dir / "log.jsonl", "a")
    for epoch in range(cfg["epochs"]):
        random.shuffle(train)
        batches = [train[i : i + bs] for i in range(0, len(train), bs)]
        for k,batch in enumerate(batches):
            loss = model(**make_batch(batch, processor, options).to(device)).loss
            (loss/accum).backward()
            running.append(loss.item())
            if (k + 1) % accum != 0 and k != len(batches) - 1:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % cfg["log_every"] == 0:
                entry = {"step": step, "epoch": epoch, "train_loss": sum(running) / len(running),
                         "lr": scheduler.get_last_lr()[0]}
                running = []
                print(entry, flush=True)
                log.write(json.dumps(entry) + "\n"); log.flush()

            if step % cfg["eval_every"] == 0 or step == total_steps:
                entry = {"step": step, "val_loss": val_loss(model, val, processor, options, bs)}
                print(entry, flush=True)
                log.write(json.dumps(entry) + "\n"); log.flush()
                model.save_pretrained(out_dir / f"step_{step}")   # adapter only

    print("\nGreedy answers after training:")
    show_predictions(model, train[:15] if cfg["limit_train"] else val[:15], processor, options)

if __name__ == '__main__':
    main()