import argparse
import json
from pathlib import Path
import os
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from utils import load_splits, answer_options, build_prompt, scorer, model_pth
from tqdm import tqdm
from peft import PeftModel

PTH = Path(__file__).resolve().parent.parent

def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--model", required=True, choices=['2B', '4B', '8B'])
    argp.add_argument("--split", default='val', choices=['test', 'val'])
    argp.add_argument("--out", required=True)
    argp.add_argument("--limit", type=int, default=None)
    argp.add_argument("--blind", action="store_true")
    argp.add_argument("--batch_size", type=int, default=32)
    argp.add_argument("--adapter", default=None)
    args = argp.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_splits(args.split)[:args.limit]
    # Have to create a list of allowed answers for the LLM will be
    # created from the train answers based on question type
    options = answer_options(load_splits("train"))
    modelpt = model_pth(args.model)

    processor = AutoProcessor.from_pretrained(modelpt)
    processor.tokenizer.padding_side = 'left'
    model = AutoModelForImageTextToText.from_pretrained(modelpt, dtype=torch.bfloat16).to(device).eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().to(device).eval()

    pbar = tqdm(total=int(len(rows)/args.batch_size), desc="Evaluating Dataset")

    preds = []
    for row in range(0, len(rows), args.batch_size):
        batch = rows[row : row + args.batch_size]
        texts, images = [], []

        # ---------- Input processing ----------
        for r in batch:
            content = [{'type': 'text', 'text': build_prompt(r['question'], options[r['type']])}]
            if not args.blind:
                content.insert(0, {'type': 'image'})
                images.append(Image.open(r['img']).convert("RGB"))
            message = [{'role': 'user', 'content': content}]
            texts.append(processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
        
        inputs = processor(text=texts, images=images or None, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)
                ]
        preds += processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        pbar.update(1)

    # ---------- Evaluate metrics ----------
    metrics = scorer(preds, rows, options)
    metrics["args"] = vars(args)
    save_pth = PTH / "results"
    save_pth.mkdir(parents=True, exist_ok=True)
    file = save_pth / f"{args.out}_{args.split}_{args.model}.json"
    file.write_text(json.dumps(metrics, indent=2))
    print(f"\nEvaluation of {args.split} done and saved at {file}")

if __name__ == "__main__":
    main()