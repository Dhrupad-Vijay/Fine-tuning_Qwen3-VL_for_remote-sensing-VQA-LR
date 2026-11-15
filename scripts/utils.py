import json
from pathlib import Path
from collections import defaultdict
import re

PTH = Path(__file__).resolve().parent.parent
LR_ROOT = PTH / "data/rsvqa_lr"

def load_splits(split):
    """
    Each row in split['train/..'] is {'qid': --,  'img': --, 'type': --, 'question': --, 'answer': --}
    """
    
    questions = json.load(open(f"{LR_ROOT}/LR_split_{split}_questions.json"))['questions']
    answers = json.load(open(f"{LR_ROOT}/LR_split_{split}_answers.json"))['answers']
    answer_by_id = {}
    no_answer = 0

    for a in answers:
        if 'answer' not in a.keys():
            continue
        else:
            answer_by_id[a['id']] = a['answer']
    rows = []
    for q in questions:
        if not q.get("active", True):
            continue
        if q['answers_ids'][0] not in answer_by_id:
            no_answer+=1
            continue
        rows.append({
            "qid": q["id"],
            "img": f"{LR_ROOT}/Images_LR/{q['img_id']}.tif",
            "type": q["type"],
            "question": q["question"],
            "answer": answer_by_id[q["answers_ids"][0]],
        })
    return rows

def answer_options(train_rows, max_options=10):
    options = defaultdict(set)
    for row in train_rows:
        options[row['type']].add(row['answer'])
    for k,v in options.items():
        if len(v) > max_options:
            options[k] = None
        else:
            options[k] = list(v)
    return options

def build_prompt(question, options):
    if options:
        return f"{question}\nanswer with only the following options: {','.join(options)}"
    return f"{question}\nanswer with only a number or single phrase or word."

def normalize(text):
    text = text.strip().lower().strip(".,!*")
    words = text.replace(",", " ").replace(".", " ").split()
    if words and words[0] in ("yes", "no"):
        return words[0]
    return text

def count_binning(txt):
    tnum = re.search(r'\d+', txt)
    if tnum is None:
        return "invalid"
    num = int(tnum.group())

    if num == 0: return 0
    if num <= 10: return "1-10"
    if num <= 100: return "11-100"
    if num <= 500: return "101-500"
    if num <= 1000: return "501-1000"
    return ">1000"

def scorer(preds, rows, options):
    assert len(preds) == len(rows), "The number of inputs is not equal to predictions"
    invalid_outputs = 0
    res_type = defaultdict(list)
    wrong = []

    for row, pred in zip(rows, preds):
        answer = normalize(row['answer'])
        pred = normalize(pred)
        allowed = options.get(row['type'])
        if allowed is not None and pred not in allowed:
            invalid_outputs += 1
        if row['type'] == "count":
            pred = count_binning(pred)
            answer = count_binning(answer)

        res_type[row['type']].append(pred == answer)
        if pred == answer:
            wrong.append({'row': row, 'pred': pred})
    correct = sum(sum(v) for k,v in res_type.items())
    per_type_acc = {k: round(sum(v)/len(v) * 100, 2) for k,v in res_type.items()}
    avg_acc = sum([v for k,v in per_type_acc.items()])/len(per_type_acc)
    
    return {"n": len(rows),
            "overall_acc": round(correct/len(rows) * 100, 2),
            "per_type_acc": per_type_acc,
            "avg_acc": round(avg_acc, 2),
            "invalid_rate": round(invalid_outputs/len(rows) * 100, 2),
            "wrong_res": wrong}

def model_pth(model):
    root = PTH / "models"
    if model in ['2B', '4B', '8B']:
        return root / f"Qwen3-VL-{model}-Instruct"
    else:
        return root / f"Qwen3-VL-2B-Instruct"