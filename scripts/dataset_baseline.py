import os
import sys
import json
import torch
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from utils import load_splits, normalize, count_binning

def compare(ques_ans_lookup, type_ans_lookup, ds_type='test'):
    """
    Using train lookup dict to answer test/val questions
    based on highest answer count per question or type
    """
    result = []
    res_type = defaultdict(list)
    invalid = 0
    for row in load_splits(ds_type):
        qid = row['qid']
        pred_ans = None
        answer = row['answer']
        if row['question'] in ques_ans_lookup:
            pred_ans = ques_ans_lookup[row['question']]
        else:
            pred_ans = type_ans_lookup[row['type']]
        if row['type'] == "count":
            pred_ans = count_binning(pred_ans)
            answer = count_binning(answer)
        res_type[row['type']].append(pred_ans == answer)
        if pred_ans != answer:
            invalid += 1
        result.append({'qid': qid, 'pred_ans': pred_ans, 'answer': answer})

    correct = sum(sum(v) for k,v in res_type.items())
    per_type_acc = {k: round(sum(v)/len(v) * 100, 2) for k,v in res_type.items()}
    avg_acc = sum([v for k,v in per_type_acc.items()])/len(per_type_acc)

    return {"n": len(result),
            "overall_acc": round(correct/len(result) * 100, 2),
            "per_type_acc": per_type_acc,
            "avg_acc": round(avg_acc, 2),
            "invalid_rate": round(invalid/len(result) * 100, 2)}

def majority(lookup):
    return {k: Counter(v).most_common(1)[0][0] for k, v in lookup.items()}

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="majority_census")
    args = ap.parse_args()
    save_pth = Path("results")
    save_pth.mkdir(parents=True, exist_ok=True)
    res_dict = {}

    # Create a lookup with the train questions and type
    ques_ans_lookup = defaultdict(list)
    type_ans_lookup = defaultdict(list)

    for train_row in load_splits("train"):
        ques_ans_lookup[train_row['question']].append(train_row['answer'])
        type_ans_lookup[train_row['type']].append(train_row['answer'])

    ques_ans_lookup = majority(ques_ans_lookup)
    type_ans_lookup = majority(type_ans_lookup)

    print(f"The number of unique questions in train = {len(ques_ans_lookup)} unique types = {len(type_ans_lookup)}")

    # Lookup answers for test and compare
    test_dict = compare(ques_ans_lookup, type_ans_lookup, ds_type="test")
    print(f"Correctly predicted test avg is = {test_dict['overall_acc']:.02f}%")
    print(f"Per type test accuracy is:\n{test_dict['per_type_acc']}")
    res_dict['Test_dataset'] = test_dict

    # Same process on the val dataset
    val_dict = compare(ques_ans_lookup, type_ans_lookup, ds_type="val")
    print(f"Correctly predicted val avg is = {val_dict['overall_acc']:.02f}%")
    print(f"Per type val accuracy is:\n{val_dict['per_type_acc']}")
    res_dict['Val_dataset'] = val_dict

    (save_pth / f"baseline_{args.out}.json").write_text(json.dumps(res_dict, indent=2))

if __name__ == "__main__":
    main()