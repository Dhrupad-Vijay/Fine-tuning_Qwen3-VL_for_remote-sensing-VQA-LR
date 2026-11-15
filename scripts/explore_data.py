import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from utils import load_splits

PTH = Path(__file__).resolve().parent.parent
LR_ROOT = PTH / "data/rsvqa_lr"

print("------------------------------------------------")
print("--------- RAW data format: Train split ---------")
print("------------------------------------------------")

for name in ['answers', 'images', 'questions']:
    data = json.load(open(f"{LR_ROOT}/LR_split_train_{name}.json"))
    key = name if name in data else data.keys()[0]
    print(f"\n{name}.json  top-level keys: {list(data)}  entries (incl. inactive): {len(data[key])}")
    print("  first entry:", data[key][0],"\n")

print("-----------------------------------------------------------")
print("--------- Splits and exploring 'active' questions ---------")
print("-----------------------------------------------------------")

splits = {split: load_splits(split) for split in ['train', 'test', 'val']}
"""
Each row in split['train/..'] is {"qid": --,  "img": --, "type": --, "question": --, "answer": --}
"""
print("------ X - X ------")
print("Each row in split['train/..'] is {'qid': --,  'img': --, 'type': --, 'question': --, 'answer': --}")
print("------ X - X ------")

for s, rows in splits.items():
    imgs = {r['img'] for r in rows}
    missing = sum(not Path(p).exists() for p in imgs)
    print(f"{s:5s}: {len(rows):6d} questions on {len(imgs):4d} images "
          f"(~{len(rows) / len(imgs):.0f} per image), missing image files: {missing}")
    print("types:", dict(Counter(r["type"] for r in rows)))
print('\n')

print("-----------------------------------------------------")
print("--------- Answers per question type (train) ---------")
print("-----------------------------------------------------")
    
train = splits['train']
for t in sorted({r["type"] for r in train}):
    of_type = [r for r in train if r["type"] == t]
    counts = Counter(r["answer"] for r in of_type)
    top = ", ".join(f"{a!r} {c / len(of_type):.0%}" for a, c in counts.most_common(6))
    print(f"\n{t}: {len(of_type)} questions, {len(counts)} distinct answers -> {top}")
    print(f"   {len({r['question'] for r in of_type})} distinct question strings")
    for q in list(dict.fromkeys(r["question"] for r in of_type))[:3]:
        print("   e.g.", q)

print("------------------------------------------------------------")
print("--------- Reappearing questions from test in train ---------")
print("------------------------------------------------------------")

test = splits['test']
train_qs = {r['question'] for r in train}
test_qs = [r['question'] for r in test]
cout = sum(q in train_qs for q in test_qs)
print(f"\nThis many questions of test repeat in train {cout} and percentage = {cout/len(test_qs) * 100:.2f} %")

print("-----------------------------------------------------")
print("--------- Image size and view of few images ---------")
print("-----------------------------------------------------")

first = Image.open(train[0]["img"])
print(f"\nImge size is {first.size}")
by_image = {}
for r in train:
    by_image.setdefault(r["img"], []).append(r)
fig, axes = plt.subplots(1, 4, figsize=(24, 8))
for ax, (path, qs) in zip(axes, list(by_image.items())[:4]):
    ax.imshow(Image.open(path).convert("RGB"))
    ax.axis("off")
    ax.set_title("\n".join(f"{q['question']} -> {q['answer']}" for q in qs[:5]), fontsize=8, loc="left")
out = Path("results/image_examples.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=120)
print("saved", out)

"""

------------------------------------------------
--------- RAW data format: Train split ---------
------------------------------------------------

answers.json  top-level keys: ['answers']  entries (incl. inactive): 77232
  first entry: {'id': 0, 'date_added': 1543570371.6343634, 'question_id': 0, 'people_id': 0, 'answer': 'urban', 'active': True} 


images.json  top-level keys: ['images']  entries (incl. inactive): 772
  first entry: {'id': 0, 'date_added': 1543570371.5997384, 'original_name': 'S2B_MSIL1C_20181010T104019_N0206_R008_T31UGS_20181010T161145_1024-2560.tif', 'sensor': 'S2', 'upperleft_map_x': 663601.4384423415, 'upperleft_map_y': 6650583.5827256255, 'res_x': '10m', 'res_y': '10m', 'people_id': 0, 'type': 'RGB', 'questions_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], 'active': True} 


questions.json  top-level keys: ['questions']  entries (incl. inactive): 77232
  first entry: {'id': 0, 'date_added': 1543570371.6343553, 'img_id': 0, 'people_id': 0, 'type': 'rural_urban', 'question': 'Is it a rural or an urban area', 'answers_ids': [0], 'active': True} 

-----------------------------------------------------------
--------- Splits and exploring 'active' questions ---------
-----------------------------------------------------------
------ X - X ------
Each row in split['train/..'] is {'qid': --,  'img': --, 'type': --, 'question': --, 'answer': --}
------ X - X ------
train:  57223 questions on  572 images (~100 per image), missing image files: 0
types: {'rural_urban': 572, 'presence': 17008, 'count': 17089, 'comp': 22554}
test :  10004 questions on  100 images (~100 per image), missing image files: 0
types: {'rural_urban': 100, 'presence': 2955, 'count': 2947, 'comp': 4002}
val  :  10005 questions on  100 images (~100 per image), missing image files: 0
types: {'rural_urban': 100, 'presence': 2919, 'count': 2966, 'comp': 4020}


-----------------------------------------------------
--------- Answers per question type (train) ---------
-----------------------------------------------------

comp: 22554 questions, 2 distinct answers -> 'no' 67%, 'yes' 33%
   3718 distinct question strings
   e.g. Are there more water areas than commercial buildings?
   e.g. Are there less roads than buildings?
   e.g. Are there more buildings than roads?

count: 17089 questions, 1259 distinct answers -> '0' 27%, '1' 8%, '2' 5%, '3' 4%, '4' 3%, '5' 2%
   1815 distinct question strings
   e.g. What is the amount of buildings?
   e.g. What is the number of commercial buildings in the image?
   e.g. What is the amount of residential buildings?

presence: 17008 questions, 2 distinct answers -> 'yes' 73%, 'no' 27%
   1496 distinct question strings
   e.g. Is there a water area?
   e.g. Is there a small water area?
   e.g. Is a building present?

rural_urban: 572 questions, 2 distinct answers -> 'rural' 58%, 'urban' 42%
   1 distinct question strings
   e.g. Is it a rural or an urban area
------------------------------------------------------------
--------- Reappearing questions from test in train ---------
------------------------------------------------------------

This many questions of test repeat in train 9289 and percentage = 92.85 %
-----------------------------------------------------
--------- Image size and view of few images ---------
-----------------------------------------------------

Imge size is (256, 256)
saved results/image_examples.png

"""