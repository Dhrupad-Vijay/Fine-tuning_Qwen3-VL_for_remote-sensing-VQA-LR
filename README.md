# Fine-tuning Qwen3-VL-4B for remote-sensing VQA (RSVQA-LR)

This repo contains the full pipeline for fine-tuning (LoRA) of [Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) on RSVQA-LR dataset. RSVQA-LR is a visual question answering benchmark on 10 m Sentinel-2 images. The evaluation is conducted to understand what the model can inherently understand (zero-shot testing) and how far fine-tuning can improve results and separates what the model can learn through images and what it can guess from text questins alone.

**TL;DR**

- LoRA on the VLM raises test accuracy from 57.3% (zero-shot) to **86.6%**. On the question types reported by recent vision-language models, this is in the same range as published fine-tuned models.
- A data baseline that never sees the images (the most frequent training answer for each question text) already reaches 74.4% overall accuracy, with 88.3% on presence and 87.1% on comparison. Showing repetition of questions and how much of the benchmark can be answered from questions.
- Over that prior, fine-tuning adds +3.7 points on presence and +6.1 on comparison, but +28.2 on counting and +38.0 on rural/urban, the two types where the question gives little away and image context is most important.

<!-- TODO: adapter link once uploaded -->
LoRA adapter: [Download here](https://huggingface.co/Random-Sandwiche/rsvqa-qwen3-vl-lora) from the Hugging Face Hub.

## Results

Test split, evaluated once after all choices were made on validation. Count is scored after binning (see [Evaluation](#evaluation)).

| Method | Presence | Comparison | Count | Rural/urban | AA | OA |
|---|---:|---:|---:|---:|---:|---:|
| Majority answer per question (no image) | 88.3 | 87.1 | 43.7 | 56.0 | 68.8 | 74.4 |
| Qwen3-VL-4B zero-shot, no image | 72.2 | 70.2 | 25.0 | 56.0 | 55.9 | 57.4 |
| Qwen3-VL-4B zero-shot | 64.5 | 72.5 | 28.9 | 70.0 | 59.0 | 57.3 |
| **Qwen3-VL-4B + LoRA** | **92.0** | **93.2** | **71.9** | **94.0** | **87.8** | **86.6** |
| *LoRA minus no-image prior* | *+3.7* | *+6.1* | *+28.2* | *+38.0* | *+19.0* | *+12.2* |

AA (average accuracy) is the mean of the four per-type accuracies and is the selection metric. OA (overall accuracy) is accuracy over all questions.

<details>
<summary>Validation results</summary>

| Method | Presence | Comparison | Count | Rural/urban | AA | OA |
|---|---:|---:|---:|---:|---:|---:|
| Majority answer per question (no image) | 90.8 | 90.7 | 42.3 | 50.0 | 68.4 | 76.0 |
| Qwen3-VL-4B zero-shot, no image | 73.4 | 71.1 | 23.1 | 50.0 | 54.4 | 57.3 |
| Qwen3-VL-4B zero-shot | 62.7 | 73.4 | 27.6 | 67.0 | 57.7 | 56.6 |
| Qwen3-VL-4B + LoRA | 92.2 | 95.5 | 75.3 | 91.0 | 88.5 | 88.5 |

</details>

## Findings

**1. The question carries much of the answer.** RSVQA-LR questions come from a small set of templates and the answers come from OpenStreetMap, so answer distributions per question are very skewed. Predicting the most frequent training answer for each question text, without looking at the image, scores 88.3% on presence and 87.1% on comparison. High accuracy on these two types, for this model or any other, therefore says little about visual understanding. This agrees with earlier work on language bias in RSVQA (Chappuis et al., 2023).

**2. Where the image is needed, fine-tuning helps a lot.** The gains over the prior are concentrated in counting (+28.2) and rural/urban (+38.0). Rural/urban is a pure image test, because every image gets the same question, so the prior can only ever predict one class. It has only 100 test questions, though: 94% correct has a 95% confidence interval of roughly 87–97%. For the larger types (several thousand test questions each) the interval is about ±1–2 points.

**3. Zero-shot, adding the image hurts on presence.** Without the image the base model scores 72.2% on presence, with it 64.5% (val: 73.4% vs 62.7%). The image helps on the other three types, and overall accuracy is unchanged (57.4% vs 57.3%). A plausible explanation, not verified here: at 10 m, many objects that OpenStreetMap records (single buildings, small roads) are not visible, so the model answers "no" where the map-derived answer is "yes". The RSVQA authors note this resolution limit themselves. Comparing the share of "yes" in predictions and ground truth would test it.

## Error examples

Wrong predictions of the fine-tuned model on the test split. These are a handful of examples, not a systematic analysis. Count predictions are shown after binning.

<!-- TODO: add a PNG of each image (GitHub does not display .tif), e.g. assets/errors/289.png -->

| Type | Question | Truth | Prediction | Image |
|---|---|---|---|---|
| Count | What is the amount of square commercial buildings? | 1–10 | 0 | 289 |
| Count | How many commercial buildings are there? | 11–100 | 1–10 | 279 |
| Presence | Is a medium commercial building present? | no | yes | 305 |
| Presence | Is a commercial building present? | yes | no | 235 |
| Comparison | Are there less residential buildings than farmlands? | no | yes | 324 |
| Comparison | Are there less roads than water areas? | yes | no | 279 |
| Rural/urban | Is it a rural or an urban area? | urban | rural | 256 |
| Rural/urban | Is it a rural or an urban area? | rural | urban | 252 |

What these examples suggest:

- Both count errors are one bin too low. A commercial building is only a few pixels at 10 m. A confusion matrix over the count bins would show whether undercounting by one bin is the main error mode.
- Size and shape words ("medium", "square") come from the geometry of OpenStreetMap polygons and are hard to verify at this resolution.
- Comparison answers depend on how objects are mapped in OpenStreetMap (for example how many separate road and water objects exist), which is not always inferable from the image.
<!-- TODO: one line on the two rural/urban images once you have looked at them -->

## Comparison with published results

| Method | Presence | Comparison | Rural/urban | Count |
|---|---:|---:|---:|---:|
| RSVQA (Lobry et al., 2020) | 87.46 | 81.50 | 90.00 | 67.01 |
| RSGPT, fine-tuned on RSVQA-LR (Hu et al., 2023) | 91.17 | 91.70 | 94.00 | – |
| GeoChat (Kuckreja et al., 2024) | 91.09 | 90.33 | 94.00 | – |
| **Qwen3-VL-4B + LoRA (this repo)** | **92.0** | **93.2** | **94.0** | **71.9** |
| Majority answer per question, no image (this repo) | 88.3 | 87.1 | 56.0 | 43.7 |

Published numbers are taken from the RSGPT and GeoChat papers. Recent VLM papers leave out count questions. The RSVQA paper uses the same five count bins as this repo.

How to read this:

- On presence, comparison and rural/urban, this result is in the same range as RSGPT and GeoChat. Differences of 1–2 points are within what different prompts, answer parsing, question filtering (this repo skips questions marked inactive in the Zenodo release) and a single training run can produce, so I do not claim an improvement.
- On rural/urban, all three recent models score 94.0 on a 100-question test set, so this type cannot separate them.
- The no-image prior scores above the original RSVQA model on presence and comparison (the question sets may differ slightly, see above). This is one more reason to report a prior baseline next to accuracy.

## Setup

### Data

RSVQA-LR (Lobry et al., 2020): 772 Sentinel-2 RGB tiles of 256×256 pixels at 10 m over the Netherlands, with 77,232 questions generated from templates and answered from OpenStreetMap. Four question types: presence, comparison, count and rural/urban. I use the official train/val/test splits from [Zenodo record 6344334](https://zenodo.org/records/6344334) (CC-BY-4.0). Questions marked inactive, and answers removed in that release, are skipped.

<!-- TODO: number of questions per split and type after filtering -->

### Model and fine-tuning

- Base model: Qwen3-VL-4B-Instruct, revision pinned in `manifest.json`.
- LoRA (PEFT) on the language model's linear layers: q, k, v, o, gate, up, down. Vision encoder and vision-language merger frozen. r = 16, alpha = 32, dropout 0.05.
- AdamW, learning rate 2e-4, cosine schedule with 3% warmup, batch 8 with gradient accumulation 2, 2 epochs (7,154 optimizer steps), gradient clipping 1.0, bf16.
- Loss only on the answer tokens and the end-of-turn token. Prompt, image tokens and padding are masked.
- Plain PyTorch training loop. Before the full run, the loop was checked by overfitting 50 samples to perfect accuracy.

<!-- TODO: GPU(s) used and training time -->

### Prompt

Built with Qwen's chat template and identical in training and evaluation. For question types with a small answer set (yes/no, rural/urban) the prompt lists the allowed answers, taken from the training split: `<question> Answer with one of: <options>.` Otherwise: `<question> Answer the question using a single word or phrase.`

### Evaluation

- Greedy decoding, batched generation with left padding, bf16.
- Predictions are normalized and matched exactly against the ground truth. Unparseable answers count as wrong; none occurred in any model run.
- Count: ground truth and prediction (the first number in the output) are binned into 0 / 1–10 / 11–100 / 101–1000 / >1000, as in the RSVQA paper, and accuracy is computed on the bins.
- All choices (prompt, hyperparameters, checkpoint) were made on validation. Test was evaluated once.
- Checkpoint: several saved checkpoints were evaluated on validation and the one with the highest AA, step 7,154, is reported. <!-- TODO: confirm and list val AA per checkpoint -->

### Baselines

- **Majority answer per question (no image):** for each exact question text, the most frequent answer in the training split. <!-- TODO: fallback for question texts not seen in training -->
- **Zero-shot, no image:** the base model with the same prompt, image removed.
- **Zero-shot:** the base model with image and the same prompt.

## Limitations

- Single training run and seed. Batched bf16 generation can change results slightly between runs.
- During training, a loss curve was logged on the test split by mistake. It was not used for any decision, but the test set was not completely untouched during development.
- High accuracy on presence and comparison mostly reflects the language prior (see Findings).
- Labels come from OpenStreetMap and are noisy and incomplete. Some questions are template artefacts, e.g. "more commercial buildings than commercial buildings".
- Count is scored on five coarse bins, using the first number in the output.
- Only the language model was adapted. Whether adapting the vision encoder helps was not tested.
- RSVQA-LR covers one country with one sensor (Sentinel-2 RGB). No claim is made about other regions or sensors.

Possible next steps: accuracy per type on only the questions the prior gets wrong, LoRA on the vision encoder, and evaluation on RSVQA-HR.

## Reproduce

Python 3.12 with [uv](https://docs.astral.sh/uv/). torch is pinned to the CUDA 12.6 build in `pyproject.toml`.

<!-- TODO: replace with the exact commands -->
```bash
uv sync
uv run python dw_assets.py                                  # dataset and models at pinned revisions
uv run python scripts/dataset_baseline.py                   # majority-answer baseline
uv run python scripts/fine_tune.py --config configs/fine_tune.yaml
uv run python scripts/evaluate.py --model <model> --split test                         # zero-shot
uv run python scripts/evaluate.py --model <model> --split test --blind                 # zero-shot, no image
uv run python scripts/evaluate.py --model <model> --split test --adapter <adapter-dir> # fine-tuned
```

Loading the adapter from the Hub:

```python
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "<hf-user>/<repo-name>")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
```

## Repository structure

```
configs/     training configs (fine_tune.yaml, overfit_test.yaml)
scripts/
  utils.py             data loading, prompt building, normalization, scoring
  explore_data.py      dataset statistics
  explore_model.py     model inspection
  dataset_baseline.py  majority-answer baseline
  fine_tune.py         LoRA training loop
  evaluate.py          batched evaluation, writes metrics JSON and per-question CSV
results/     metrics JSON per run, training config
dw_assets.py downloads dataset and models at pinned revisions
```

## References

- Lobry, S., Marcos, D., Murray, J., Tuia, D. RSVQA: Visual Question Answering for Remote Sensing Data. IEEE TGRS, 2020.
- Chappuis, C., Walt, E., Mendez, V., Lobry, S., Le Saux, B., Tuia, D. The curse of language biases in remote sensing VQA: the role of spatial attributes, language diversity, and the need for clear evaluation. arXiv:2311.16782, 2023.
- Hu, Y. et al. RSGPT: A Remote Sensing Vision Language Model and Benchmark. arXiv:2307.15266, 2023.
- Kuckreja, K. et al. GeoChat: Grounded Large Vision-Language Model for Remote Sensing. CVPR, 2024.
- Hu, E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.
- Qwen Team. Qwen3-VL. Model card: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

## License

<!-- TODO: choose a code license -->
Code: TBD. RSVQA-LR is CC-BY-4.0. Qwen3-VL weights and the adapter follow the base model's license (see its model card).