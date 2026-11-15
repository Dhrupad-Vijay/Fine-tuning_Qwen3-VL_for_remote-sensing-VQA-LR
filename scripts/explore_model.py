import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoModelForImageTextToText, AutoProcessor
from utils import load_splits

PTH = Path(__file__).resolve().parent.parent
MODEL = PTH / "models/Qwen3-VL-4B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
qa_1 = load_splits("val")[0]

print("------------------------------------")
print("--------- Image input part ---------")
print("------------------------------------")

image = Image.open(qa_1['img']).convert('RGB')
sdf = processor.image_processor(images=[image], return_tensors="pt")
t, h, w = sdf['image_grid_thw'][0]
print(f"\nInput image size - {image.size}")
print(f"pixel_values: {tuple(sdf['pixel_values'].shape)}  (one row per patch, pixels flattened (1536 cause qwen pads 2nd frame (another 3 channel RGB) so 16*16*3*2))")
print(f"image_grid_thw: t={t}, h={h}, w={w} -> {t * h * w} patches\n")

print("-----------------------------------")
print("--------- Text input part ---------")
print("-----------------------------------")

message = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': qa_1['question']}]}]
input_text = processor.apply_chat_template(message, tokenizer=False, add_generation_prompt=True)
print('\n',input_text)

print("-----------------------------------------------")
print("--------- Combined Image & Text input ---------")
print("-----------------------------------------------")

inputs = processor(images=[image], text=[input_text], return_tensors="pt")
ids = inputs['input_ids'][0]
decoded = processor.decode(ids)
print(f"keys: {list(inputs)}")
print(f"The encoded text prompt is:\n {inputs['input_ids'][0]}")
print(f"The decoded input prompt is: \n{decoded}")
print(f"The shape of the input image: {inputs['pixel_values'].shape}")

print("--------------------------------------------------")
print("--------- One forward pass through model ---------")
print("--------------------------------------------------")

inputs = inputs.to(device)
with torch.no_grad():
    logits = model(**inputs, max_new_tokens=10).logits
probs = torch.softmax(logits[0,48], dim=-1).float().cpu().numpy()
print(f"It generated logits {logits.shape}")
plt.plot(probs)

out = model.generate(**inputs, max_new_tokens=16, do_sample=True)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, out)
]
print(processor.decode(generated_ids_trimmed)[0])

"""

------------------------------------
--------- Image input part ---------
------------------------------------

Input image size - (256, 256)
pixel_values: (256, 1536)  (one row per patch, pixels flattened (1536 cause qwen pads 2nd frame (another 3 channel RGB) so 16*16*3*2))
image_grid_thw: t=1, h=16, w=16 -> 256 patches

-----------------------------------
--------- Text input part ---------
-----------------------------------
[transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`

 <|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Is it a rural or an urban area<|im_end|>
<|im_start|>assistant

-----------------------------------------------
--------- Combined Image & Text input ---------
-----------------------------------------------
keys: ['input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values', 'image_grid_thw']
The encoded text prompt is:
 tensor([151644,    872,    198, 151652, 151655, 151655, 151655, 151655, 151655,
        151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
        151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
        151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
        151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
        151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
        151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
        151655, 151655, 151655, 151655, 151655, 151653,   3872,    432,    264,
         19082,    476,    458,  15662,   3082, 151645,    198, 151644,  77091,
           198] )
The decoded input prompt is: 
<|im_start|>user
<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>Is it a rural or an urban area<|im_end|>
<|im_start|>assistant

The shape of the input image: torch.Size([256, 1536])
--------------------------------------------------
--------- One forward pass through model ---------
--------------------------------------------------
It generated logits torch.Size([1, 82, 151936])
This is a **rural area**.

"""