import os

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_NAME = os.getenv(
    "QWEN_VL_MODEL",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)

print("=" * 60)
print("PROJECT  - QWEN2.5-VL HARDWARE VIABILITY TEST")
print("=" * 60)

print("\n[1] Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("[2] Loading frozen VLM...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

model.eval()

print("\n[3] Model loaded successfully")

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM allocated after load:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB",
    )
    print(
        "VRAM reserved after load:",
        round(torch.cuda.memory_reserved() / 1024**3, 2),
        "GB",
    )

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {
                "type": "text",
                "text": "What is shown in this image? Answer in one short sentence.",
            },
        ],
    }
]

print("\n[4] Preparing multimodal input...")

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

print("[5] Running frozen VLM inference...")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
    )

generated_ids_trimmed = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(
        inputs.input_ids,
        generated_ids
    )
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\n" + "=" * 60)
print("MODEL ANSWER")
print("=" * 60)

print(output_text[0])

if torch.cuda.is_available():
    print("\n" + "=" * 60)
    print("GPU MEMORY")
    print("=" * 60)

    print(
        "Current allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB",
    )

    print(
        "Peak allocated:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "GB",
    )

print("\nTest complete.")