import os
import math
import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "QWEN_VL_MODEL",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)

QUESTION = "What is shown in this image? Answer in one short sentence."

IMAGE_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/"
    "Qwen-VL/assets/demo.jpeg"
)

MAX_NEW_TOKENS = 16


# ============================================================
# HELPER
# ============================================================

def gb(x):
    return round(x / 1024**3, 2)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PROJECT 3 - QWEN2.5-VL CONFIDENCE / ENTROPY TEST")
print("=" * 70)

print("\nMODEL:", MODEL_NAME)

# ------------------------------------------------------------
# 1. Load processor
# ------------------------------------------------------------

print("\n[1] Loading processor...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)


# ------------------------------------------------------------
# 2. Load frozen VLM
# ------------------------------------------------------------

print("[2] Loading frozen Qwen VLM...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

model.eval()

print("\nModel loaded successfully.")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM allocated after load:", gb(torch.cuda.memory_allocated()), "GB")
    print("VRAM reserved after load:", gb(torch.cuda.memory_reserved()), "GB")


# ------------------------------------------------------------
# 3. Construct multimodal prompt
# ------------------------------------------------------------

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": IMAGE_URL,
            },
            {
                "type": "text",
                "text": QUESTION,
            },
        ],
    }
]

print("\n[3] Preparing image + question...")

prompt_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[prompt_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

prompt_length = inputs.input_ids.shape[1]

print("Prompt token count:", prompt_length)


# ------------------------------------------------------------
# 4. Run generation WITH token scores
# ------------------------------------------------------------

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

print("\n[4] Running inference and collecting token logits...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,

        # IMPORTANT:
        # Return logits/scores for every generated token.
        return_dict_in_generate=True,
        output_scores=True,
    )


# ------------------------------------------------------------
# 5. Extract generated tokens
# ------------------------------------------------------------

generated_sequence = outputs.sequences[0]

generated_token_ids = generated_sequence[prompt_length:]

answer = processor.decode(
    generated_token_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\n" + "=" * 70)
print("MODEL ANSWER")
print("=" * 70)
print(answer)


# ------------------------------------------------------------
# 6. Compute probability, log-probability and entropy
# ------------------------------------------------------------

token_rows = []

print("\n" + "=" * 70)
print("TOKEN-LEVEL CONFIDENCE")
print("=" * 70)

for step, score_tensor in enumerate(outputs.scores):

    # score_tensor shape:
    # [batch_size, vocabulary_size]
    logits = score_tensor[0].float()

    # Convert logits -> log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)

    # Convert log probabilities -> probabilities
    probs = torch.exp(log_probs)

    token_id = generated_token_ids[step].item()

    token_log_prob = log_probs[token_id].item()
    token_prob = probs[token_id].item()

    # Entropy:
    # H = - sum_i p_i log(p_i)
    entropy = -(probs * log_probs).sum().item()

    token_text = processor.tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
    )

    token_rows.append(
        {
            "step": step + 1,
            "token_id": token_id,
            "token": token_text,
            "probability": token_prob,
            "log_probability": token_log_prob,
            "entropy": entropy,
        }
    )

    print(
        f"Step {step + 1:02d} | "
        f"token={token_text!r:<15} | "
        f"P={token_prob:.6f} | "
        f"logP={token_log_prob:.6f} | "
        f"H={entropy:.4f}"
    )


# ------------------------------------------------------------
# 7. Sequence-level statistics
# ------------------------------------------------------------

if token_rows:

    avg_log_prob = sum(
        row["log_probability"]
        for row in token_rows
    ) / len(token_rows)

    avg_entropy = sum(
        row["entropy"]
        for row in token_rows
    ) / len(token_rows)

    # Convert average log probability to an easier 0-1 quantity.
    geometric_mean_probability = math.exp(avg_log_prob)

    min_token_probability = min(
        row["probability"]
        for row in token_rows
    )

    print("\n" + "=" * 70)
    print("SEQUENCE-LEVEL RELIABILITY SIGNALS")
    print("=" * 70)

    print("Generated token count:", len(token_rows))

    print(
        "Average token log-probability (c_seq):",
        round(avg_log_prob, 6),
    )

    print(
        "Geometric mean token probability:",
        round(geometric_mean_probability, 6),
    )

    print(
        "Average predictive entropy:",
        round(avg_entropy, 6),
    )

    print(
        "Minimum generated-token probability:",
        round(min_token_probability, 6),
    )


# ------------------------------------------------------------
# 8. GPU memory
# ------------------------------------------------------------

if torch.cuda.is_available():

    print("\n" + "=" * 70)
    print("GPU MEMORY")
    print("=" * 70)

    print(
        "Current allocated:",
        gb(torch.cuda.memory_allocated()),
        "GB",
    )

    print(
        "Peak allocated:",
        gb(torch.cuda.max_memory_allocated()),
        "GB",
    )


print("\n" + "=" * 70)
print("P1 TEST COMPLETE")
print("=" * 70)
