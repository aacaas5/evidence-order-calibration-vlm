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

IMAGE_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/"
    "Qwen-VL/assets/demo.jpeg"
)

QUESTION = "What is shown in this image? Answer briefly."

MAX_NEW_TOKENS = 12


def gb(x):
    return round(x / 1024**3, 2)


print("=" * 72)
print("PROJECT 3 - P2 HIDDEN-STATE EXTRACTION TEST")
print("=" * 72)

print("\nModel:", MODEL_NAME)


# ============================================================
# 1. LOAD PROCESSOR
# ============================================================

print("\n[1] Loading processor...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)


# ============================================================
# 2. LOAD FROZEN VLM
# ============================================================

print("[2] Loading frozen Qwen VLM...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

model.eval()

print("Model loaded.")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Allocated after load:",
        gb(torch.cuda.memory_allocated()),
        "GB",
    )


# ============================================================
# 3. LOCATE FINAL LANGUAGE TRANSFORMER LAYER
# ============================================================

print("\n[3] Locating final language transformer layer...")

candidate_layers = []

for name, module in model.named_modules():
    parts = name.split(".")

    if len(parts) >= 2 and parts[-2] == "layers":
        try:
            layer_index = int(parts[-1])
            candidate_layers.append(
                (layer_index, name, module)
            )
        except ValueError:
            pass

if not candidate_layers:
    raise RuntimeError(
        "Could not automatically locate Qwen language layers."
    )

candidate_layers.sort(
    key=lambda item: item[0]
)

final_layer_index, final_layer_name, final_layer = candidate_layers[-1]

print("Final layer index:", final_layer_index)
print("Final layer module:", final_layer_name)


# ============================================================
# 4. REGISTER MEMORY-SAFE FORWARD HOOK
# ============================================================

captured_states = []


def capture_final_state(module, inputs, output):
    """
    Capture ONLY the final sequence position produced by
    the final transformer layer.

    We immediately detach and copy it to CPU.
    """

    if isinstance(output, tuple):
        hidden = output[0]
    else:
        hidden = output

    if not torch.is_tensor(hidden):
        return

    # hidden shape is usually:
    # [batch, sequence_length, hidden_dimension]
    if hidden.ndim != 3:
        return

    last_token_state = (
        hidden[0, -1, :]
        .detach()
        .float()
        .cpu()
    )

    captured_states.append(last_token_state)


hook_handle = final_layer.register_forward_hook(
    capture_final_state
)

print("Forward hook registered.")


# ============================================================
# 5. PREPARE IMAGE + QUESTION
# ============================================================

print("\n[4] Preparing image + question...")

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


# ============================================================
# 6. GENERATE ANSWER + SCORES
# ============================================================

print("\n[5] Running generation...")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

# No longer need hook after generation.
hook_handle.remove()


# ============================================================
# 7. DECODE ANSWER
# ============================================================

sequence = outputs.sequences[0]

generated_token_ids = sequence[prompt_length:]

answer = processor.decode(
    generated_token_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\n" + "=" * 72)
print("MODEL ANSWER")
print("=" * 72)
print(answer)


# ============================================================
# 8. CONFIDENCE + ENTROPY
# ============================================================

valid_rows = []

special_ids = set(
    processor.tokenizer.all_special_ids
)

for step, score_tensor in enumerate(outputs.scores):

    if step >= len(generated_token_ids):
        break

    token_id = generated_token_ids[step].item()

    # Ignore control tokens such as <|im_end|>
    if token_id in special_ids:
        continue

    logits = score_tensor[0].float()

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    probs = torch.exp(log_probs)

    token_log_prob = log_probs[token_id].item()
    token_probability = probs[token_id].item()

    entropy = -(
        probs * log_probs
    ).sum().item()

    token_text = processor.tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
    )

    valid_rows.append(
        {
            "token": token_text,
            "probability": token_probability,
            "log_probability": token_log_prob,
            "entropy": entropy,
        }
    )


print("\n" + "=" * 72)
print("NATIVE RELIABILITY SIGNALS")
print("=" * 72)

if valid_rows:

    avg_log_prob = sum(
        r["log_probability"]
        for r in valid_rows
    ) / len(valid_rows)

    avg_entropy = sum(
        r["entropy"]
        for r in valid_rows
    ) / len(valid_rows)

    geometric_mean_probability = math.exp(
        avg_log_prob
    )

    print(
        "Semantic generated tokens:",
        len(valid_rows),
    )

    print(
        "c_seq:",
        round(avg_log_prob, 6),
    )

    print(
        "Geometric mean probability:",
        round(geometric_mean_probability, 6),
    )

    print(
        "Average entropy:",
        round(avg_entropy, 6),
    )

else:
    raise RuntimeError(
        "No semantic answer tokens were generated."
    )


# ============================================================
# 9. PROCESS CAPTURED HIDDEN STATES
# ============================================================

print("\n" + "=" * 72)
print("HIDDEN REPRESENTATION")
print("=" * 72)

print(
    "Captured forward-pass states:",
    len(captured_states),
)

if not captured_states:
    raise RuntimeError(
        "No hidden states were captured."
    )

print(
    "Hidden dimension:",
    captured_states[0].numel(),
)


# ------------------------------------------------------------
# During autoregressive generation:
#
# First forward pass processes the complete prompt and predicts
# the FIRST answer token.
#
# Later forward passes predict later answer tokens.
#
# Therefore the first captured vector corresponds to the
# representation immediately before the first generated token.
# ------------------------------------------------------------

first_answer_hidden = captured_states[0]

print(
    "First-answer hidden vector shape:",
    tuple(first_answer_hidden.shape),
)

print(
    "First 10 hidden values:",
    [
        round(float(v), 6)
        for v in first_answer_hidden[:10]
    ],
)

print(
    "Hidden-vector L2 norm:",
    round(
        torch.linalg.vector_norm(
            first_answer_hidden
        ).item(),
        6,
    ),
)


# ============================================================
# 10. BUILD OUR FUTURE RELIABILITY FEATURE VECTOR
# ============================================================

scalar_features = torch.tensor(
    [
        avg_log_prob,
        avg_entropy,
    ],
    dtype=torch.float32,
)

research_feature_vector = torch.cat(
    [
        first_answer_hidden,
        scalar_features,
    ]
)

print("\n" + "=" * 72)
print("RESEARCH FEATURE VECTOR x")
print("=" * 72)

print(
    "Hidden features:",
    first_answer_hidden.numel(),
)

print(
    "Scalar features:",
    scalar_features.numel(),
)

print(
    "Total x dimension:",
    research_feature_vector.numel(),
)

print(
    "\nx = [hidden_state ; c_seq ; entropy]"
)


# ============================================================
# 11. GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    print("\n" + "=" * 72)
    print("GPU MEMORY")
    print("=" * 72)

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


print("\n" + "=" * 72)
print("P2 TEST COMPLETE")
print("=" * 72)
