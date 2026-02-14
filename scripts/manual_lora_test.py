import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------
# Paths
# -----------------------
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "outputs/lora-qwen-ipl"

# -----------------------
# Device
# -----------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# -----------------------
# Load tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# -----------------------
# Load base model
# -----------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float32
).to(device)

# -----------------------
# Load LoRA adapters
# -----------------------
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# -----------------------
# TEST PROMPT (IMPORTANT)
# -----------------------
prompt = """### Question:
How many runs were scored in the last 3 overs of the innings?

### Answer:"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# -----------------------
# Generate
# -----------------------
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

# -----------------------
# Print output
# -----------------------
print("\n=== MODEL OUTPUT ===")
print(tokenizer.decode(output[0], skip_special_tokens=True))
