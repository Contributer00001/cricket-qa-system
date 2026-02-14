import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


# ---------------------------------------------------------
# IMPORTANT:
# Your dataset already contains fully formatted instruction text
# under the key: "text"
# ---------------------------------------------------------
def format_example(example):
    return example["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="data/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/lora-qwen-ipl")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=8)
    args = parser.parse_args()

    print(">>> Args:", vars(args))

    # ---------------------------------------------------------
    # Device (Apple Silicon support)
    # ---------------------------------------------------------
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    model.to(device)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    dataset = load_dataset(
        "json",
        data_files=args.dataset,
        split="train",
    )

    print(f"Loaded {len(dataset)} training samples")

    # ---------------------------------------------------------
    # LoRA Configuration
    # ---------------------------------------------------------
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        print("✅ LoRA enabled")

    # ---------------------------------------------------------
    # Training Arguments
    # ---------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=2e-4,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to="none",
        optim="adamw_torch",
    )

    # ---------------------------------------------------------
    # SFT Trainer (CORRECT USAGE)
    # ---------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=format_example,
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    trainer.train()

    # ---------------------------------------------------------
    # Save LoRA adapter + tokenizer
    # ---------------------------------------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✅ Training complete. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
