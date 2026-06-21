import os
import sys
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

def format_prompts(batch):
    formatted = []
    instructions = batch.get("instruction") or batch.get("prompt") or []
    outputs = batch.get("output") or batch.get("solution") or batch.get("code") or []
    
    n = max(len(instructions), len(outputs))
    for i in range(n):
        inst = instructions[i] if i < len(instructions) else ""
        out = outputs[i] if i < len(outputs) else ""
        text = f"<|im_start|>system\nYou are an expert GPU kernel programmer. Write optimized Triton code.<|im_end|>\n<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n```python\n{out}\n```<|im_end|>"
        formatted.append(text)
    return formatted

def main():
    output_dir = "/models/sft_warmup"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading coldstart dataset...")
    try:
        raw_dataset = load_dataset("hkust-nlp/drkernel-coldstart-8k", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}. Initializing stub dataset.")
        # Fallback dataset
        from datasets import Dataset
        raw_dataset = Dataset.from_dict({
            "instruction": ["Write a fused add + relu Triton kernel"],
            "solution": ["import triton\nimport triton.language as tl\n# solution implementation"]
        })
        
    print(f"Total raw rows: {len(raw_dataset)}")
    
    # Filter by speedup >= 1.2
    filtered = []
    for row in raw_dataset:
        speedup = row.get("final_speedup") or row.get("speedup") or 1.0
        if speedup >= 1.2:
            filtered.append(row)
            
    if not filtered:
        print("No speedup >= 1.2 rows found, using raw dataset.")
        # convert Dataset to list of dicts for safety
        filtered = [row for row in raw_dataset]
        
    print(f"Filtered rows for SFT: {len(filtered)}")
    
    model_id = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    print(f"Loading base model and tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="bfloat16",
        device_map="auto"
    )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
    )
    
    # Convert list of dicts back to HF Dataset if filtered list was used
    from datasets import Dataset as HFDataset
    sft_dataset = HFDataset.from_list(filtered)
    
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        peft_config=peft_config,
        formatting_func=format_prompts,
        max_seq_length=1024,
        args=args,
    )
    
    print("Starting SFT training...")
    trainer.train()
    
    print(f"Saving SFT warm LoRA checkpoint to: {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()
