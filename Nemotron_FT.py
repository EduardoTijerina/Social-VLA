

"""
## Social Fine tunning for `Nemotron nano 3`/////////////////

* Base model: `Nemotron nano 3`
* Data-format: `RLHF`
* Fine tunning methods: Supervised fine tunning `STF` and Direct preference optimization `DPO`
* Quantization method: Activation aware weight quantization `AWQ`
* Inference engine: `vLLM`
* Running hardware: Nvidia DGX Spark
* Deployment hardware: Jetson Orin


=============== ENVIRONMENT SNAPSHOT ================
Timestamp: 2026-02-10
User: Francisco Eduardo Tijerina Jr.
Project: Robotics LLM Fine-Tuning (Mamba/Unsloth)

[FRAMEWORK]
PyTorch: 2.11.0.dev20260105+cu128
Mamba-SSM: 2.3.0

[ACCELERATOR]
Detected GPU: NVIDIA GB100
CUDA Version: 12.8
Driver Version: Latest (JetPack/Blackwell Compatible)
======================================================
 """




import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

import os
# Force disable just in case, though uninstalling is the real fix
os.environ["UNSLOTH_RETURN_LOGITS"] = "1" 
os.environ["UNSLOTH_DISABLE_VLLM"] = "1"
os.environ["UNSLOTH_USE_HUGGINGFACE_MODELS"] = "1"  # Fallback if needed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import gc
import shutil
from pathlib import Path
from trl import SFTTrainer, DPOTrainer, DPOConfig, SFTConfig
from transformers import TrainingArguments, AutoModelForCausalLM, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported, PatchDPOTrainer, FastLanguageModel
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
import subprocess
import matplotlib.pyplot as plt
from transformers import TrainerCallback
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


""" ///////////////////////////////////SUPERVISED FINE-TUNING (SFT)////////////////////////  """

model_name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
max_seq_length = 2048
sft_output_dir = "outputs_sft"
dpo_output_dir = "outputs_dpo"
final_model_dir = "nemotron_social_robot_finetuned"

print("=" * 70)
print("PHASE 1: Supervised Fine-Tuning (SFT)")
print("Teaching the model socially appropriate responses")
print("=" * 70)


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length= max_seq_length,
    dtype= None,
    load_in_4bit = True,  #Enable for memory efficiency
    trust_remote_code = True, 
    device_map={"": torch.cuda.current_device()},
)
print(model.device)
print(f"Model loaded on: {next(model.parameters()).device}")



## LoRA adapters//////////////////////////



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length= max_seq_length,
    dtype= None,
    load_in_4bit = True,  #Enable for memory efficiency
    trust_remote_code = True, 
    device_map={"": torch.cuda.current_device()},
)
print(model.device)
print(f"Model loaded on: {next(model.parameters()).device}")

# Load dataset
dataset = load_dataset("ProlificAI/social-reasoning-rlhf", split="train")

# Format for SFT: Train ONLY on chosen (appropriate) responses
def formatting_sft_func(examples):
    """
    Format for SFT: prompt + chosen response.
    Teaches the model to generate socially appropriate responses.
    """
    questions = examples["question"]
    chosen_responses = examples["chosen"]
    
    texts = []
    for question, chosen in zip(questions, chosen_responses):
        text = f"""Below is a social situation that requires thoughtful reasoning. Provide a response that demonstrates empathy, respect, and sound judgment.

### Situation:
{question}

### Response:
{chosen}{tokenizer.eos_token}"""
        texts.append(text)
    
    return {"text": texts}

print("\nFormatting dataset for SFT (using only 'chosen' responses)...")
sft_dataset = dataset.map(
    formatting_sft_func,
    batched=True,
    remove_columns=dataset.column_names
)

print(f"SFT dataset size: {len(sft_dataset)}")
print(f"\nSFT example preview:\n{sft_dataset[0]['text'][:300]}...")


## SFT Trainer ///////////////////////

from trl import SFTConfig
import inspect
import trl
print(f"TRL version: {trl.__version__}")

# See all valid parameters
sig = inspect.signature(SFTConfig.__init__)
print("Valid SFTConfig parameters:")
for param in sig.parameters:
    print(f"  - {param}")



class LossVisualizationCallback(TrainerCallback):
    """Callback to visualize training loss during and after training"""
    
    def __init__(self, output_dir="./"):
        self.losses = []
        self.steps = []
        self.output_dir = output_dir
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs is not None and 'loss' in logs:
            self.losses.append(logs['loss'])
            self.steps.append(state.global_step)
            
            # Update plot every 50 steps
            if len(self.steps) % 50 == 0:
                self.plot_loss(save_path=os.path.join(self.output_dir, 'training_loss_live.png'))
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        self.plot_loss(save_path=os.path.join(self.output_dir, 'training_loss_final.png'))
        
        loss_data = {
            'steps': self.steps,
            'losses': self.losses
        }
        with open(os.path.join(self.output_dir, 'loss_history.json'), 'w') as f:
            json.dump(loss_data, f, indent=2)
    
    def plot_loss(self, save_path='training_loss.png'):
        """Create and save the loss plot"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to: {save_path}")


from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq

# Initialize the callback
loss_callback = LossVisualizationCallback(output_dir=sft_output_dir)

sft_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=sft_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    
    args=SFTConfig(
        # Output directory (required as first positional or named arg)
        output_dir=sft_output_dir,
        
        # SFT-specific parameters
        dataset_text_field="text",
        max_seq_length=max_seq_length,  # This IS valid in TRL 0.24.0
        dataset_num_proc=2,
        packing=False,
        
        # Batch size optimization for DGX
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        
        # Learning rate schedule - use only ONE of warmup_steps or warmup_ratio
        warmup_ratio=0.05,  # Remove warmup_steps to avoid conflict
        num_train_epochs=3,
        learning_rate=2e-5,
        
        # Precision
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        
        # Logging and evaluation
        logging_steps=10,
        eval_strategy="no",
        
        # Optimization
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        
        # Regularization
        seed=3407,
        dataloader_num_workers=4,
        
        # Saving
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
    ),
    callbacks=[loss_callback],
)

# Train the model
sft_trainer.train()   

# Saving SFT Checkpoint///////////////////

print(f"\nSaving SFT checkpoint to {sft_output_dir}/final...")
model.save_pretrained(f"{sft_output_dir}/final")
tokenizer.save_pretrained(f"{sft_output_dir}/final")

print("\n" + "=" * 70)
print("SFT Phase Complete! Model has learned appropriate responses.")
print("=" * 70)


""" ////////////////////////////////////Direct preference Optimization (DPO) /////////////////////////////////////"""

print("\n" + "=" * 70)
print("PHASE 2: Direct Preference Optimization (DPO)")
print("Teaching the model to prefer appropriate over inappropriate responses")
print("=" * 70)



# Format for DPO: prompt, chosen, rejected
def formatting_dpo_func(examples):
    """
    Format for DPO: Separate prompt, chosen, and rejected.
    Teaches the model to prefer appropriate over inappropriate responses.
    """
    questions = examples["question"]
    chosen_responses = examples["chosen"]
    rejected_responses = examples["rejected"]
    
    prompts = []
    chosen_completions = []
    rejected_completions = []
    
    for question, chosen, rejected in zip(questions, chosen_responses, rejected_responses):
        prompt = f"""Below is a social situation that requires thoughtful reasoning. Provide a response that demonstrates empathy, respect, and sound judgment.

### Situation:
{question}

### Response:
"""
        prompts.append(prompt)
        chosen_completions.append(chosen + tokenizer.eos_token)
        rejected_completions.append(rejected + tokenizer.eos_token)
    
    return {
        "prompt": prompts,
        "chosen": chosen_completions,
        "rejected": rejected_completions
    }

print("Formatting dataset for DPO (using chosen + rejected pairs)...")
dpo_dataset = dataset.map(
    formatting_dpo_func,
    batched=True,
    remove_columns=dataset.column_names
)

print(f"DPO dataset size: {len(dpo_dataset)}")


## DPO Trainer //////////////////////

PatchDPOTrainer()

dpo_trainer = DPOTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dpo_dataset,
    ref_model = None,
    
    args = DPOConfig(
        # DPO-specific parameters
        beta = 0.1,
        max_length = 1024,  # Reduced from 2048
        max_prompt_length = 512,  # Reduced from 1024
        loss_type = "sigmoid",
        
        # Batch size - conservative
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 16,  # Effective batch = 32
        
        # Learning rate
        learning_rate = 5e-6,
        warmup_ratio = 0.1,
        
        # Training duration
        num_train_epochs = 1,
        max_steps = -1,
        
        # Precision
        fp16 = False,
        bf16 = True,
        
        # Logging
        logging_steps = 10,
        eval_strategy = "no",
        
        # Optimization
        optim = "adamw_8bit",  # 8-bit Adam saves memory
        weight_decay = 0.0,
        lr_scheduler_type = "cosine",
        max_grad_norm = 0.3,
        
        # Regularization
        seed = 3407,
        gradient_checkpointing = True,
        
        # Dataloader
        dataloader_num_workers = 4,
        dataloader_pin_memory = True,
        
        # Saving
        output_dir = dpo_output_dir,
        save_strategy = "steps",
        save_steps = 100,
        save_total_limit = 3,
        
        remove_unused_columns = False,
        report_to = "none",
    ),
)

print("\nStarting DPO training...")
dpo_trainer.train()

""" ////////////////////////////// Saving final Model ////////////////////// """


print(f"\nSaving LoRA adapter to {final_model_dir}...")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

# Merge LoRA weights into base model for quantization
print("\nMerging LoRA adapters into base model...")
model = model.merge_and_unload()

# Save as full precision (float16) for quantization
merged_model_dir = "nemotron_merged_fp16"
model.save_pretrained(
    merged_model_dir,
    safe_serialization=True,
    max_shard_size="5GB"
)
tokenizer.save_pretrained(merged_model_dir)
print(f"Merged model saved to: {merged_model_dir}")

print("\n" + "=" * 70)
print("TRAINING PIPELINE COMPLETE!")
print(f"LoRA adapter saved to: {final_model_dir}/")
print(f"Merged full model saved to: {merged_model_dir}/")
print("\nThe model is now optimized for social robot interactions:")
print("  \u2713 Phase 1 (SFT): Learned appropriate social responses")
print("  \u2713 Phase 2 (DPO): Learned to avoid inappropriate responses")
print("  \u2713 Merged model ready for INT4 quantization")
print("=" * 70)

