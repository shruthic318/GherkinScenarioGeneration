import os
import shutil
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from sklearn.model_selection import train_test_split 
import gc
from pathlib import Path
from utils.training_visualizer import TrainingMetricsLogger

def run(input_path, config):
    # Load requirements and gherkin scenarios(labelled data)
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    formatted_data = [{
        "text": f"For the following requirement, generate Gherkin scenarios: {item['requirement']}\n{item['generated_gherkin_scenarios']}"
    } for item in raw_data]

    train_data, test_data = train_test_split(formatted_data, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_list(train_data)

    #Save test data for inference
    test_data_path = config["inference"]["test_data_100"]
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
    with open(test_data_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"Test data saved to: {test_data_path}")

    # Set MPS device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Output directories from config
    output_dir = config["phi"]["finetuned_100_model_dir"]
    logging_dir = config["phi"]["training"]["logging_dir"]
    metrics_dir = config["phi"]["training"]["metrics_100_dir"]
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
  
    # Load model and tokenizer
    base_model = config["phi"]["base_model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    model = model.to(device)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Initialize metrics logger
    metrics_logger = TrainingMetricsLogger(output_dir=metrics_dir)

    # Training configuration
    training_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        optim="adamw_torch",
        save_steps=25,
        logging_steps=1, 
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        save_total_limit=2,
        logging_dir=logging_dir,
        report_to=[],  
        log_level="info",
        logging_first_step=True,  
    )

    # Fine-tuning trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_params,
        args=training_params,
        callbacks=[metrics_logger]  
    )

    # Clean memory
    gc.collect()
    torch.mps.empty_cache()

    # Start training
    trainer.train()
    
    # Save LoRA adapter weights
    trainer.model.save_pretrained(output_dir)

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Generate and save metrics plots
    metrics_logger.plot_metrics()
    
    # Ensure all tokenizer files are present
    from huggingface_hub import snapshot_download
    base_model_dir = snapshot_download(base_model, allow_patterns=[
        "tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"
    ])
    required_files = [
        "tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    for fname in required_files:
        src = os.path.join(base_model_dir, fname)
        dst = os.path.join(output_dir, fname)
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy(src, dst)
            
    return output_dir