from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from evaluate import load  
import pandas as pd
from transformers import GPT2TokenizerFast

def run(input_path, config):
    # Device setup for Mac (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        pipeline_device = -1  # pipeline does not support MPS, so use -1 (CPU) but move model to MPS
    else:
        device = torch.device("cpu")
        pipeline_device = -1

    # Load test data from config
    test_data_path = input_path
    with open(test_data_path) as f:
        test_data = json.load(f)
    
    results = []
    references = []
    finetuned_predictions = []
    base_predictions = []
    
    # Prepare raw_data from test_data
    raw_data = [{"requirement": item["text"].split("\n")[0].replace("For the following requirement, generate Gherkin scenarios: ", ""),
                "generated_gherkin_scenarios": "\n".join(item["text"].split("\n")[1:]).strip()}
               for item in test_data]

    # Load tokenizer and model with adapter
    finetuned_model_path = config["phi"]["finetuned_ia3_model_dir"]
    
    # Load the base model first
    base_model_path = config["phi"]["base_model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load model with PEFT
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Load the adapter
    finetuned_model = PeftModel.from_pretrained(
        base_model,
        finetuned_model_path,
        is_trainable=False
    )
    finetuned_model = finetuned_model.to(device)
    finetuned_tokenizer = tokenizer
    # Create pipeline without device argument when using device_map
    pipe_kwargs = {
        "model": finetuned_model,
        "tokenizer": finetuned_tokenizer,
        "max_new_tokens": 300,
        "do_sample": False
    }
    
    # Only add device if not using device_map
    if not hasattr(finetuned_model, 'hf_device_map'):
        pipe_kwargs["device"] = pipeline_device
    
    finetuned_pipe = pipeline("text-generation", **pipe_kwargs)

    # Load base model and tokenizer ONCE
    base_model_path = config["phi"]["base_model_dir"]
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True).to(device)
    base_pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=base_tokenizer,
        device=pipeline_device,
        max_new_tokens=300,
        do_sample=False
    )

    for item in raw_data:
        requirement = item["requirement"]
        reference = item["generated_gherkin_scenarios"]
        formatted_prompt = f"For the following requirement, generate Gherkin scenarios: {requirement}\n"

        # Inference with fine-tuned model
        finetuned_outputs = finetuned_pipe(formatted_prompt)
        finetuned_gherkin = finetuned_outputs[0]["generated_text"].replace(formatted_prompt, "").strip()

        # Inference with base model
        base_outputs = base_pipe(formatted_prompt)
        base_gherkin = base_outputs[0]["generated_text"].replace(formatted_prompt, "").strip()

        results.append({
            "requirement": requirement,
            "reference": reference,
            "finetuned_gherkin": finetuned_gherkin,
            "base_gherkin": base_gherkin
        })

        references.append(reference)
        finetuned_predictions.append(finetuned_gherkin)
        base_predictions.append(base_gherkin)

    # Calculate BLEU and ROUGE for fine-tuned model
    bleu = load("bleu")
    rouge = load("rouge")
    bertscore = load("bertscore")  # <- NEW

    bleu_finetuned = bleu.compute(predictions=finetuned_predictions, references=references)
    rouge_finetuned = rouge.compute(predictions=finetuned_predictions, references=references)
    bertscore_finetuned = bertscore.compute(
        predictions=finetuned_predictions,
        references=references,
        lang="en"
    )

    # Calculate BLEU, ROUGE, and BERTScore for base model
    bleu_base = bleu.compute(predictions=base_predictions, references=references)
    rouge_base = rouge.compute(predictions=base_predictions, references=references)
    bertscore_base = bertscore.compute(
        predictions=base_predictions,
        references=references,
        lang="en"
    )

    def format_bertscore(score_dict):
        return {
            'precision': sum(score_dict['precision']) / len(score_dict['precision']),
            'recall': sum(score_dict['recall']) / len(score_dict['recall']),
            'f1': sum(score_dict['f1']) / len(score_dict['f1'])
        }
    
    bertscore_finetuned = format_bertscore(bertscore_finetuned)
    bertscore_base = format_bertscore(bertscore_base)

    # Create a DataFrame and add metrics to it
    metrics_data = []
    metrics_data.append({
        'model': 'fine-tuned',
        'bleu': bleu_finetuned['bleu'],
        'rouge1': rouge_finetuned['rouge1'],
        'rouge2': rouge_finetuned['rouge2'],
        'rougeL': rouge_finetuned['rougeL'],
        'rougeLsum': rouge_finetuned['rougeLsum'],
        'bertscore_precision': bertscore_finetuned['precision'],
        'bertscore_recall': bertscore_finetuned['recall'],
        'bertscore_f1': bertscore_finetuned['f1']
    })
    
    metrics_data.append({
        'model': 'base',
        'bleu': bleu_base['bleu'],
        'rouge1': rouge_base['rouge1'],
        'rouge2': rouge_base['rouge2'],
        'rougeL': rouge_base['rougeL'],
        'rougeLsum': rouge_base['rougeLsum'],
        'bertscore_precision': bertscore_base['precision'],
        'bertscore_recall': bertscore_base['recall'],
        'bertscore_f1': bertscore_base['f1']
    })
    
    metrics_df = pd.DataFrame(metrics_data)
    output_csv_path = "inference_results_ia3_phi.csv"
    metrics_df.to_csv(output_csv_path, index=False)
    
    print(f"\nMetrics saved to: {output_csv_path}")
    
    
    return output_csv_path

