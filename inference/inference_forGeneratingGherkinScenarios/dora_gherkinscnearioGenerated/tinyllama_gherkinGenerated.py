from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import yaml
import os
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load the fine-tuned model 
model_name = config["tinyllama"]["finetuned_model_dir"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Single requirement for demo
requirement = "The application should fetch live market prices with a latency of less than 1 second."

# Create the prompt
prompt = f"For the following requirement, generate Gherkin scenarios: {requirement}\n"

# Run inference
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=300)
result = generator(prompt)[0]['generated_text']
#Create directory 
output_dir = "generated_gherkin"
os.makedirs(output_dir, exist_ok=True)
# Save to file
output_file = "generated_gherkin/dora_tinyllama.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result)