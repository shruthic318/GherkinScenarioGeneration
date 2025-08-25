import argparse
import yaml
import os
from finetuned_models import (
     finetune_100_phi, finetune_400_phi, finetune_ia3_phi, finetune_dora_phi, finetune_dora_tinyllama, finetune_dora_starcoder, finetune_dora_deepseek,finetune_400_tinyllama, finetune_400_starcoder, finetune_400_deepseek,
    finetune_100_tinyllama, finetune_100_starcoder, finetune_100_deepseek,finetune_ia3_deepseek,
    finetune_ia3_tinyllama,finetune_ia3_starcoder,finetune_1000_deepseek,finetune_1000_phi,finetune_1000_starcoder,finetune_1000_tinyllama
)
from inference.inference_forMetrics import (
     inference_100_phi, inference_400_phi, inference_ia3_phi, inference_phi, inference_tinyllama, inference_starcoder, inference_deepseek,inference_400_tinyllama, inference_400_starcoder, inference_400_deepseek,
    inference_100_tinyllama, inference_100_starcoder, inference_100_deepseek,inference_ia3_deepseek,inference_ia3_starcoder,inference_ia3_tinyllama,
    inference_1000_deepseek,inference_1000_phi,inference_1000_starcoder,inference_1000_tinyllama
)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune or run inference on models.")
    parser.add_argument("--model", choices=["phi", "tinyllama","starcoder","deepseek"], required=True)
    parser.add_argument("--mode", choices=["finetune","finetune_100","finetune_400","finetune_1000","finetune_ia3lora","inference_ia3lora","inference_1000", "inference_100","inference_400","inference"], required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON")
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    args = parser.parse_args()

    #config = load_config(os.path.join("config", "config.yaml"))
    config = load_config(args.config)

    if args.model == "phi":
        if args.mode == "finetune":
            finetune_dora_phi.run(args.input, config)
        elif args.mode == "finetune_100":
            finetune_100_phi.run(args.input, config)
        elif args.mode == "inference_100":
            inference_100_phi.run(args.input, config)
        elif args.mode == "finetune_400":
            finetune_400_phi.run(args.input, config)
        elif args.mode == "inference_400":
            inference_400_phi.run(args.input, config)
        elif args.mode == "finetune_ia3lora":
            finetune_ia3_phi.run(args.input, config)
        elif args.mode == "inference_ia3lora":
            inference_ia3_phi.run(args.input, config)
        elif args.mode == "finetune_1000":
            finetune_1000_phi.run(args.input, config)
        elif args.mode == "inference_1000":
            inference_1000_phi.run(args.input, config)
        else:
            inference_phi.run(args.input, config)
    elif args.model == "tinyllama":
        if args.mode == "finetune":
            finetune_dora_tinyllama.run(args.input, config)
        elif args.mode == "finetune_100":
            finetune_100_tinyllama.run(args.input, config)
        elif args.mode == "inference_100":
            inference_100_tinyllama.run(args.input, config)
        elif args.mode == "finetune_400":
            finetune_400_tinyllama.run(args.input, config)
        elif args.mode == "inference_400":
            inference_400_tinyllama.run(args.input, config)
        elif args.mode == "finetune_ia3lora":
            finetune_ia3_tinyllama.run(args.input, config)
        elif args.mode == "inference_ia3lora":
            inference_ia3_tinyllama.run(args.input, config)
        elif args.mode == "finetune_1000":
            finetune_1000_tinyllama.run(args.input, config)
        elif args.mode == "inference_1000":
            inference_1000_tinyllama.run(args.input, config)
        else:
            inference_tinyllama.run(args.input, config)
    elif args.model == "starcoder":
        if args.mode == "finetune":
            finetune_dora_starcoder.run(args.input, config)
        elif args.mode == "finetune_100":
            finetune_100_starcoder.run(args.input, config)    
        elif args.mode == "inference_100":
            inference_100_starcoder.run(args.input, config)
        elif args.mode == "finetune_400":
            finetune_400_starcoder.run(args.input, config)    
        elif args.mode == "inference_400":
            inference_400_starcoder.run(args.input, config)
        elif args.mode == "finetune_ia3lora":
            finetune_ia3_starcoder.run(args.input, config)    
        elif args.mode == "inference_ia3lora":
            inference_ia3_starcoder.run(args.input, config)
        elif args.mode == "finetune_1000":
            finetune_1000_starcoder.run(args.input, config)    
        elif args.mode == "inference_1000":
            inference_1000_starcoder.run(args.input, config)
            
        else:
            inference_starcoder.run(args.input, config)
    elif args.model == "deepseek":
        if args.mode == "finetune":
            finetune_dora_deepseek.run(args.input, config)
        elif args.mode == "finetune_100":
            finetune_100_deepseek.run(args.input, config)
        elif args.mode == "inference_100":
            inference_100_deepseek.run(args.input, config)
        elif args.mode == "finetune_400":
            finetune_400_deepseek.run(args.input, config)
        elif args.mode == "inference_400":
            inference_400_deepseek.run(args.input, config)
        elif args.mode == "finetune_ia3lora":
            finetune_ia3_deepseek.run(args.input, config)
        elif args.mode == "inference_ia3lora":
            inference_ia3_deepseek.run(args.input, config)
        elif args.mode == "finetune_1000":
            finetune_1000_deepseek.run(args.input, config)
        elif args.mode == "inference_1000":
            inference_1000_deepseek.run(args.input, config)
        else:
            inference_deepseek.run(args.input, config)

if __name__ == "__main__":
    main()