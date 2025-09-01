# Steps to set the up the project.

# Prerequisites
python3.8+
git to clone the repository

# Install all the required libraries, by running the following command in terminal
pip install -r requirements.txt

# Create a .env file 
with key as "OPENAI_API_KEY" and set the openai api key against it (so that labelling can be performed, using openAI)

# Set the hugging face token in config/config/yaml , against the
hugging face:
 token:

# to run the finetune model script /inference script, the following command needs to be run
python3 main.py --model [model_name] --mode [mode] --input [input_path] --config config/config.yaml

Possible values for model:
tinyllama
phi
deepseek
starcoder

 Possible values for mode: 
 finetune |
 finetune_100 |
 finetune_400| 
 finetune_1000| 
 inference|
 inference_100 |
 inference_400 |
 inference_1000 |
 inference_ia3

# sample command to run finetune deepseek model with 100 data samples. 
python3 main.py --model tinyllama --mode finetune_1000 --input /data/input/100_finance_with_chain_of_thought_gherkin.json --config config/config.yaml  

# sample command to run finetune tinyllama model with ia3 fine tuning 
python3 main.py --model tinyllama --mode finetune_ia3lora --input /data/input/latest_finance_with_chain_of_thought_gherkin.json --config config/config.yaml  

# sample command to run inference for starcoder model fientuned with 100 data samples. 
python3 main.py --model starcoder --mode inference_100 --input /data/testdata/test_data_100.json --config config/config.yaml  

# sample command to run finetune phi model with ia3 fine tuning 
python3 main.py --model phi --mode inference_ia3lora --input /data/testdata/test_data.json --config config/config.yaml

# to generate gherkin scenarios for sample requirement, run the files in the folder: inference/inference_forGeneratingGherkinScenarios

# to generate training loss comparison graphs
python3 utils/compare_training_metrics.py \ 
    --base-dir training_metrics \
    --models deepseek phi starcoder tinyllama \
    --output-dir comparison_plots
 
# to generate inference results graphs
python3 utils/visualize_inference_results.py 

GherkinScenarioGeneration/
├── config/                          # Model configurations and hyperparameters
├── data/                           # Raw dataset for training (100,400a dn 1000) |Test datasets (100, 400, 1000 samples) |unlabelled data
├── finetuned_models/               # Fine-tuning scripts for models
├── inference/                      # Inference scripts and inference scenario generation scripts
├── saved_models/                   # Fine-tuned models
├── training_metrics/               # Training performance logs and metrics
├── generated_gherkin/              # Generated Gherkin scenario outputs from inference
├── comparison_plots/               # Training and inference visualizations
├── logs/                          # Training logs by model
├── utils/                         # Script to generate training and inference comparison graphs
├── cohenKappa/                    # Labelled scenarios analysed for interannotator agreement
└── LabellingGherkinScenarios/     # Scenario labeling tool
└── main.py                        # to invoke the finetune /inference script
└── README.md                      # Project Documentation
└── requirements.txt               # Required python libraries




