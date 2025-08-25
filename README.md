# Steps to set the up the project.

# Prerequisites
# python3.8+
# git to clone the repository

# create and activate virtual environment

# Install all the required libraries, by running the following command in terminal
pip install -r requirements.txt

# Set the Open API key in .env file against OPEN_AP_KEY(so that labelling can be performed, using openAI)
# Set the hugging face token in config/config/yaml , against the hugging face:
#                                                                   token:

# to run the finetune model script /inference script, the following command needs to be run
python3 main.py --model [model_name] --mode [mode] --input [input_path] --config config/config.yaml

# Possible values for model: phi, tinyllama, deepseek, starcoder
# Possible values for mode: 
# inference
# inference_100
# inference_400
# inference_1000
# inference_ia3

# sample command to run finetune deepseek model with 100 data samples. 
python3 main.py --model tinyllama --mode finetune_1000 --input /Applications/GherkinScenarioGeneration/data/input/100_finance_with_chain_of_thought_gherkin.json --config config/config.yaml  

# sample command to run finetune tinyllama model with ia3 fine tuning 
python3 main.py --model tinyllama --mode finetune_ia3 --input /Applications/GherkinScenarioGeneration/data/input/latest_finance_with_chain_of_thought_gherkin.json --config config/config.yaml  

# sample command to run inference for starcoder model fientuned with 100 data samples. 
python3 main.py --model starcoder --mode inference_100 --input /Applications/GherkinScenarioGeneration/data/testdata/test_data_100.json --config config/config.yaml  

# sample command to run finetune phi model with ia3 fine tuning 
python3 main.py --model phi --mode inference_ia3 --input /Applications/GherkinScenarioGeneration/data/testdata/test_data.json --config config/config.yaml

# to generate gherkin scenarios for sample requirement, run the files in the folder: inference/inference_forGeneratingGherkinScenarios

# to generate training loss comparison graphs
python3 utils/compare_training_metrics.py \ 
    --base-dir training_metrics \
    --models deepseek phi starcoder tinyllama \
    --output-dir comparison_plots
 
# to generate inference results graphs
python3 utils/visualize_inference_results.py 

