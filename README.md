(1)To run OpenAI model for labelling the gherkin scenarios, please save the open API key in .env folder, against OPENAI_API_KEY

rough
combined reults
python utils/visualize_inference_results.py


comapre training metrics
python3 utils/compare_training_metrics.py \ 
    --base-dir training_metrics \
    --models deepseek phi starcoder tinyllama \
    --output-dir comparison_plots

running fine tuning
python3 main.py --model tinyllama --mode finetune_1000 --input /Applications/GherkinScenarioGeneration/data/input/latest_finance_with_chain_of_thought_gherkin.json --config config/config.yaml   

running inference
python3 main.py --model deepseek --mode inference_100 --input /Applications/GherkinScenarioGeneration/data/testdata/test_data_100.json --config config/config.yaml



python3 utils/compare_training_metrics.py 
    --base-dir training_metrics 
    --models deepseek phi starcoder tinyllama 
    --output-dir comparison_plots