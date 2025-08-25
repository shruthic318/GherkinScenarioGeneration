import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score
import os

# Load config.yaml
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Get the Excel file path from config and load it
survey_path = config["path"]["gherkin_scenario_survey"]
survey_path = os.path.abspath(survey_path)
df = pd.read_excel(survey_path)

# Extract annotator columns and map Yes/No  to numbers
annotator1 = df['Annotator 1(Shruthi Chandra Babu)']
annotator2 = df['Annotator 2(Kathija Afrose Sathar)']
mapping = {'Yes': 1, 'No': 0}
annotator1_num = annotator1.map(mapping)
annotator2_num = annotator2.map(mapping)

# Calculate Cohen's Kappa score
kappa = cohen_kappa_score(annotator1_num, annotator2_num)
print(f"Cohen's Kappa Score: {kappa:.3f}")
