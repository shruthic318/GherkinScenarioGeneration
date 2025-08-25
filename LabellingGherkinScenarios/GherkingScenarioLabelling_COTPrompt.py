import json
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv


load_dotenv()
# Initialize OpenAI client with API key
api_key=os.getenv("OPENAI_API_KEY")
client=OpenAI(api_key=api_key)

# Load the requirements file
unlabelled_data_path=os.path.join("data","unlabelledData","finance_user_requirements.json")
with open(unlabelled_data_path, "r") as f:
    finance_data = json.load(f)

# Define the chain-of-thought prompt format
chainofthought_template = """
You are an expert QA engineer. Convert the requirements into positive and negative gherkin scenarios. 
In case , there are no negative scenarios, ensure that only positive scenarios are given to the user.

Instruction
Identify the actors , preconditions, actions and expected outcomes . In case of any numbers involved, keep the scenarios generic.
Ensure that there are no adjacent '\\n'
The Gherkin scenario should be constructed with the following consideration:
(i)Feature - Mandatory.Should give a brief description about the requirement being tested.
(ii) Scenario - Mandatory. Should give a brief description about the scenario from the requirement being verified.
(iii) Given - Should give the preconditions of the scenario
(iv) When - Should give the action , which would be performed in the scenario
(v) Then - Should give the expected outcome from the scenario
(vi) And - Will be used along in Given/When/Then condition as and when required
(vii) But - Will be used along in Given/When/Then condition as and when required

Ensure the following best practices are followed:
(i) The scenarios should be independent and atomic
(ii) Use Active voice for step description

Now convert this to :

Requirement: {requirement}
Gherkin Scenarios:
"""

# Generate Gherkin scenarios with Chain of Thought prompting
results = []
start_time = time.time()

checkpoint_path=os.path.join("data", "input", "checkpoint_output.json")


for i, item in enumerate(tqdm(finance_data, desc="Generating Gherkin Scenarios")):
    requirement = item["requirement"]
    chainofthought_prompt = chainofthought_template.format(requirement=requirement)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You generate Gherkin scenarios from user requirements."},
                {"role": "user", "content": chainofthought_prompt}
            ],
            timeout=30
        )

        generated_content = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"\n⚠️ Error on item {i+1}: {e}")
        generated_content = "ERROR: Failed to generate"

    results.append({
        "requirement": requirement,
        "generated_gherkin_scenarios": generated_content
    })

    # Save a checkpoint every 20 records
    if (i + 1) % 20 == 0:
        
        with open(checkpoint_path, "w") as cp:
            json.dump(results, cp, indent=2)
        print(f"\n💾 Checkpoint saved at item {i+1}")

# Save the final output

output_path = os.path.join("data", "input", "latest_finance_with_chain_of_thought_gherkin.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - start_time
print(f"Chain-of-thought Gherkin generation complete in {int(elapsed//60)} min {int(elapsed%60)} sec.")
print(f"📄 Output saved to {output_path}")
