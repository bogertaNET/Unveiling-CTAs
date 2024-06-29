import json

from config import RAW_DATA_PATH
from tokenizer import Tokenizer

with open(RAW_DATA_PATH) as f:
    data = json.load(f)

print(f"Threat Actor Count: {len(data.keys())}")

# You do not need this step as we have already sharing the preprocessed data
preprocessor = Tokenizer(data)


