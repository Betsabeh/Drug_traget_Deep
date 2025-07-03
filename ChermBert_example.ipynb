#import package
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

# For ChemBERTa
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MLM") # Or other ChemBERTa variants
model = AutoModel.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("model move to", device)

smiles_string = ["CCOc1c(Cl)cc(NC(=O)CCc2cn(C)c3cc(C)ccc23)cc1Cl", "CCC"]
print("smile lenght=",len(smiles_string))
inputs = tokenizer(smiles_string, return_tensors="pt", padding=True, truncation=True).to(model.device)
print("Input IDs (first SMILES example):")
print(inputs['input_ids'][0])
print("\nAttention Mask (first SMILES example):")
print(inputs['attention_mask'][0])
print(f"\nVocabulary size: {len(tokenizer.vocab)}")
print(f"Max input length for this model: {tokenizer.model_max_length}")

with torch.no_grad():
    outputs = model(**inputs)
    #print(outputs)

embeddings = outputs.last_hidden_state # Or pooler_output depending on your needs
print("embeding size=",embeddings.shape) # Should be (batch_size, sequence_length, hidden_size)
