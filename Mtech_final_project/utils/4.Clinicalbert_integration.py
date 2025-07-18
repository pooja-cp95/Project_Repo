from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch
import pandas as pd

# Load preprocessed data
df = pd.read_csv('preprocessed_notes.csv')

# Load ClinicalBERT model for NER
def load_clinicalbert_model():
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    # Create NER pipeline
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    return ner_pipeline

# Extract medical entities using ClinicalBERT
def extract_entities(text, ner_pipeline):
    entities = ner_pipeline(text)
    # Filter for relevant entity types (symptoms, diseases, etc.)
    medical_entities = [
        ent for ent in entities 
        if ent['entity_group'] in ['SYMPTOM', 'DISEASE', 'BODY_PART']
    ]
    return medical_entities

# Main execution
ner_pipeline = load_clinicalbert_model()

# Apply to a sample of notes (for demo - would be computationally intensive for full dataset)
sample_notes = df['note_text'].head(3).tolist()
results = []
for note in sample_notes:
    entities = extract_entities(note, ner_pipeline)
    results.append({
        'note_text': note,
        'entities': entities
    })

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv('clinicalbert_ner_results.csv', index=False)

# Display sample results
print(results_df.head())