import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

# Load data from previous stages
kg_results = pd.read_csv('notes_with_disease_predictions.csv')
bert_results = pd.read_csv('clinicalbert_ner_results.csv')

# Simulate ClinicalBERT disease predictions (would come from actual model in full implementation)
def simulate_bert_disease_predictions(entities):
    # This is simplified - in reality would use a classifier on top of BERT embeddings
    disease_keywords = ['angina', 'heart attack', 'COPD', 'asthma', 'pneumonia', 
                       'depression', 'hypothyroidism', 'food poisoning', 'migraine']
    predicted = []
    for ent in eval(entities) if isinstance(entities, str) else entities:
        if ent['entity_group'] == 'DISEASE':
            predicted.append(ent['word'])
        elif ent['word'].lower() in disease_keywords:
            predicted.append(ent['word'])
    return list(set(predicted))

# Add simulated BERT predictions
bert_results['bert_diseases'] = bert_results['entities'].apply(simulate_bert_disease_predictions)

# Merge results (note: in full implementation would match by note_id)
merged_results = kg_results.merge(bert_results, left_index=True, right_index=True)

# Decision fusion function
def fuse_predictions(kg_diseases, bert_diseases, kg_weight=0.6, bert_weight=0.4):
    # Simple weighted fusion - more sophisticated methods possible
    all_diseases = list(set(kg_diseases + bert_diseases))
    
    # Create scores
    scores = {}
    for disease in all_diseases:
        kg_score = kg_weight if disease in kg_diseases else 0
        bert_score = bert_weight if disease in bert_diseases else 0
        scores[disease] = kg_score + bert_score
    
    # Sort by score
    sorted_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_diseases

# Apply fusion
merged_results['fused_predictions'] = merged_results.apply(
    lambda row: fuse_predictions(
        eval(row['predicted_diseases']) if isinstance(row['predicted_diseases'], str) else row['predicted_diseases'],
        row['bert_diseases']
    ),
    axis=1
)

# Save final results
merged_results.to_csv('final_predictions.csv', index=False)

# Display sample results
print(merged_results[['note_text', 'extracted_symptoms', 'predicted_diseases', 'bert_diseases', 'fused_predictions']].head())