import pandas as pd
import spacy
from spacy import displacy

# Load sample data (from preprocessing stage)
df = pd.read_csv('preprocessed_notes.csv')

# Define symptom keywords (this would be more comprehensive in real implementation)
SYMPTOM_KEYWORDS = {
    'chest pain', 'shortness of breath', 'fatigue', 'nausea', 'headache',
    'dizziness', 'vomiting', 'lack of appetite', 'weakness'
}

# Rule-based NER
def extract_symptoms(text):
    found_symptoms = []
    for symptom in SYMPTOM_KEYWORDS:
        if symptom in text.lower():
            found_symptoms.append(symptom)
    return found_symptoms

# Apply to dataframe
df['extracted_symptoms'] = df['note_text'].apply(extract_symptoms)

# Visualization using spaCy (simulated highlighting)
def visualize_ner(note_text, symptoms):
    colors = {"SYMPTOM": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    options = {"ents": ["SYMPTOM"], "colors": colors}
    
    # Create spaCy-like entities for visualization
    ents = []
    for symptom in symptoms:
        start = note_text.lower().find(symptom.lower())
        if start != -1:
            end = start + len(symptom)
            ents.append({"start": start, "end": end, "label": "SYMPTOM"})
    
    ex = {"text": note_text, "ents": ents}
    
    # Display with displacy
    html = displacy.render(ex, style="ent", manual=True, options=options)
    return html

# Save results with visualization
df['ner_visualization'] = df.apply(lambda row: visualize_ner(row['note_text'], row['extracted_symptoms']), axis=1)
df.to_csv('notes_with_symptoms.csv', index=False)

# Display sample results
print(df[['note_id', 'note_text', 'extracted_symptoms']].head())