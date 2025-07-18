
# ----------------------------------------------
# 1. Input Layer (Clinical Text Input)
# ----------------------------------------------
def get_clinical_input():
    return input("Enter clinical note: ")

# ----------------------------------------------
# 2. Text Preprocessing
# ----------------------------------------------
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------------------------
# 3. NLP Module – Named Entity Recognition using Transformer
# ----------------------------------------------
from transformers import pipeline

def extract_entities(text):
    ner = pipeline("ner", model="dslim/bert-base-NER")
    entities = ner(text)
    symptoms = [ent['word'] for ent in entities if ent['entity'].startswith("B-")]
    return list(set(symptoms))

# ----------------------------------------------
# 4. Knowledge Graph Construction (Symptom-Disease Mapping)
# ----------------------------------------------
import networkx as nx

def build_knowledge_graph(symptoms):
    graph = nx.Graph()
    symptom_disease_map = {
        "chest pain": ["angina", "heart attack"],
        "fever": ["flu", "malaria"],
        "fatigue": ["anemia", "thyroid disorder"],
        "headache": ["migraine", "brain tumor"]
    }
    for symptom in symptoms:
        if symptom in symptom_disease_map:
            for disease in symptom_disease_map[symptom]:
                graph.add_edge(symptom, disease)
    return graph

# ----------------------------------------------
# 5. Decision Fusion (Simple Rule-based Scoring)
# ----------------------------------------------
from collections import Counter

def fuse_decisions(graph):
    diseases = [node for node in graph.nodes if " " in node and node not in graph]
    counter = Counter()
    for edge in graph.edges:
        if edge[1] in diseases:
            counter[edge[1]] += 1
    return counter.most_common(3)

# ----------------------------------------------
# 6. Disease Prediction Output
# ----------------------------------------------
def predict_disease(note):
    cleaned = preprocess_text(note)
    symptoms = extract_entities(cleaned)
    graph = build_knowledge_graph(symptoms)
    prediction = fuse_decisions(graph)
    return symptoms, prediction

# Example Run
if __name__ == "__main__":
    note = get_clinical_input()
    symptoms, prediction = predict_disease(note)
    print("Extracted Symptoms:", symptoms)
    print("Predicted Diseases:", prediction)
