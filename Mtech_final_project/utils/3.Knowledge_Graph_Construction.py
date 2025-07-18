import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load symptom data (from previous stage)
df = pd.read_csv('notes_with_symptoms.csv')

# Define symptom-disease relationships (would come from UMLS/SymCat in real implementation)
symptom_disease_map = {
    'chest pain': ['angina', 'heart attack', 'anxiety'],
    'shortness of breath': ['COPD', 'asthma', 'pneumonia'],
    'fatigue': ['depression', 'hypothyroidism', 'anemia'],
    'nausea': ['food poisoning', 'pregnancy', 'migraine'],
    'headache': ['migraine', 'tension headache', 'hypertension'],
    'dizziness': ['vertigo', 'low blood pressure', 'anemia'],
    'vomiting': ['food poisoning', 'migraine', 'pregnancy'],
    'lack of appetite': ['depression', 'infection', 'cancer'],
    'weakness': ['stroke', 'multiple sclerosis', 'anemia']
}

# Create knowledge graph
def build_knowledge_graph():
    G = nx.Graph()
    
    # Add nodes and edges
    for symptom, diseases in symptom_disease_map.items():
        G.add_node(symptom, type='symptom', color='lightblue')
        for disease in diseases:
            G.add_node(disease, type='disease', color='lightgreen')
            G.add_edge(symptom, disease, weight=1)  # weight could be based on probability
            
    return G

# Visualize the knowledge graph
def visualize_knowledge_graph(G):
    plt.figure(figsize=(12, 12))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Get node colors
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=node_colors, 
            font_size=10, font_weight='bold', edge_color='gray')
    
    plt.title('Medical Knowledge Graph: Symptoms to Diseases')
    plt.show()

# Query the knowledge graph for diseases
def query_diseases(G, symptoms):
    diseases = set()
    for symptom in symptoms:
        if symptom in G:
            for neighbor in G.neighbors(symptom):
                if G.nodes[neighbor]['type'] == 'disease':
                    diseases.add(neighbor)
    return list(diseases)

# Main execution
kg = build_knowledge_graph()
visualize_knowledge_graph(kg)

# Apply disease prediction to each note
df['predicted_diseases'] = df['extracted_symptoms'].apply(
    lambda x: query_diseases(kg, eval(x)) if isinstance(x, str) else query_diseases(kg, x)
)

# Save results
df.to_csv('notes_with_disease_predictions.csv', index=False)
print(df[['note_id', 'extracted_symptoms', 'predicted_diseases']].head())