import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import display, HTML

# Load final predictions
df = pd.read_csv('final_predictions.csv')

# Evaluation metrics (simulated - would need ground truth for real evaluation)
def calculate_metrics(df):
    # In real implementation, would compare with actual diagnoses
    metrics = {
        'avg_predictions_per_note': df['fused_predictions'].apply(lambda x: len(eval(str(x)))).mean(),
        'kg_unique_diseases': len(set([d for sublist in df['predicted_diseases'].apply(eval) for d in sublist])),
        'bert_unique_diseases': len(set([d for sublist in df['bert_diseases'] for d in sublist])),
        'fused_unique_diseases': len(set([d[0] for sublist in df['fused_predictions'].apply(eval) for d in sublist]))
    }
    return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

# Explainability visualization
def create_explainability_report(row, kg):
    note_text = row['note_text']
    symptoms = eval(row['extracted_symptoms']) if isinstance(row['extracted_symptoms'], str) else row['extracted_symptoms']
    diseases = [d[0] for d in eval(row['fused_predictions'])]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Highlight symptoms in text
    highlighted_text = note_text
    for symptom in symptoms:
        highlighted_text = highlighted_text.replace(symptom, f'<span style="background-color: yellow">{symptom}</span>')
    
    ax1.axis('off')
    ax1.set_title('Note with Symptoms Highlighted')
    ax1.text(0.1, 0.5, highlighted_text, ha='left', va='center', wrap=True)
    
    # 2. Show subgraph of relevant knowledge graph paths
    subgraph = nx.Graph()
    for symptom in symptoms:
        if symptom in kg:
            subgraph.add_node(symptom, color='lightblue')
            for neighbor in kg.neighbors(symptom):
                if kg.nodes[neighbor]['type'] == 'disease' and neighbor in diseases:
                    subgraph.add_node(neighbor, color='lightgreen')
                    subgraph.add_edge(symptom, neighbor)
    
    pos = nx.spring_layout(subgraph)
    colors = [subgraph.nodes[n]['color'] for n in subgraph.nodes()]
    nx.draw(subgraph, pos, with_labels=True, node_color=colors, ax=ax2)
    ax2.set_title('Knowledge Graph Paths for Predicted Diseases')
    
    plt.show()
    
    # Return top diseases
    return diseases[:3]  # return top 3 predictions

# Main execution
metrics = calculate_metrics(df)
print("Evaluation Metrics:")
print(metrics)

# Rebuild knowledge graph for visualization
kg = build_knowledge_graph()  # using function from Stage 3

# Generate explainability report for first few notes
for idx, row in df.head(2).iterrows():
    print(f"\nExplainability Report for Note ID {row['note_id']}")
    top_diseases = create_explainability_report(row, kg)
    print(f"Top predicted diseases: {', '.join(top_diseases)}")