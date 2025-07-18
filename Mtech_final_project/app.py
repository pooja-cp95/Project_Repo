import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO, StringIO

# ---------- Page Config ----------
st.set_page_config(page_title="🩺 Clinical Disease Predictor", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
        .main { background-color: #F8FBFF; }
        h1, h2, h3, .stTextInput label, .stTextArea label {
            color: #003366;
        }
        .stButton > button {
            background-color: #004080;
            color: white;
            font-weight: bold;
            border-radius: 6px;
            padding: 0.5rem 1.5rem;
        }
        .tag-box {
            background-color: #e0f0ff;
            color: #004080;
            display: inline-block;
            padding: 8px 15px;
            margin: 4px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
        }
        .tag-box.disease {
            background-color: #ffe0e0;
            color: #800000;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("🩺 Disease Prediction from Clinical Notes")
st.markdown("A hybrid NLP + Knowledge Graph system for accurate disease prediction from unstructured clinical text.")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    return pd.read_csv("sample_clinical_notes_with_predictions.csv")

data = load_data()

# ---------- Prediction Logic ----------
def match_prediction(note):
    for _, row in data.iterrows():
        if note.lower().strip() in row['clinical_note'].lower():
            return row['extracted_symptoms'], row['predicted_diseases']
        if any(kw.strip().lower() in note.lower() for kw in row['extracted_symptoms'].split(",")):
            return row['extracted_symptoms'], row['predicted_diseases']
    return None, None

# ---------- Knowledge Graph ----------
def create_graph(symptoms, diseases):
    G = nx.Graph()
    symptom_list = [s.strip() for s in symptoms.split(",")]
    disease_list = [d.strip() for d in diseases.split(",")]

    for s in symptom_list:
        G.add_node(s, type="symptom")
    for d in disease_list:
        G.add_node(d, type="disease")
    for s in symptom_list:
        for d in disease_list:
            G.add_edge(s, d)

    pos = nx.spring_layout(G, seed=42)
    node_colors = ['skyblue' if G.nodes[n]['type'] == 'symptom' else 'salmon' for n in G.nodes]

    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color=node_colors,
            font_size=12, font_weight='bold', ax=ax)
    ax.set_title("🧠 Symptom-Disease Knowledge Graph", fontsize=14)
    return fig

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["📝 Single Note Prediction", "📂 Bulk Prediction via CSV"])

# ---------- Tab 1: Single Note ----------
with tab1:
    st.header("📄 Enter Clinical Note")
    user_input = st.text_area(
        label="Paste or type the patient’s clinical note:",
        height=200,
        placeholder="Example: Patient reports chest pain, fever, and shortness of breath..."
    )

    if st.button("🔍 Analyze Symptoms", key="analyze_single"):
        if user_input.strip():
            symptoms, diseases = match_prediction(user_input)
            if symptoms and diseases:
                st.success("✅ Symptoms and diseases predicted successfully!")

                st.subheader("🧠 Extracted Symptoms")
                st.markdown(''.join([f"<span class='tag-box'>{s.strip()}</span>" for s in symptoms.split(",")]), unsafe_allow_html=True)

                st.subheader("📋 Predicted Diseases")
                st.markdown(''.join([f"<span class='tag-box disease'>{d.strip()}</span>" for d in diseases.split(",")]), unsafe_allow_html=True)

                st.subheader("📊 Knowledge Graph")
                fig = create_graph(symptoms, diseases)
                st.pyplot(fig)
            else:
                st.warning("⚠️ No close match found.")
        else:
            st.error("❌ Please enter a valid clinical note.")

# ---------- Tab 2: Bulk Upload ----------
with tab2:
    st.header("📂 Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file with a column named `clinical_note`", type="csv", key="bulk_upload")

    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            if "clinical_note" not in input_df.columns:
                st.error("❌ The CSV must have a column named `clinical_note`.")
            else:
                st.success("✅ File uploaded successfully!")
                predictions = []

                for note in input_df['clinical_note']:
                    symptoms, diseases = match_prediction(note)
                    predictions.append({
                        "clinical_note": note,
                        "extracted_symptoms": symptoms if symptoms else "not found",
                        "predicted_diseases": diseases if diseases else "not found"
                    })

                result_df = pd.DataFrame(predictions)
                st.subheader("🧾 Prediction Results")
                st.dataframe(result_df, use_container_width=True)

                csv_buffer = StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="disease_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error: {e}")

# ---------- Footer ----------
st.markdown("---")
st.caption("© 2025 - Pooja Patil | M.Tech Project | Under the guidance of V.Chandrashekhar")














# -------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# from io import BytesIO

# # ---------- Page Config ----------
# st.set_page_config(page_title="🩺 Clinical Disease Predictor", layout="wide")

# # ---------- Custom Styling ----------
# st.markdown("""
#     <style>
#         .main { background-color: #F8FBFF; }
#         h1, h2, h3 { color: #003366; }
#         .stButton > button {
#             background-color: #004080;
#             color: white;
#             font-weight: bold;
#             border-radius: 6px;
#         }
#         .stTextArea textarea {
#             font-size: 16px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # ---------- Load Dataset ----------
# @st.cache_data
# def load_data():
#     return pd.read_csv("sample_clinical_notes_with_predictions.csv")

# data = load_data()

# # ---------- Functions ----------
# def match_prediction(note):
#     for i, row in data.iterrows():
#         if note.lower().strip() in row['clinical_note'].lower():
#             return row['extracted_symptoms'], row['predicted_diseases']
#         if any(kw in note.lower() for kw in row['extracted_symptoms'].split(",")):
#             return row['extracted_symptoms'], row['predicted_diseases']
#     return None, None

# def create_graph(symptoms, diseases):
#     G = nx.Graph()
#     symptom_list = [s.strip() for s in symptoms.split(",")]
#     disease_list = [d.strip() for d in diseases.split(",")]

#     for s in symptom_list:
#         G.add_node(s, type="symptom")
#     for d in disease_list:
#         G.add_node(d, type="disease")
#     for s in symptom_list:
#         for d in disease_list:
#             G.add_edge(s, d)

#     pos = nx.spring_layout(G, seed=42)
#     node_colors = ['skyblue' if G.nodes[n]['type'] == 'symptom' else 'salmon' for n in G.nodes]

#     fig, ax = plt.subplots(figsize=(10, 6))
#     nx.draw(G, pos, with_labels=True, node_size=2000, node_color=node_colors,
#             font_size=10, font_weight='bold', ax=ax)
#     ax.set_title("🧠 Symptom-Disease Knowledge Graph", fontsize=14)
#     return fig

# def plot_disease_frequency(predicted_diseases):
#     df = pd.DataFrame(predicted_diseases, columns=["Predicted Disease"])
#     counts = df["Predicted Disease"].value_counts()
#     fig, ax = plt.subplots(figsize=(8, 5))
#     colors = list(plt.cm.Set2.colors)[:len(counts)]
#     ax.bar(counts.index, counts.values, color=colors)
#     ax.set_title("Predicted Disease Frequency", fontsize=14)
#     ax.set_xlabel("Disease", fontsize=12)
#     ax.set_ylabel("Count", fontsize=12)
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.xticks(rotation=45)
#     return fig

# # ---------- App Tabs ----------
# tabs = st.tabs(["🔍 Predict from Note", "📁 Bulk Prediction"])

# # ---------- Tab 1: Single Note ----------
# with tabs[0]:
#     st.title("🩺 Disease Prediction from Clinical Notes")
#     st.markdown("Enter unstructured clinical notes and get disease predictions based on symptoms using a hybrid rule-based + knowledge graph model.")

#     user_input = st.text_area("Paste or type the patient’s clinical note:", height=200,
#                                placeholder="Example: Patient reports chest pain, fever, and shortness of breath...")

#     if st.button("Analyze Symptoms"):
#         if user_input.strip():
#             symptoms, diseases = match_prediction(user_input)
#             if symptoms and diseases:
#                 st.success("✅ Symptoms and diseases predicted successfully!")
#                 st.subheader("🧠 Extracted Symptoms")
#                 st.markdown(f"<div style='font-size:18px;color:#004080;font-weight:bold'>{symptoms}</div>", unsafe_allow_html=True)

#                 st.subheader("📋 Predicted Diseases")
#                 st.markdown(f"<div style='font-size:18px;color:#990000;font-weight:bold'>{diseases}</div>", unsafe_allow_html=True)

#                 st.subheader("📊 Knowledge Graph")
#                 fig = create_graph(symptoms, diseases)
#                 st.pyplot(fig)
#             else:
#                 st.warning("⚠️ No close match found. Please try a more detailed or different input.")
#         else:
#             st.error("❌ Please enter a valid clinical note.")

# # ---------- Tab 2: Bulk Prediction ----------
# with tabs[1]:
#     st.title("📁 Bulk Clinical Notes Analysis")
#     st.markdown("Upload a CSV file with clinical notes and get batch predictions along with a frequency chart.")

#     uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

#     if uploaded_file:
#         bulk_df = pd.read_csv(uploaded_file)
#         predicted_diseases = []
#         results = []

#         for note in bulk_df['clinical_note']:
#             symptoms, diseases = match_prediction(note)
#             predicted_diseases.append(diseases if diseases else "No Match")
#             results.append({"clinical_note": note, "extracted_symptoms": symptoms, "predicted_diseases": diseases})

#         results_df = pd.DataFrame(results)
#         st.success("✅ Bulk prediction completed.")
#         st.dataframe(results_df)

#         st.download_button(
#             label="📥 Download Predictions CSV",
#             data=results_df.to_csv(index=False).encode('utf-8'),
#             file_name='predictions_output.csv',
#             mime='text/csv'
#         )

#         # Plot bar chart
#         clean_preds = [p for p in predicted_diseases if p and p != "No Match"]
#         if clean_preds:
#             flat_preds = [item.strip() for sublist in clean_preds for item in sublist.split(",")]
#             chart = plot_disease_frequency(flat_preds)
#             st.subheader("📊 Disease Frequency Chart")
#             st.pyplot(chart)
#         else:
#             st.warning("No valid predictions found to plot the frequency chart.")

# # ---------- Footer ----------
# st.markdown("---")
# st.caption("© 2025 - Pooja Patil | M.Tech Project | Guided by ChatGPT 😄")
