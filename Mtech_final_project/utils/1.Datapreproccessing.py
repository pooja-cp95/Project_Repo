import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load clinical notes data (simulated since we don't have direct access to MIMIC-III here)
def load_sample_data():
    data = {
        'note_id': [1, 2, 3, 4, 5, 6, 7],
        'note_text': [
            "Patient reports chest pain and shortness of breath during physical activity",
            "Complaints of fatigue nausea and headache after taking new medication",
            "Headache and dizziness experienced for the last 3 days",
            "Severe nausea and vomiting noted might be food poisoning",
            "Patient stable no symptoms currently",
            "Reports fatigue lack of appetite and weakness",
            "Chest pain worsens while climbing stairs shortness of breath noted"
        ]
    }
    return pd.DataFrame(data)

# Preprocessing functions
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# EDA functions
def perform_eda(df):
    # Add note length feature
    df['note_length'] = df['note_text'].apply(lambda x: len(x.split()))
    
    # Plot distribution of note lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(df['note_length'], kde=True)
    plt.title('Distribution of Clinical Note Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()
    
    # Boxplot for outlier detection
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['note_length'])
    plt.title('Boxplot of Note Lengths')
    plt.xlabel('Number of Words')
    plt.show()
    
    # Word cloud
    all_text = ' '.join(df['note_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Clinical Notes')
    plt.show()
    
    # Top 10 frequent words
    words = ' '.join(df['note_text']).split()
    freq_dist = nltk.FreqDist(words)
    top_words = freq_dist.most_common(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[count for word, count in top_words], y=[word for word, count in top_words])
    plt.title('Top 10 Most Frequent Words')
    plt.xlabel('Frequency')
    plt.show()

# Main execution
df = load_sample_data()
df['cleaned_text'] = df['note_text'].apply(preprocess_text)
perform_eda(df)
df.to_csv('preprocessed_notes.csv', index=False)