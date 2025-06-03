import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter
from itertools import chain

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = {}
if 'monthly_data' not in st.session_state:
    st.session_state.monthly_data = {}
if 'custom_labels' not in st.session_state:
    st.session_state.custom_labels = {}
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'cluster_model' not in st.session_state:
    st.session_state.cluster_model = None
if 'search_term' not in st.session_state:
    st.session_state.search_term = ""

def clean_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into string
    return ' '.join(tokens)

@st.cache_data
def process_uploaded_file(uploaded_file):
    """Process the uploaded CSV file and return cleaned dataframe."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Ticket ID', 'Subject', 'Full Text']
        
        # Verify required columns exist
        if not all(col in df.columns for col in required_columns):
            st.error("CSV must contain columns: 'Ticket ID', 'Subject', 'Full Text'")
            return None
        
        # Clean text data
        df['Cleaned Text'] = df['Full Text'].apply(clean_text)
        return df
    return None

def get_vectorizer(vectorizer_type, **kwargs):
    """Get the specified vectorizer with given parameters."""
    # Set more lenient defaults for small datasets
    default_params = {
        'min_df': 1,  # Accept terms that appear in at least 1 document
        'max_df': 1.0,  # Accept all terms, even if they appear in all documents
        'max_features': None,  # Don't limit the number of features
        'strip_accents': 'unicode',
        'lowercase': True,
        'stop_words': 'english'
    }
    
    # Update defaults with any provided parameters
    params = {**default_params, **kwargs}
    
    if vectorizer_type == "TF-IDF":
        return TfidfVectorizer(**params)
    elif vectorizer_type == "Count":
        return CountVectorizer(**params)
    return TfidfVectorizer(**params)

def get_clustering_algorithm(algorithm, n_clusters=5, **kwargs):
    """Get the specified clustering algorithm with parameters."""
    # Remove any irrelevant parameters for each algorithm
    if algorithm == "K-Means":
        # Only use n_clusters and random_state for K-Means
        kmeans_params = {
            'n_clusters': n_clusters,
            'random_state': 42
        }
        return KMeans(**kmeans_params)
    elif algorithm == "DBSCAN":
        # Only use eps and min_samples for DBSCAN
        dbscan_params = {
            'eps': kwargs.get('eps', 0.5),
            'min_samples': kwargs.get('min_samples', 5)
        }
        return DBSCAN(**dbscan_params)
    elif algorithm == "Hierarchical":
        # Only use n_clusters for Hierarchical
        hierarchical_params = {
            'n_clusters': n_clusters
        }
        return AgglomerativeClustering(**hierarchical_params)
    
    # Default to K-Means if unknown algorithm
    return KMeans(n_clusters=n_clusters, random_state=42)

def extract_ngrams(text, n=2):
    """Extract n-grams from text."""
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def get_representative_phrases(texts, n_phrases=3):
    """Get most common meaningful phrases from texts."""
    # Extract both unigrams and bigrams
    unigrams = list(chain.from_iterable(text.split() for text in texts))
    bigrams = list(chain.from_iterable(extract_ngrams(text) for text in texts))
    
    # Count frequencies
    unigram_freq = Counter(unigrams)
    bigram_freq = Counter(bigrams)
    
    # Filter out common stopwords and short terms
    stop_words = set(stopwords.words('english'))
    meaningful_unigrams = {word: freq for word, freq in unigram_freq.items() 
                         if word not in stop_words and len(word) > 2}
    meaningful_bigrams = {phrase: freq for phrase, freq in bigram_freq.items()
                         if not any(word in stop_words for word in phrase.split())}
    
    # Combine and sort phrases by frequency
    all_phrases = {**meaningful_unigrams, **meaningful_bigrams}
    return sorted(all_phrases.items(), key=lambda x: x[1], reverse=True)[:n_phrases]

def summarize_cluster_content(texts, subjects):
    """Generate a human-readable summary of cluster content."""
    # Get the most common words from cleaned texts
    words = ' '.join(texts).split()
    word_freq = Counter(words)
    common_words = [word for word, count in word_freq.most_common(5) if len(word) > 3]
    
    # Analyze subjects for common patterns
    subject_words = ' '.join(subjects).lower().split()
    subject_freq = Counter(subject_words)
    common_subject_terms = [word for word, count in subject_freq.most_common(5) if len(word) > 3]
    
    # Identify the main theme
    main_theme = common_subject_terms[0] if common_subject_terms else common_words[0] if common_words else "General"
    main_theme = main_theme.title()
    
    # Identify action/issue type
    action_words = {'error', 'request', 'bug', 'issue', 'problem', 'update', 'question', 'help', 'support'}
    action_type = next((word for word in subject_words if word in action_words), None)
    
    # Generate summary
    if action_type:
        summary = f"{main_theme} {action_type.title()}s"
    else:
        summary = f"{main_theme} Related Issues"
    
    # Add descriptive terms
    if len(common_words) > 1:
        descriptive_terms = [word.title() for word in common_words[1:3] if word != main_theme.lower()]
        if descriptive_terms:
            summary += f" ({' / '.join(descriptive_terms)})"
    
    return summary

def generate_cluster_label(cluster_texts, subjects, vectorizer=None, cluster_center=None):
    """Generate a descriptive label for a cluster using content analysis."""
    if not cluster_texts:
        return "Empty Cluster"
    
    try:
        # Generate human-readable summary
        summary = summarize_cluster_content(cluster_texts, subjects)
        
        # Get TF-IDF keywords if available for additional context
        keywords = []
        if vectorizer and cluster_center is not None and hasattr(vectorizer, 'get_feature_names_out'):
            try:
                feature_names = vectorizer.get_feature_names_out()
                top_indices = cluster_center.argsort()[-3:][::-1]
                keywords = [feature_names[i].title() for i in top_indices]
            except:
                pass
        
        # Combine summary with keywords if they add value
        if keywords and not any(keyword.lower() in summary.lower() for keyword in keywords):
            additional_context = ' / '.join(keywords)
            return f"{summary} - Keywords: {additional_context}"
        
        return summary
    except:
        return "Miscellaneous Issues"

def perform_clustering(df, clustering_params):
    """Perform text vectorization and clustering with given parameters."""
    # Extract parameters
    vectorizer_type = clustering_params["vectorizer_type"]
    algorithm = clustering_params["algorithm"]
    n_clusters = clustering_params.get("n_clusters", 5)
    min_df = clustering_params.get("min_df", 1)
    max_df = clustering_params.get("max_df", 1.0)
    
    try:
        # Get vectorizer and vectorize text
        vectorizer = get_vectorizer(
            vectorizer_type,
            min_df=min_df,
            max_df=max_df
        )
        vectors = vectorizer.fit_transform(df['Cleaned Text'])
        
        # Check if we have any features
        if vectors.shape[1] == 0:
            st.error("No features were extracted from the text. Try adjusting the min_df and max_df parameters.")
            return None, None, None
        
        # Ensure we don't try to create more clusters than samples
        n_samples = vectors.shape[0]
        if algorithm in ["K-Means", "Hierarchical"]:
            n_clusters = min(n_clusters, n_samples - 1)
            if n_clusters < 2:
                st.error("Not enough unique samples for clustering. Try adding more varied text data.")
                return None, None, None
        
        # Get clustering algorithm with appropriate parameters
        if algorithm == "DBSCAN":
            cluster_model = get_clustering_algorithm(
                algorithm,
                eps=clustering_params.get("eps", 0.5),
                min_samples=clustering_params.get("min_samples", 5)
            )
        else:
            cluster_model = get_clustering_algorithm(
                algorithm,
                n_clusters=n_clusters
            )
        
        # Perform clustering
        cluster_labels = cluster_model.fit_predict(vectors)
        
        # Calculate quality metrics
        silhouette, calinski = evaluate_clustering(vectors, cluster_labels)
        
        # Generate automatic labels for clusters
        cluster_labels_dict = {}
        for cluster_id in range(len(np.unique(cluster_labels))):
            cluster_texts = df[cluster_labels == cluster_id]['Cleaned Text'].tolist()
            cluster_subjects = df[cluster_labels == cluster_id]['Subject'].tolist()
            
            # Get cluster center if available
            cluster_center = None
            if hasattr(cluster_model, 'cluster_centers_'):
                cluster_center = cluster_model.cluster_centers_[cluster_id]
            
            # Generate label
            label = generate_cluster_label(cluster_texts, cluster_subjects, vectorizer, cluster_center)
            cluster_labels_dict[cluster_id] = label
        
        # Store models and labels in session state
        st.session_state.vectorizer = vectorizer
        st.session_state.cluster_model = cluster_model
        st.session_state.custom_labels = cluster_labels_dict
        
        return cluster_labels, silhouette, calinski
    
    except Exception as e:
        st.error(f"Error during clustering: {str(e)}")
        return None, None, None

def evaluate_clustering(vectors, labels):
    """Calculate clustering quality metrics."""
    if len(np.unique(labels)) < 2:
        return None, None
    
    try:
        silhouette = silhouette_score(vectors, labels)
        calinski = calinski_harabasz_score(vectors, labels)
        return silhouette, calinski
    except:
        return None, None

def filter_dataframe(df, search_term):
    """Filter dataframe based on search term."""
    if not search_term:
        return df
    
    # Convert search term to lowercase for case-insensitive search
    search_term = search_term.lower()
    
    # Search in Subject and Full Text
    mask = (df['Subject'].str.lower().str.contains(search_term, na=False) |
            df['Full Text'].str.lower().str.contains(search_term, na=False))
    
    return df[mask]

def highlight_text(text, search_term):
    """Highlight search term in text using markdown."""
    if not search_term or not isinstance(text, str):
        return text
    
    # Case-insensitive replacement
    pattern = re.compile(re.escape(search_term), re.IGNORECASE)
    return pattern.sub(f"**{search_term}**", text)

def get_cluster_keywords(vectorizer, cluster_center, top_n=5):
    """Get top keywords for a cluster based on TF-IDF weights."""
    feature_names = vectorizer.get_feature_names_out()
    top_indices = cluster_center.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]

def calculate_cluster_similarity(keywords1, keywords2):
    """Calculate Jaccard similarity between two sets of keywords."""
    set1 = set(keywords1)
    set2 = set(keywords2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def display_cluster_summary(df, month=None):
    """Display summary of each cluster with example tickets and full ticket list."""
    # Get filtered dataframe if search term exists
    filtered_df = filter_dataframe(df, st.session_state.search_term)
    
    if len(filtered_df) == 0:
        st.warning("No tickets match the search criteria.")
        return
    
    for cluster_id in range(len(df['Cluster'].unique())):
        cluster_data = filtered_df[filtered_df['Cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            continue
        
        # Get cluster label
        cluster_label = st.session_state.custom_labels.get(cluster_id, f"Cluster {cluster_id}")
        
        # Create an expander for each cluster
        with st.expander(f"### {cluster_label} ({len(cluster_data)} tickets)", expanded=False):
            # Display cluster statistics
            st.write(f"#### Cluster Statistics")
            st.write(f"Total tickets: {len(cluster_data)}")
            
            # Display all tickets in a table
            st.write("#### All Tickets in Cluster")
            st.dataframe(
                cluster_data[['Ticket ID', 'Subject', 'Full Text']],
                use_container_width=True
            )
            
            # Add download button for cluster data
            csv = cluster_data.to_csv(index=False)
            st.download_button(
                label="Download Cluster Data",
                data=csv,
                file_name=f"cluster_{cluster_id}_{month if month else 'data'}.csv",
                mime="text/csv",
                key=f"download_{cluster_id}"
            )
        
        st.write("---")

def plot_cluster_distribution(df, month):
    """Create a bar chart of cluster distribution."""
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    
    # Add custom labels
    cluster_counts['Label'] = cluster_counts['Cluster'].apply(
        lambda x: st.session_state.custom_labels.get(x, f"Cluster {x}")
    )
    
    fig = px.bar(
        cluster_counts,
        x='Label',
        y='Count',
        title=f'Cluster Distribution - {month}',
        labels={'Count': 'Number of Tickets', 'Label': 'Cluster'}
    )
    return fig

def plot_cluster_metrics(metrics_history):
    """Plot clustering quality metrics."""
    if not metrics_history:
        return None
    
    fig = go.Figure()
    
    # Add traces for each metric
    x = list(range(2, len(metrics_history) + 2))
    
    silhouette_scores = [m['silhouette'] for m in metrics_history if m['silhouette'] is not None]
    calinski_scores = [m['calinski'] for m in metrics_history if m['calinski'] is not None]
    
    if silhouette_scores:
        fig.add_trace(go.Scatter(
            x=x[:len(silhouette_scores)],
            y=silhouette_scores,
            name='Silhouette Score'
        ))
    
    if calinski_scores:
        fig.add_trace(go.Scatter(
            x=x[:len(calinski_scores)],
            y=calinski_scores,
            name='Calinski-Harabasz Score'
        ))
    
    fig.update_layout(
        title='Clustering Quality Metrics',
        xaxis_title='Number of Clusters',
        yaxis_title='Score',
        showlegend=True
    )
    
    return fig

# App title and description
st.title("Support Ticket Clustering App")
st.write("Upload your support ticket CSV files to analyze ticket clusters and trends.")

# Month selection for file upload
selected_month = st.selectbox(
    "Select month for the data",
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"]
)

# File upload section
uploaded_file = st.file_uploader(
    f"Upload CSV file for {selected_month}",
    type="csv",
    help="File should contain columns: Ticket ID, Subject, Full Text"
)

if uploaded_file:
    # Process the uploaded file
    df = process_uploaded_file(uploaded_file)
    
    if df is not None:
        # Store the processed data for the selected month
        st.session_state.monthly_data[selected_month] = df
        st.success(f"Successfully processed {len(df)} tickets for {selected_month}!")
        
        # Display sample of processed data
        st.subheader(f"Sample of Processed Data - {selected_month}")
        st.dataframe(df[['Ticket ID', 'Subject', 'Cleaned Text']].head())
        
        # Clustering section
        st.subheader("Ticket Clustering")
        
        # Clustering parameters
        st.write("### Clustering Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            vectorizer_type = st.selectbox(
                "Vectorization Method",
                ["TF-IDF", "Count"],
                help="Choose the text vectorization method"
            )
            
            algorithm = st.selectbox(
                "Clustering Algorithm",
                ["K-Means", "DBSCAN", "Hierarchical"],
                help="Choose the clustering algorithm"
            )
        
        with col2:
            min_df = st.slider(
                "Minimum Document Frequency",
                min_value=1,
                max_value=10,
                value=1,  # Changed default to 1
                help="Minimum number of documents a term must appear in"
            )
            
            max_df = st.slider(
                "Maximum Document Frequency",
                min_value=0.0,
                max_value=1.0,
                value=1.0,  # Changed default to 1.0
                help="Maximum fraction of documents a term can appear in"
            )
        
        # Algorithm-specific parameters
        if algorithm == "K-Means" or algorithm == "Hierarchical":
            n_clusters = st.slider(
                "Number of clusters",
                min_value=2,
                max_value=20,
                value=5,
                help="Select the number of clusters to generate"
            )
        elif algorithm == "DBSCAN":
            eps = st.slider(
                "DBSCAN epsilon",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                help="Maximum distance between samples for DBSCAN"
            )
        
        # Search and filter section
        st.subheader("Search and Filter")
        search_term = st.text_input(
            "Search tickets",
            value=st.session_state.search_term,
            help="Enter keywords to filter tickets"
        )
        st.session_state.search_term = search_term
        
        if st.button("Perform Clustering"):
            # Prepare clustering parameters
            clustering_params = {
                "vectorizer_type": vectorizer_type,
                "algorithm": algorithm,
                "min_df": min_df,
                "max_df": max_df,
                "n_clusters": n_clusters if algorithm != "DBSCAN" else None,
                "eps": eps if algorithm == "DBSCAN" else None
            }
            
            # Perform clustering
            cluster_labels, silhouette, calinski = perform_clustering(df, clustering_params)
            
            # Add cluster labels to dataframe
            df['Cluster'] = cluster_labels
            st.session_state.monthly_data[selected_month] = df
            
            # Display clustering metrics
            if silhouette is not None and calinski is not None:
                st.write("### Clustering Quality Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score", f"{silhouette:.3f}")
                with col2:
                    st.metric("Calinski-Harabasz Score", f"{calinski:.1f}")
            
            # Display cluster summary
            st.subheader(f"Cluster Summary - {selected_month}")
            display_cluster_summary(df, selected_month)
            
            # Show cluster distribution plot
            st.plotly_chart(plot_cluster_distribution(df, selected_month))
        
        # If clustering was previously performed, show the summary
        elif 'Cluster' in st.session_state.monthly_data[selected_month].columns:
            st.subheader(f"Cluster Summary - {selected_month}")
            display_cluster_summary(st.session_state.monthly_data[selected_month], selected_month)
            
            # Show cluster distribution plot
            st.plotly_chart(plot_cluster_distribution(
                st.session_state.monthly_data[selected_month],
                selected_month
            ))

# Month-to-month comparison section
if len(st.session_state.monthly_data) > 1:
    st.subheader("Month-to-Month Comparison")
    
    # Select months to compare
    months = list(st.session_state.monthly_data.keys())
    col1, col2 = st.columns(2)
    
    with col1:
        month1 = st.selectbox("Select first month", months, index=0)
    with col2:
        month2 = st.selectbox("Select second month", months, index=min(1, len(months)-1))
    
    if month1 != month2:
        df1 = st.session_state.monthly_data[month1]
        df2 = st.session_state.monthly_data[month2]
        
        if 'Cluster' in df1.columns and 'Cluster' in df2.columns:
            # Compare cluster distributions
            fig1 = plot_cluster_distribution(df1, month1)
            fig2 = plot_cluster_distribution(df2, month2)
            
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)
            
            # Compare cluster similarities if vectorizer is available
            if st.session_state.vectorizer and st.session_state.cluster_model:
                st.subheader("Cluster Topic Similarity")
                
                for cluster1 in range(len(df1['Cluster'].unique())):
                    keywords1 = get_cluster_keywords(
                        st.session_state.vectorizer,
                        st.session_state.cluster_model.cluster_centers_[cluster1]
                    )
                    
                    for cluster2 in range(len(df2['Cluster'].unique())):
                        keywords2 = get_cluster_keywords(
                            st.session_state.vectorizer,
                            st.session_state.cluster_model.cluster_centers_[cluster2]
                        )
                        
                        similarity = calculate_cluster_similarity(keywords1, keywords2)
                        if similarity > 0.3:  # Only show significant similarities
                            st.write(
                                f"Similarity between {month1} Cluster {cluster1} and "
                                f"{month2} Cluster {cluster2}: {similarity:.2f}"
                            ) 