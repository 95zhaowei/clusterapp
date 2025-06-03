import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
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
if 'topic_model' not in st.session_state:
    st.session_state.topic_model = None
if 'topic_summaries' not in st.session_state:
    st.session_state.topic_summaries = {}

def clean_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

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

def create_topic_model(min_topic_size=5):
    """Create a BERTopic model with custom parameters."""
    # Initialize UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    # Initialize HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model='all-MiniLM-L6-v2',
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True
    )
    
    return topic_model

def generate_natural_language_summary(topic_terms):
    """Generate a natural language summary from topic terms."""
    if not topic_terms or len(topic_terms) == 0:
        return "No clear pattern identified in this topic."
    
    # Extract terms and their scores
    terms_with_scores = [(term, score) for term, score in topic_terms[:10]]
    
    # Group terms by importance
    high_importance = terms_with_scores[:3]  # Top 3 terms
    medium_importance = terms_with_scores[3:6]  # Next 3 terms
    additional_terms = terms_with_scores[6:]  # Remaining terms
    
    # Create the summary
    summary_parts = []
    
    # Start with the main theme using top terms
    main_terms = [term for term, _ in high_importance]
    if main_terms:
        if len(main_terms) == 1:
            summary_parts.append(f"This topic primarily focuses on issues related to **{main_terms[0]}**")
        else:
            terms_str = ", ".join(f"**{term}**" for term in main_terms[:-1]) + f" and **{main_terms[-1]}**"
            summary_parts.append(f"This topic encompasses issues involving {terms_str}")
    
    # Add medium importance terms
    if medium_importance:
        med_terms = [f"**{term}**" for term, _ in medium_importance]
        summary_parts.append(f"Common related aspects include {', '.join(med_terms)}")
    
    # Add additional context from remaining terms
    if additional_terms:
        add_terms = [f"**{term}**" for term, _ in additional_terms]
        summary_parts.append(f"Other relevant factors involve {', '.join(add_terms)}")
    
    # Add importance scores for key terms
    summary_parts.append("\n\n**Key term importance:**")
    for term, score in high_importance:
        summary_parts.append(f"• **{term}**: {score:.3f}")
    
    # Join all parts
    return "\n\n".join(summary_parts)

def perform_clustering(df, clustering_params):
    """Perform text clustering using BERTopic."""
    try:
        # Get parameters
        min_topic_size = clustering_params.get("min_topic_size", 5)
        
        # Combine subject and full text for better context
        texts = df['Subject'] + " " + df['Cleaned Text']
        
        # Create and fit topic model
        topic_model = create_topic_model(min_topic_size=min_topic_size)
        topics, probs = topic_model.fit_transform(texts)
        
        if topics is None or len(topics) == 0:
            st.error("No topics were generated. Please try adjusting the minimum topic size.")
            return None, None, "No topics were generated"
        
        # Store model in session state
        st.session_state.topic_model = topic_model
        
        return topics, probs, None
    
    except Exception as e:
        st.error(f"Error during clustering: {str(e)}")
        return None, None, str(e)

def display_cluster_summary(df, month=None):
    """Display summary of each cluster with example tickets."""
    # Get unique topics excluding outliers (-1)
    unique_topics = sorted([t for t in df['Cluster'].unique() if t != -1])
    
    if not unique_topics:
        st.warning("No topics were found. Try adjusting the clustering parameters.")
        return
    
    st.write("### Topic Analysis")
    
    # Display detailed analysis for each topic
    for topic in unique_topics:
        cluster_data = df[df['Cluster'] == topic]
        
        # Get top terms for the topic
        top_terms = st.session_state.topic_model.get_topic(topic)
        top_words = [word for word, _ in top_terms[:3]] if top_terms else []
        topic_header = f"Topic {topic}: {' / '.join(top_words)}" if top_words else f"Topic {topic}"
        
        # Create an expander for each cluster
        with st.expander(f"### {topic_header} ({len(cluster_data)} tickets)", expanded=False):
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📝 Overview & Tickets", "📊 Term Analysis"])
            
            with tab1:
                # Display topic description
                if top_terms:
                    description = generate_topic_description(top_terms)
                    st.markdown(description)
                else:
                    st.warning("No terms available for this topic.")
                
                st.write("---")
                
                # Display tickets table
                st.write("#### Tickets in This Topic")
                
                # Display the tickets table with ID, Subject, and Full Text
                display_df = cluster_data[['Ticket ID', 'Subject', 'Full Text']]
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Ticket ID": st.column_config.TextColumn("Ticket ID", width="small"),
                        "Subject": st.column_config.TextColumn("Subject", width="medium"),
                        "Full Text": st.column_config.TextColumn("Full Text", width="large")
                    }
                )
                
                # Create download data once and store in session state
                if f'download_data_{topic}' not in st.session_state:
                    st.session_state[f'download_data_{topic}'] = cluster_data.to_csv(index=False).encode('utf-8')
                
                # Add download button using stored data
                st.download_button(
                    label="📥 Download Topic Data",
                    data=st.session_state[f'download_data_{topic}'],
                    file_name=f"topic_{topic}_{month if month else 'data'}.csv",
                    mime="text/csv",
                    key=f"download_{topic}"
                )
            
            with tab2:
                if st.session_state.topic_model:
                    # Display top terms with scores
                    st.write("#### Key Terms and Importance")
                    if top_terms:
                        # Create a more visual representation of term importance
                        terms_df = pd.DataFrame(top_terms, columns=['Term', 'Score'])
                        terms_df['Score'] = terms_df['Score'].round(3)
                        
                        # Create a bar chart for term importance
                        fig = px.bar(
                            terms_df.head(10),
                            x='Term',
                            y='Score',
                            title='Term Importance in Topic',
                            labels={'Score': 'Importance Score', 'Term': 'Key Terms'}
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show detailed table
                        st.write("#### Detailed Term Analysis")
                        st.dataframe(
                            terms_df,
                            use_container_width=True,
                            column_config={
                                "Term": st.column_config.TextColumn("Term", width="medium"),
                                "Score": st.column_config.ProgressColumn(
                                    "Importance Score",
                                    min_value=0,
                                    max_value=float(terms_df['Score'].max()),
                                    format="%.3f"
                                )
                            }
                        )
        
        st.write("---")

def generate_topic_description(topic_terms):
    """Generate a natural language description from topic terms."""
    if not topic_terms or len(topic_terms) == 0:
        return "No clear pattern identified in this topic."
    
    # Extract terms and their scores
    terms_with_scores = [(term, score) for term, score in topic_terms[:10]]
    
    # Group terms by importance
    high_importance = terms_with_scores[:3]  # Top 3 terms
    medium_importance = terms_with_scores[3:6]  # Next 3 terms
    additional_terms = terms_with_scores[6:]  # Remaining terms
    
    # Create description parts
    description_parts = []
    
    # Start with the main theme using top terms
    main_terms = [term for term, _ in high_importance]
    if main_terms:
        if len(main_terms) == 1:
            description_parts.append(f"Tickets in this topic are primarily focused on **{main_terms[0]}** related issues.")
        else:
            terms_str = ", ".join(f"**{term}**" for term in main_terms[:-1]) + f" and **{main_terms[-1]}**"
            description_parts.append(f"This group of tickets deals with issues involving {terms_str}.")
    
    # Add medium importance terms with their context
    if medium_importance:
        med_terms = [f"**{term}**" for term, _ in medium_importance]
        description_parts.append(f"These issues frequently involve {', '.join(med_terms)}.")
    
    # Add additional context from remaining terms
    if additional_terms:
        add_terms = [f"**{term}**" for term, _ in additional_terms]
        description_parts.append(f"Additional aspects often include {', '.join(add_terms)}.")
    
    # Join all parts with proper spacing
    return "\n\n".join(description_parts)

def plot_cluster_distribution(df, month):
    """Create a bar chart of cluster distribution."""
    # Exclude outliers (-1) from visualization
    cluster_counts = df[df['Cluster'] != -1]['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    
    # Add custom labels
    cluster_counts['Label'] = cluster_counts['Cluster'].apply(
        lambda x: st.session_state.custom_labels.get(x, f"Topic {x}")
    )
    
    fig = px.bar(
        cluster_counts,
        x='Label',
        y='Count',
        title=f'Topic Distribution - {month}',
        labels={'Count': 'Number of Tickets', 'Label': 'Topic'}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis={'tickmode': 'array'},
        height=500
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
        
        min_topic_size = st.slider(
            "Minimum Topic Size",
            min_value=2,
            max_value=20,
            value=3,
            help="Minimum number of tickets in a topic"
        )
        
        if st.button("Perform Clustering"):
            with st.spinner("Performing clustering..."):
                try:
                    # Prepare clustering parameters
                    clustering_params = {
                        "min_topic_size": min_topic_size
                    }
                    
                    # Perform clustering
                    topics, probs, error = perform_clustering(df, clustering_params)
                    
                    if topics is not None and error is None:
                        # Add cluster labels to dataframe
                        df['Cluster'] = topics
                        st.session_state.monthly_data[selected_month] = df
                        
                        # Count number of meaningful topics
                        n_topics = len(set([t for t in topics if t != -1]))
                        st.success(f"Successfully identified {n_topics} topics!")
                        
                        # Display cluster summary
                        display_cluster_summary(df, selected_month)
                    else:
                        st.error(f"Clustering failed: {error if error else 'Unknown error'}")
                        st.info("Try adjusting the minimum topic size or check your input data.")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please try again with different parameters or check your input data.")

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
            # Compare distributions
            fig1 = plot_cluster_distribution(df1, month1)
            fig2 = plot_cluster_distribution(df2, month2)
            
            st.plotly_chart(fig1)
            st.plotly_chart(fig2) 