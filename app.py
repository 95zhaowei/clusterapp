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
from datetime import datetime, timedelta
from collections import Counter
from itertools import chain
from transformers import pipeline

# Initialize summarization model
@st.cache_resource
def get_summarizer():
    """Initialize and return the BART summarization model."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize session state for summaries
if 'ticket_summaries' not in st.session_state:
    st.session_state.ticket_summaries = {}

def parse_date(date_str):
    """Parse date string to datetime object."""
    try:
        # Try common date formats
        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    except:
        return None

def get_time_period(date, period='week'):
    """Convert date to specified time period."""
    if period == 'day':
        return date.strftime('%Y-%m-%d')
    elif period == 'week':
        return f"{date.year}-W{date.isocalendar()[1]}"
    else:  # month
        return date.strftime('%Y-%m')

def analyze_topic_trends(df, time_period='week'):
    """Analyze topic trends over time."""
    if 'Date' not in df.columns or 'Cluster' not in df.columns:
        return None
    
    # Convert dates and group by time period
    df['Period'] = df['Date'].apply(lambda x: get_time_period(x, time_period))
    
    # Calculate topic frequencies per period
    topic_trends = df.groupby(['Period', 'Cluster']).size().unstack(fill_value=0)
    
    # Sort by period
    topic_trends = topic_trends.sort_index()
    
    return topic_trends

def plot_topic_trends(topic_trends, title="Topic Trends Over Time"):
    """Create an interactive line plot of topic trends with descriptive topic names."""
    if topic_trends is None or topic_trends.empty:
        return None
    
    # Create a mapping of topic numbers to descriptive names
    topic_names = {}
    for topic_num in topic_trends.columns:
        if topic_num != -1:  # Skip outlier topic
            # Get top terms for the topic
            top_terms = st.session_state.topic_model.get_topic(topic_num)
            if top_terms:
                # Use top 3 terms as the topic name
                top_words = [word for word, _ in top_terms[:3]]
                topic_names[topic_num] = f"Topic {topic_num}: {' / '.join(top_words)}"
            else:
                topic_names[topic_num] = f"Topic {topic_num}"
        else:
            topic_names[topic_num] = "Outliers"
    
    # Create a copy of the dataframe with renamed columns
    renamed_trends = topic_trends.rename(columns=topic_names)
    
    # Melt the dataframe for plotting
    df_melted = renamed_trends.reset_index().melt(
        id_vars=['Period'],
        var_name='Topic',
        value_name='Count'
    )
    
    # Create line plot
    fig = px.line(
        df_melted,
        x='Period',
        y='Count',
        color='Topic',
        title=title,
        labels={'Period': 'Time Period', 'Count': 'Number of Tickets', 'Topic': 'Topic Clusters'},
    )
    
    # Customize the layout with improved legend appearance
    fig.update_layout(
        font=dict(
            family="Source Sans Pro, sans-serif",
            color="black"
        ),
        xaxis=dict(
            title=dict(
                text="Time Period",
                font=dict(size=14, color="black")
            ),
            tickfont=dict(size=12, color="black"),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            linecolor='black'
        ),
        yaxis=dict(
            title=dict(
                text="Number of Tickets",
                font=dict(size=14, color="black")
            ),
            tickfont=dict(size=12, color="black"),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            linecolor='black'
        ),
        legend_title=dict(
            text="Topic Clusters",
            font=dict(
                size=16,
                color='black',
                family="Source Sans Pro, sans-serif"
            )
        ),
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="white",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            font=dict(
                size=12,
                color='black',
                family="Source Sans Pro, sans-serif"
            ),
            itemsizing='constant',
            itemwidth=30,
            tracegroupgap=5
        ),
        margin=dict(r=350, t=50, b=50, l=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update line styles for better visibility
    for trace in fig.data:
        trace.update(
            line=dict(width=2),
            mode='lines+markers',
            marker=dict(size=6)
        )
    
    return fig

def summarize_ticket(text, max_length=130):
    """Summarize ticket text using BART model."""
    if not text or len(text.split()) < 30:  # Don't summarize short texts
        return text
    
    try:
        summarizer = get_summarizer()
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )[0]['summary_text']
        return summary
    except Exception as e:
        st.warning(f"Could not generate summary: {str(e)}")
        return text

def extract_action_items(text):
    """Extract action items and key points from text."""
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    action_items = []
    for sentence in sentences:
        # Look for action-oriented language
        if any(word in sentence.lower() for word in [
            'need', 'must', 'should', 'will', 'required',
            'action', 'todo', 'fix', 'implement', 'update'
        ]):
            action_items.append(sentence)
    
    return action_items

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
if 'topic_model' not in st.session_state:
    st.session_state.topic_model = None
if 'topic_summaries' not in st.session_state:
    st.session_state.topic_summaries = {}

def clean_text(text):
    """Clean and preprocess text data while preserving technical terms."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace common technical patterns with placeholders
    text = re.sub(r'(?<=[a-zA-Z])_(?=[a-zA-Z])', ' UNDERSCORE ', text)  # preserve snake_case
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # split camelCase
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)  # split letters and numbers
    
    # Remove URLs but keep domains
    text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*', '', text)
    
    # Keep alphanumeric, spaces, and important punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?@#$%&*()-_]', ' ', text)
    
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    
    # Replace placeholders back
    text = text.replace(' UNDERSCORE ', '_')
    
    return text

@st.cache_data
def process_uploaded_file(uploaded_file):
    """Process the uploaded CSV file and return cleaned dataframe."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Ticket ID', 'Subject', 'Full Text', 'Date']
        
        # Verify required columns exist
        if not all(col in df.columns for col in required_columns):
            st.error("CSV must contain columns: 'Ticket ID', 'Subject', 'Full Text', 'Date'")
            return None
        
        # Clean text data
        df['Cleaned Text'] = df['Full Text'].apply(clean_text)
        
        # Parse dates
        df['Date'] = df['Date'].apply(parse_date)
        if df['Date'].isna().any():
            st.warning("Some dates could not be parsed. Please ensure dates are in YYYY-MM-DD format.")
            df = df.dropna(subset=['Date'])
        
        # Generate summaries for long tickets
        df['Summary'] = None
        df['Action Items'] = None
        
        with st.spinner("Generating summaries and extracting action items..."):
            for idx, row in df.iterrows():
                if len(row['Full Text'].split()) > 50:  # Only summarize long tickets
                    df.at[idx, 'Summary'] = summarize_ticket(row['Full Text'])
                    df.at[idx, 'Action Items'] = "\n• " + "\n• ".join(extract_action_items(row['Full Text']))
        
        return df
    return None

def create_topic_model(min_topic_size=5):
    """Create a BERTopic model with optimized parameters."""
    # Initialize UMAP for dimensionality reduction with optimized parameters
    umap_model = UMAP(
        n_neighbors=min(15, min_topic_size),  # Adaptive neighborhood size
        n_components=5,
        min_dist=0.1,  # Increased for better separation
        metric='cosine',
        random_state=42,
        low_memory=True,
        n_jobs=-1  # Use all available cores
    )
    
    # Initialize HDBSCAN with optimized parameters
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=min(5, min_topic_size),  # Adaptive minimum samples
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=-1,  # Use all available cores
        alpha=1.2  # Slightly more conservative clustering
    )
    
    # Create BERTopic model with improved parameters
    topic_model = BERTopic(
        # Using a more powerful multilingual model
        embedding_model="paraphrase-multilingual-mpnet-base-v2",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True,
        # Additional parameters for better topic representation
        top_n_words=15,  # Increase number of words per topic
        min_topic_size=min_topic_size,
        nr_topics="auto"  # Let the model determine optimal number of topics
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

def plot_topic_distribution(df, topic_model):
    """Create a bar chart showing topic distribution with descriptive names."""
    # Get topic information
    topic_info = []
    for topic_num in sorted(set(df['Cluster'].unique()) - {-1}):
        # Get topic frequency
        freq = len(df[df['Cluster'] == topic_num])
        # Get top terms
        top_terms = topic_model.get_topic(topic_num)
        # Get descriptive topic name
        top_words = [word for word, _ in top_terms[:3]]
        topic_name = f"Topic {topic_num}: {' / '.join(top_words)}"
        
        topic_info.append({
            'topic_name': topic_name,
            'frequency': freq
        })
    
    # Sort topics by frequency
    topic_info.sort(key=lambda x: x['frequency'], reverse=True)
    
    # Create topic frequency chart with descriptive names
    freq_data = pd.DataFrame([{
        'Topic': t['topic_name'],
        'Frequency': t['frequency']
    } for t in topic_info])
    
    fig = px.bar(
        freq_data,
        x='Topic',
        y='Frequency',
        title='Topic Distribution Overview',
        color_discrete_sequence=['#1f77b4']  # Use a single color for all bars
    )
    
    fig.update_layout(
        font=dict(
            family="Source Sans Pro, sans-serif",
            color="black"
        ),
        xaxis=dict(
            title=dict(
                text="Topic",
                font=dict(size=14, color="black")
            ),
            tickfont=dict(size=12, color="black"),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            linecolor='black',
            tickangle=-45
        ),
        yaxis=dict(
            title=dict(
                text="Number of Tickets",
                font=dict(size=14, color="black")
            ),
            tickfont=dict(size=12, color="black"),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            linecolor='black'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(b=150),  # Add more bottom margin for rotated labels
        showlegend=False  # Remove the legend
    )
    
    return fig

def display_cluster_summary(df):
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
                    file_name=f"topic_{topic}.csv",
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
                        fig.update_layout(
                            font=dict(
                                family="Source Sans Pro, sans-serif",
                                color="black"
                            ),
                            xaxis=dict(
                                title=dict(
                                    text="Key Terms",
                                    font=dict(size=14, color="black")
                                ),
                                tickfont=dict(size=12, color="black"),
                                tickangle=-45
                            ),
                            yaxis=dict(
                                title=dict(
                                    text="Importance Score",
                                    font=dict(size=14, color="black")
                                ),
                                tickfont=dict(size=12, color="black")
                            )
                        )
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
    
    # Add topic distribution visualization at the bottom
    st.write("---")
    st.write("### Overall Topic Distribution")
    fig = plot_topic_distribution(df, st.session_state.topic_model)
    st.plotly_chart(fig, use_container_width=True)

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

# App title and description
st.title("Support Ticket Clustering App")
st.write("Upload your support ticket CSV files to analyze ticket clusters and trends.")

# Remove month selection and simplify file upload
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type="csv",
    help="File should contain columns: Ticket ID, Subject, Full Text, Date"
)

if uploaded_file:
    # Process the uploaded file
    df = process_uploaded_file(uploaded_file)
    
    if df is not None:
        # Store the processed data
        st.session_state.processed_data = df
        st.success(f"Successfully processed {len(df)} tickets!")
        
        # Display sample of processed data
        st.subheader("Sample of Processed Data")
        st.dataframe(
            df[['Ticket ID', 'Subject', 'Full Text', 'Date']].head(),
            use_container_width=True,
            column_config={
                "Ticket ID": st.column_config.TextColumn("Ticket ID", width="small"),
                "Subject": st.column_config.TextColumn("Subject", width="medium"),
                "Full Text": st.column_config.TextColumn("Full Text", width="large"),
                "Date": st.column_config.DateColumn("Date", width="small")
            }
        )
        
        # Clustering section
        st.subheader("Ticket Clustering")
        
        # Clustering parameters
        st.write("### Clustering Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            min_topic_size = st.slider(
                "Minimum Topic Size",
                min_value=2,
                max_value=20,
                value=3,
                help="Minimum number of tickets in a topic"
            )
        
        with col2:
            time_period = st.selectbox(
                "Trend Analysis Period",
                options=['day', 'week', 'month'],
                index=1,
                help="Time period for trend analysis"
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
                        st.session_state.processed_data = df
                        
                        # Count number of meaningful topics
                        n_topics = len(set([t for t in topics if t != -1]))
                        st.success(f"Successfully identified {n_topics} topics!")
                        
                        # Create tabs for different analyses
                        cluster_tab, trend_tab = st.tabs([
                            "🔍 Cluster Analysis",
                            "📈 Topic Trends"
                        ])
                        
                        with cluster_tab:
                            # Display cluster summary
                            display_cluster_summary(df)
                        
                        with trend_tab:
                            # Calculate and display topic trends
                            topic_trends = analyze_topic_trends(df, time_period)
                            if topic_trends is not None:
                                st.write(f"### Topic Trends by {time_period.title()}")
                                fig = plot_topic_trends(
                                    topic_trends,
                                    f"Topic Distribution Over Time ({time_period.title()})"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show trend statistics
                                st.write("### Trend Statistics")
                                trend_stats = topic_trends.describe()
                                st.dataframe(trend_stats, use_container_width=True)
                            else:
                                st.warning("Could not generate trend analysis. Please check date format in your data.")
                    else:
                        st.error(f"Clustering failed: {error if error else 'Unknown error'}")
                        st.info("Try adjusting the minimum topic size or check your input data.")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please try again with different parameters or check your input data.") 