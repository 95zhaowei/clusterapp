# Support Ticket Clustering App

A Streamlit application that helps analyze and cluster support tickets to identify common patterns and issues. The app uses machine learning techniques to automatically group similar tickets together, making it easier to identify trends and recurring problems.

## Features

- Upload CSV files containing support ticket data
- Automatic text preprocessing and cleaning
- Multiple clustering algorithms (K-means, DBSCAN, Hierarchical)
- Interactive cluster visualization
- Automatic cluster labeling
- Month-to-month comparison capabilities
- Search and filtering functionality
- Export cluster data

## Requirements

- Python 3.7+
- Streamlit
- scikit-learn
- pandas
- numpy
- nltk
- plotly

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/clusterapp.git
cd clusterapp
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Select a month for the data you want to analyze
2. Upload a CSV file containing support ticket data with the following columns:
   - Ticket ID
   - Subject
   - Full Text
3. Configure clustering parameters:
   - Choose vectorization method (TF-IDF or Count)
   - Select clustering algorithm
   - Adjust algorithm-specific parameters
4. Click "Perform Clustering" to analyze the data
5. View results and explore clusters
6. Compare data across different months

## Data Format

The input CSV file should contain the following columns:
- `Ticket ID`: Unique identifier for each ticket
- `Subject`: Short description or title of the ticket
- `Full Text`: Complete ticket description or content

## License

MIT License 
