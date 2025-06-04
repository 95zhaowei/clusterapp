import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read the CSV file
df = pd.read_csv('sample_tickets.csv')

# Set random seed for reproducibility
np.random.seed(42)

# Generate random dates between Jan 1, 2025 and Feb 28, 2025
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 2, 28)
days_between = (end_date - start_date).days

random_dates = [start_date + timedelta(days=np.random.randint(0, days_between)) for _ in range(len(df))]
random_dates.sort()  # Sort dates to make it more realistic

# Add dates to the dataframe
df['Date'] = [date.strftime('%Y-%m-%d') for date in random_dates]

# Save the updated CSV
df.to_csv('sample_tickets.csv', index=False)
print("Added random dates to sample_tickets.csv") 