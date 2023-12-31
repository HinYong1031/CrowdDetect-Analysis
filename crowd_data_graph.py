import pandas as pd
import matplotlib.pyplot as plt
from config import NEW_DESIRED_FRAME

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('processed_data/temp.csv')

# Calculate time in minutes
df['Time'] = df['Time'] / (NEW_DESIRED_FRAME * 60) # Convert to minutes

# Create the plot
plt.plot(df['Time'], df['Human Count'])

# Add labels and title
plt.xlabel('Time (minutes)')
plt.ylabel('Human Count')
plt.title('Human Count Over Time')

# Customize x-axis ticks
tick_interval = 1
max_time = df['Time'].max()
plt.xticks(range(0, int(max_time) + tick_interval, tick_interval))

# Show the plot
plt.show()
