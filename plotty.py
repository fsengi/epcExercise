import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('testout1.csv')

# Set the figure size
fig, ax1 = plt.subplots(figsize=(20, 12))

# Plot Total Energy on the first y-axis
color = 'tab:red'
ax1.set_xlabel('Bit')
ax1.set_ylabel('Total Energy', color=color)
ax1.plot(df['Bit'], df['Total Energy'], color=color, label='Total Energy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.yaxis.label.set_size(14)  # Set font size for the y-axis label

# Create a second y-axis for Steps
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Steps', color=color)
ax2.plot(df['Bit'], df['Steps'], color=color, label='Steps')
ax2.tick_params(axis='y', labelcolor=color)
ax2.yaxis.label.set_size(14)  # Set font size for the y-axis label

# Set font size for x-axis label
ax1.xaxis.label.set_size(14)

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show the plot
plt.title('Total Energy and Steps vs. Bit')

# Save the plot to a file
plt.savefig('testplot.png')

# Display the plot
plt.show()
