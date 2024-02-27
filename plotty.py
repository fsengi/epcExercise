import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('testout1.csv')

# Set the figure size
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Total Energy on the first y-axis
color = 'tab:red'
ax1.set_xlabel('Number of approximate bits')
ax1.set_ylabel('Total Energy [pJ]', color=color)
ax1.plot(df['Bit'], df['Total Energy'], color=color, label='Total Energy[pJ]',linewidth=1, linestyle='dashed', marker='o')
ax1.tick_params(axis='y', labelcolor=color)
ax1.yaxis.label.set_size(12)  # Set font size for the y-axis label

# Create a second y-axis for Steps
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Steps [count]', color=color)
ax2.plot(df['Bit'], df['Steps'], color=color, label='Steps[count]',linewidth=1, linestyle='dashed', marker='o')
ax2.tick_params(axis='y', labelcolor=color)
ax2.yaxis.label.set_size(12)  # Set font size for the y-axis label

# Set font size for x-axis label
ax1.xaxis.label.set_size(12)

# # Add legends
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')

# Show the plot
plt.title('Total energy and compute step count for varying number of approximate bits', fontsize=12)

# Save the plot to a file
plt.savefig('stepsplot.png')


