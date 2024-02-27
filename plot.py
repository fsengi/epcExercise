import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file into a DataFrame
df = pd.read_csv('convnet.csv')

# Set the global font size
plt.rcParams.update({'font.size': 12})

# Set the size of the plot
plt.figure(figsize=(8, 5))

# Separate data based on 'Approx. Algo'
algorithms = df['Approx. Algo'].unique()

# Plot each algorithm separately
for algo in algorithms:
    algo_data = df[df['Approx. Algo'] == algo]
    #plt.scatter(algo_data['Bit'], algo_data['Total Energy'], s=50, alpha=0.7)  # dots at data points
    line, = plt.plot(algo_data['Bit'], algo_data['Total Energy'], label=algo, linewidth=1, linestyle='dashed', marker='o')

# Set labels and title
plt.xlabel('Number of approximate bits')
plt.ylabel('Total Energy [pJ]')
plt.title('Total energy for varying number of approximate bits for multiple algorithms')

# Create a separate legend for the lines
handles, labels = plt.gca().get_legend_handles_labels()
line_legend = plt.legend(handles, labels, loc='lower left')

# Save the plot to a file
plt.savefig('convnet_new.png')
