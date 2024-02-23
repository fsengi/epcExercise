import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file into a DataFrame
df = pd.read_csv('convnet.csv')

# Set the global font size
plt.rcParams.update({'font.size': 14})

# Set the size of the plot
plt.figure(figsize=(20, 12))

# Separate data based on 'Approx. Algo'
algorithms = df['Approx. Algo'].unique()


# Plot each algorithm separately
for algo in algorithms:
    algo_data = df[df['Approx. Algo'] == algo]
    plt.scatter(algo_data['Bit'], algo_data['Total Energy'], label=algo, s=12)
    plt.plot(algo_data['Bit'], algo_data['Total Energy'], label=algo, linewidth=0.5)

# Set labels and title
plt.xlabel('Number of approximate bits')
plt.ylabel('Total Energy[pJ]')
plt.title('Total energy for varying number of approximate bits for multiple algorithms')

# Display legend
plt.legend()

# Save the plot to a file
plt.savefig('convnet.png')

