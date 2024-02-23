import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file into a DataFrame
df = pd.read_csv('resnet.csv')


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
plt.xlabel('Bit')
plt.ylabel('Total Energy')
plt.title('Total Energy vs. Bit for Different Algorithms')

# Display legend
plt.legend()

# Save the plot to a file
plt.savefig('resnetplot.png')

