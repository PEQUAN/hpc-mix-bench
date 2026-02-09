import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("results.csv")

# Define markers and colors for different matrix sizes
matrix_sizes = sorted(df['MatrixSize'].unique())
markers = ['o', 's', '^']  # circle, square, triangle
palette = sns.color_palette("bright", n_colors=len(matrix_sizes))  # vivid colors

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['lines.linewidth'] = 2  # thicker lines

# Plot 1: Exponent vs AvgTime
plt.figure()
exp_df = df[df['Type'] == 'exp']

for i, size in enumerate(matrix_sizes):
    subset = exp_df[exp_df['MatrixSize'] == size]
    plt.plot(subset['ExpBits'], subset['AvgTime'],
             label=f"{size}x{size}",
             marker=markers[i],
             color=palette[i],
             linewidth=2,
             markersize=8)

plt.title('Matrix Multiplication Time vs Exponent Bits')
plt.xlabel('Exponent Bits')
plt.ylabel('Average Time (s)')
plt.xticks(np.arange(exp_df['ExpBits'].min(), exp_df['ExpBits'].max()+1, 2))  # integer ticks
plt.legend(title='Matrix Size')
plt.grid(True)
plt.tight_layout()
plt.savefig("exponent_vs_time.png", dpi=300)
plt.show()

# Plot 2: Significand vs AvgTime
plt.figure()
sig_df = df[df['Type'] == 'sig']

for i, size in enumerate(matrix_sizes):
    subset = sig_df[sig_df['MatrixSize'] == size]
    plt.plot(subset['SigBits'], subset['AvgTime'],
             label=f"{size}x{size}",
             marker=markers[i],
             color=palette[i],
             linewidth=2,
             markersize=8)

plt.title('Matrix Multiplication Time vs Significand Bits')
plt.xlabel('Significand Bits')
plt.ylabel('Average Time (s)')
plt.xticks(np.arange(sig_df['SigBits'].min(), sig_df['SigBits'].max()+1, 2))  # integer ticks
plt.legend(title='Matrix Size')
plt.grid(True)
plt.tight_layout()
plt.savefig("significand_vs_time.png", dpi=300)
plt.show()
