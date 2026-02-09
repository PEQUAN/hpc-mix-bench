import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results.csv")

# Define markers and colors for different matrix sizes
matrix_sizes = sorted(df['MatrixSize'].unique())
markers = ['o', 's', '^']  # circle, square, triangle
colors = ['black', 'purple', 'darkviolet']  # high-contrast

# Font sizes
title_size = 18
label_size = 14
tick_size = 14

# Set plot style
plt.style.use('bmh')
plt.rcParams['lines.linewidth'] = 2  # thicker lines
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# Plot 1: Exponent vs AvgTime
plt.figure(figsize=(8, 6))
exp_df = df[df['Type'] == 'exp']

for i, size in enumerate(matrix_sizes):
    subset = exp_df[exp_df['MatrixSize'] == size]
    plt.plot(subset['ExpBits'], subset['AvgTime'],
             label=f"{size}x{size}",
             marker=markers[i],
             color=colors[i],
             linewidth=2,
             markersize=8)

# plt.title('Matrix Multiplication Time vs Exponent Bits', fontsize=title_size)
plt.xlabel('Exponent Bits', fontsize=label_size)
plt.ylabel('Average Time (s)', fontsize=label_size)
plt.xticks(np.arange(exp_df['ExpBits'].min(), exp_df['ExpBits'].max()+1, 2), fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.legend(title='Matrix Size', fontsize=label_size, title_fontsize=label_size)
plt.grid(True)
plt.tight_layout()
plt.savefig("exponent_vs_time.png", bbox_inches='tight')
plt.show()

# Plot 2: Significand vs AvgTime
plt.figure(figsize=(8, 6))
sig_df = df[df['Type'] == 'sig']

for i, size in enumerate(matrix_sizes):
    subset = sig_df[sig_df['MatrixSize'] == size]
    plt.plot(subset['SigBits'], subset['AvgTime'],
             label=f"{size}x{size}",
             marker=markers[i],
             color=colors[i],
             linewidth=2,
             markersize=8)

# plt.title('Matrix Multiplication Time vs Significand Bits', fontsize=title_size)
plt.xlabel('Significand Bits', fontsize=label_size)
plt.ylabel('Average Time (s)', fontsize=label_size)
plt.xticks(np.arange(sig_df['SigBits'].min(), sig_df['SigBits'].max()+1, 2), fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.legend(title='Matrix Size', fontsize=label_size, title_fontsize=label_size)
plt.grid(True)
plt.tight_layout()
plt.savefig("significand_vs_time.png", bbox_inches='tight')
plt.show()
