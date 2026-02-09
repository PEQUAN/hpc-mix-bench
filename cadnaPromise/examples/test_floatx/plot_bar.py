import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Read raw data
# -----------------------------
df = pd.read_csv("results_bar.csv")

# Precision mapping (for plotting only)
precision_map = {
    'c': 'E4M3',
    'w': 'E5M2',
    'b': 'BF16',
    'p': 'FP16',
    's': 'FP32',
    'd': 'FP64'
}
df['PrecisionLabel'] = df['Precision'].map(precision_map)

# Orders
matrix_sizes = sorted(df['MatrixSize'].unique())
precisions = ['E4M3', 'E5M2', 'BF16', 'FP16', 'FP32', 'FP64']

# -----------------------------
# Pivot: rows = MatrixSize, cols = Precision
# -----------------------------
pivot = df.pivot(
    index='MatrixSize',
    columns='PrecisionLabel',
    values='AvgTime'
)[precisions]

# -----------------------------
# Plot style (BMH + white background)
# -----------------------------
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (7, 6)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# -----------------------------
# Grouped bar chart
# -----------------------------
x = np.arange(len(matrix_sizes))
bar_width = 0.1

colors = ['gray', 'yellowgreen', 'tomato', 'pink', 'deepskyblue', 'steelblue']  # high-contrast colors

plt.figure(figsize=(7, 6))

for i, prec in enumerate(precisions):
    plt.bar(
        x + i * bar_width,
        # np.log(pivot[prec].values),
        pivot[prec].values,
        width=bar_width,
        label=prec,
        color=colors[i]
    )

plt.xticks(x + bar_width * 1.5, matrix_sizes)
plt.xlabel("Matrix Size")
plt.ylabel("Average Time (s)")
# plt.title("Performance Comparison of Different Precision Formats")
plt.legend(title="Precision", fontsize=13, title_fontsize=13)
plt.tight_layout()
plt.savefig("precision_grouped_by_size.png", dpi=300, bbox_inches='tight')
plt.show()
