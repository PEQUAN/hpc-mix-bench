import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================
# Load data
# ============================
df = pd.read_csv("results_bar.csv")

# ============================
# Precision mapping
# ============================
precision_map = {
    'c': 'E4M3',
    'w': 'E5M2',
    'b': 'BF16',
    'p': 'FP16',
    's': 'FP32',
    'd': 'FP64'
}
df['PrecisionLabel'] = df['Precision'].map(precision_map)

matrix_sizes = sorted(df['MatrixSize'].unique())
precisions = ['E4M3', 'E5M2', 'BF16', 'FP16', 'FP32', 'FP64']

# ============================
# Pivot table
# ============================
pivot = df.pivot(
    index='MatrixSize',
    columns='PrecisionLabel',
    values='AvgTime'
)[precisions]

# ============================
# Global plotting style
# ============================
font_size = 16

plt.style.use('bmh')

plt.rcParams.update({
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size,

    'axes.facecolor': 'white',
    'figure.facecolor': 'white',

    'legend.frameon': False,
    'axes.grid': True
})

# ============================
# Grouped bar chart
# ============================
x = np.arange(len(matrix_sizes))
bar_width = 0.12

colors = ['gray', 'yellowgreen', 'tomato', 'pink', 'deepskyblue', 'steelblue']

fig, ax = plt.subplots(figsize=(7, 6))

for i, prec in enumerate(precisions):
    ax.bar(
        x + i * bar_width,
        pivot[prec].values,
        width=bar_width,
        label=prec,
        color=colors[i]
    )

ax.set_xticks(x + bar_width * (len(precisions) - 1) / 2)
ax.set_xticklabels(matrix_sizes)

ax.set_xlabel("Matrix Size")
ax.set_ylabel("Average Time (s)")
ax.legend(title="Precision", title_fontsize=font_size)

plt.tight_layout()
plt.savefig("precision_grouped_by_size.png", dpi=300, bbox_inches='tight')
plt.show()