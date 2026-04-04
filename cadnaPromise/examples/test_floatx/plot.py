import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================
# Load data
# ============================
df = pd.read_csv("results.csv")

# ============================
# Global plotting style（论文风格）
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

    'lines.linewidth': 2,
    'lines.markersize': 6,

    'axes.facecolor': 'white',
    'figure.facecolor': 'white',

    'legend.frameon': False,
    'axes.grid': True
})

# ============================
# Plot 1: Varying exponent bits
# ============================
fig, ax = plt.subplots(figsize=(8, 6))

exp_df = df[(df['Type'] == 'exp') & (df['MatrixSize'] == 500)]

fixed_sig_bits = sorted(exp_df['SigBits'].unique())
markers = ['o', 's', '^', 'D']
colors = ['black', 'purple', 'teal', 'darkorange']

for i, sig in enumerate(fixed_sig_bits):
    subset = exp_df[exp_df['SigBits'] == sig].sort_values('ExpBits')

    ax.plot(
        subset['ExpBits'],
        subset['AvgTime'],
        marker=markers[i % len(markers)],
        color=colors[i % len(colors)],
        label=f'Significand = {sig}'
    )

ax.set_xlabel('Exponent Bits')
ax.set_ylabel('Average Time (s)')

ax.set_xticks(
    np.arange(exp_df['ExpBits'].min(), exp_df['ExpBits'].max() + 1, 2)
)

ax.legend()

plt.tight_layout()
plt.savefig("exponent_vs_time.png", dpi=300, bbox_inches='tight')
plt.show()

# ============================
# Plot 2: Varying significand bits
# ============================
fig, ax = plt.subplots(figsize=(8, 6))

sig_df = df[(df['Type'] == 'sig') & (df['MatrixSize'] == 500)]

fixed_exp_bits = sorted(sig_df['ExpBits'].unique())

for i, exp in enumerate(fixed_exp_bits):
    subset = sig_df[sig_df['ExpBits'] == exp].sort_values('SigBits')

    ax.plot(
        subset['SigBits'],
        subset['AvgTime'],
        marker=markers[i % len(markers)],
        color=colors[i % len(colors)],
        label=f'Exponent = {exp}'
    )

ax.set_xlabel('Significand Bits')
ax.set_ylabel('Average Time (s)')

ax.set_xticks(
    np.arange(sig_df['SigBits'].min(), sig_df['SigBits'].max() + 1, 2)
)

ax.legend()

plt.tight_layout()
plt.savefig("significand_vs_time.png", dpi=300, bbox_inches='tight')
plt.show()