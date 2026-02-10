import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results.csv")

# Font sizes
label_size = 14
tick_size = 14

# Plot style
plt.style.use('bmh')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# ============================
# Plot 1: Varying exponent bits
# ============================
plt.figure(figsize=(8, 6))

exp_df = df[(df['Type'] == 'exp') & (df['MatrixSize'] == 500)]

fixed_sig_bits = sorted(exp_df['SigBits'].unique())
markers = ['o', 's']
colors = ['black', 'purple']

for i, sig in enumerate(fixed_sig_bits):
    subset = exp_df[exp_df['SigBits'] == sig].sort_values('ExpBits')

    plt.plot(
        subset['ExpBits'],
        subset['AvgTime'],
        marker=markers[i],
        color=colors[i],
        markersize=8,
        label=f'Fixed Significand = {sig}'
    )

plt.xlabel('Exponent Bits', fontsize=label_size)
plt.ylabel('Average Time (s)', fontsize=label_size)
plt.xticks(
    np.arange(exp_df['ExpBits'].min(), exp_df['ExpBits'].max() + 1, 2),
    fontsize=tick_size
)
plt.yticks(fontsize=tick_size)
plt.legend(fontsize=label_size)
plt.grid(True)
plt.tight_layout()
plt.savefig("exponent_vs_time.png", dpi=300, bbox_inches='tight')
plt.show()

# ============================
# Plot 2: Varying significand bits
# ============================
plt.figure(figsize=(8, 6))

sig_df = df[(df['Type'] == 'sig') & (df['MatrixSize'] == 500)]

fixed_exp_bits = sorted(sig_df['ExpBits'].unique())
markers = ['o', 's']
colors = ['black', 'purple']

for i, exp in enumerate(fixed_exp_bits):
    subset = sig_df[sig_df['ExpBits'] == exp].sort_values('SigBits')

    plt.plot(
        subset['SigBits'],
        subset['AvgTime'],
        marker=markers[i],
        color=colors[i],
        markersize=8,
        label=f'Fixed Exponent = {exp}'
    )

plt.xlabel('Significand Bits', fontsize=label_size)
plt.ylabel('Average Time (s)', fontsize=label_size)
plt.xticks(
    np.arange(sig_df['SigBits'].min(), sig_df['SigBits'].max() + 1, 2),
    fontsize=tick_size
)
plt.yticks(fontsize=tick_size)
plt.legend(fontsize=label_size)
plt.grid(True)
plt.tight_layout()
plt.savefig("significand_vs_time.png", dpi=300, bbox_inches='tight')
plt.show()
