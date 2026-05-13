import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# 1. PRE Typographic Configuration
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300
})

# 2. Data Ingestion
try:
    data = np.load('tebd_entropy_data.npy', allow_pickle=True)
    times = np.array(data[0], dtype=float)
    S_vn = np.array(data[1], dtype=float)
    H_inf = np.array(data[2], dtype=float)
except FileNotFoundError:
    raise SystemError("tebd_entropy_data.npy not found in the execution directory.")

# 3. Canvas Initialization
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# 4. Vector Plotting
# Axis 1: Von Neumann Entropy (Left)
color_svn = '#003366' # Deep Navy
line1, = ax1.plot(times, S_vn, color=color_svn, linewidth=2.0, linestyle='-', label=r'Von Neumann Entropy ($S_{vn}$)')
ax1.set_xlabel('Time (Trotter Steps, $t$)', fontweight='bold')
ax1.set_ylabel(r'Entanglement Entropy ($S_{vn}$)', color=color_svn, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_svn)

# Axis 2: Min-Entropy (Right)
color_hinf = '#8B0000' # Crimson Red
line2, = ax2.plot(times, H_inf, color=color_hinf, linewidth=2.0, linestyle='--', label=r'Min-Entropy ($H_\infty$)')
ax2.set_ylabel(r'Min-Entropy ($H_\infty$) [bits]', color=color_hinf, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_hinf)

# 5. Theoretical Bounds & Annotations
terminal_hinf = H_inf[-1]
ax2.axhline(y=terminal_hinf, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
ax2.annotate(f'Terminal Deficit: $H_\\infty \\approx {terminal_hinf:.4f}$', 
             xy=(times[-1], terminal_hinf), 
             xytext=(times[-1] - 3.5, terminal_hinf + 0.05),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=11)

# 6. Grid and Formatting
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax1.grid(True, which='major', linestyle='-', alpha=0.3)
ax1.grid(True, which='minor', linestyle=':', alpha=0.1)

# 7. Unified Legend
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Quantum Scrambling vs. Cryptographic Uniformity\n(Open Quantum Stochastic Walk, $N=16$)', pad=15)
fig.tight_layout()

# 8. Multi-Format Output Export
file_prefix = 'entropy_dynamics_plot'
plt.savefig(f'{file_prefix}.pdf', format='pdf', bbox_inches='tight')
plt.savefig(f'{file_prefix}.png', format='png', dpi=600, bbox_inches='tight')

print(f"Data ingestion complete.")
print(f"Vector plot saved to: {file_prefix}.pdf")
print(f"Raster plot saved to: {file_prefix}.png")
