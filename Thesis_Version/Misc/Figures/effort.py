import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

# Define parameters
alpha = 0.03
scale = 60
time = np.linspace(0, 15, 1000)  # Time range

# Norden's Effort Model
nordens_effort = 2 * scale * (alpha * time * np.exp(-alpha * time**2))


alpha2 = 0.3
scale2 = 60

# Compute total effort for original Exponential Model (before scaling)
exp_distribution = scale2 * alpha2 * np.exp(-alpha2 * time)

x_ticks = np.linspace(0, 15, 4)  # Evenly spaced between min and max
y_max = math.ceil(max(max(nordens_effort), max(exp_distribution)) * 1.05)
y_ticks = np.linspace(0, 20, 5)

# Plot with specified customizations
plt.figure(figsize=(4, 2.2)) ## 6 inches is page width

# Use Times New Roman for font (fallback if not available)
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "Times New Roman"

# Plot the curves in black
plt.plot(time, nordens_effort, label="Norden's Model", linestyle='-', linewidth=1, color='black')
plt.plot(time, exp_distribution, label="Baseline Model", linestyle='--', linewidth=1, color='black')

# Set labels with font size 12
plt.xlabel("Time (months)", fontsize=12)
plt.ylabel("Effort (person-months)", fontsize=12)

# Set x and y axis limits to start at zero
plt.xlim(0, 15)
plt.ylim(0, 20)  # Adjust slightly above max for visibility

# Customize ticks font size
plt.xticks(x_ticks, fontsize=11)
plt.yticks(y_ticks, fontsize=11)

plt.legend(frameon=False, fontsize=10)

# Remove grid
plt.grid(False)

plt.tight_layout(pad=0)

# Customize legend with font size 12
#plt.legend(fontsize=12)
plt.savefig('NordenEffort.svg', format='svg')

# Show the plot
plt.show()
