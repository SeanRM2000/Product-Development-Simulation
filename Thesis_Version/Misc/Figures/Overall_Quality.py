import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

# Define parameters

x = np.linspace(0, 1, 1000)

# Norden's Effort Model
f1 = []
for i in x:
    f1.append(1 / (1 + 50 * math.exp(-10 * i)))


x_ticks = np.linspace(0, 1, 6)  # Evenly spaced between min and max
y_ticks = np.linspace(0, 1, 6)

# Plot with specified customizations
plt.figure(figsize=(4, 2.4)) ## 6 inches is page width 4, 2.2

# Use Times New Roman for font (fallback if not available)
#mpl.rcParams['text.usetex'] = True
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"#cm

# Plot the curves in black
plt.plot(x, f1, label=r'$\overline{Q_{G,sub}}=1.0$', linestyle='solid', linewidth=1, color='black')
plt.plot(x, np.array(f1) * 0.67, label=r'$\overline{Q_{G,sub}}=0.67$', linestyle='dashed', linewidth=1, color='black')
plt.plot(x, np.array(f1) * 0.33, label=r'$\overline{Q_{G,sub}}=0.33$', linestyle='dotted', linewidth=1, color='black')

# Set labels with font size 12
plt.xlabel(r"Average Interface Quality $\overline{Q_I}$", fontsize=12)
plt.ylabel(r"Solution Goodness $Q_G$", fontsize=12)

# Set x and y axis limits to start at zero
plt.xlim(0, 1)
plt.ylim(0, 1)  # Adjust slightly above max for visibility

# Customize ticks font size
plt.xticks(x_ticks, fontsize=11)
plt.yticks(y_ticks, fontsize=11)

# Remove grid
plt.grid(False)

plt.legend(frameon=False, fontsize=10)


plt.tight_layout(pad=0)

# Customize legend with font size 12
#plt.legend(fontsize=12)
plt.savefig('Solution_Goodness.svg', format='svg')

# Show the plot
plt.show()
