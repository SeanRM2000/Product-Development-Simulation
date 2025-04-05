import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

# Define parameters

x = np.linspace(0, 20, 1000)

# Norden's Effort Model
f1 = []
f2 = []
f3 = []
for i in x:
    complexity = 1
    y = 1 / (1 + ( (1 / 0.25) - 1) * np.exp(-1.2 * i / complexity) )
    f1.append(y)
    complexity = 2
    y = 1 / (1 + ( (1 / 0.25) - 1) * np.exp(-1.2 * i / complexity) )
    f2.append(y)
    complexity = 3
    y = 1 / (1 + ( (1 / 0.25) - 1) * np.exp(-1.2 * i / complexity) )
    f3.append(y)

x_ticks = np.linspace(0, 20, 5)  # Evenly spaced between min and max
y_ticks = np.linspace(0.25, 1, 4)

# Plot with specified customizations
plt.figure(figsize=(4, 2.4)) ## 6 inches is page width 4, 2.2

# Use Times New Roman for font (fallback if not available)
#mpl.rcParams['text.usetex'] = True
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"#cm

# Plot the curves in black
plt.plot(x, f1, label=r'$IC=1$', linestyle='solid', linewidth=1, color='black')
plt.plot(x, f2, label=r'$IC=2$', linestyle='dashed', linewidth=1, color='black')
plt.plot(x, f3, label=r'$IC=3$', linestyle='dotted', linewidth=1, color='black')

# Set labels with font size 12
plt.xlabel(r"Collaboration Effort $E_{collab}$", fontsize=12)
plt.ylabel(r"New Knowledge Level $PK^{New}$", fontsize=12)

# Set x and y axis limits to start at zero
plt.xlim(0, 20)
plt.ylim(0.25, 1)  # Adjust slightly above max for visibility

# Customize ticks font size
plt.xticks(x_ticks, fontsize=11)
plt.yticks(y_ticks, fontsize=11)

# Remove grid
plt.grid(False)

plt.legend(frameon=False, fontsize=10, loc='lower right')


plt.tight_layout(pad=0)

# Customize legend with font size 12
#plt.legend(fontsize=12)
plt.savefig('Collaboration_Effect.svg', format='svg')

# Show the plot
plt.show()
