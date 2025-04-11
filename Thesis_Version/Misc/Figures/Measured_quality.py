import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

# Define parameters

x = np.linspace(0, 1, 1000)

# Norden's Effort Model
accuracy = 1
f1 = 1 - (1 - x) ** (1 / accuracy)

accuracy = 0.8
f2 = 1 - (1 - x) ** (1 / accuracy)

accuracy = 0.5
f3 = 1 - (1 - x) ** (1 / accuracy)

accuracy = 0.2
f4 = 1 - (1 - x) ** (1 / accuracy)


x_ticks = np.linspace(0, 1, 6)  # Evenly spaced between min and max
y_max = math.ceil(max(max(f1), max(f2)) * 1.05)
y_max = 1
y_ticks = np.linspace(0, 1, 6)

# Plot with specified customizations
plt.figure(figsize=(3, 2.8)) ## 6 inches is page width 4, 2.2

# Use Times New Roman for font (fallback if not available)
#mpl.rcParams['text.usetex'] = True
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"#cm

# Plot the curves in black
plt.plot(x, f1, label="1.0", linestyle='solid', linewidth=1, color='black')
plt.plot(x, f2, label="0.8", linestyle='dashed', linewidth=1, color='black')
plt.plot(x, f3, label="0.5", linestyle='dotted', linewidth=1, color='black')
plt.plot(x, f4, label="0.2", linestyle='dashdot', linewidth=1, color='black')

# Set labels with font size 12
plt.xlabel("Actual Quality $Q$", fontsize=12)
plt.ylabel("Measured Quality $\hat{Q}$", fontsize=12)

# Set x and y axis limits to start at zero
plt.xlim(0, 1)
plt.ylim(0, 1)  # Adjust slightly above max for visibility

# Customize ticks font size
plt.xticks(x_ticks, fontsize=11)
plt.yticks(y_ticks, fontsize=11)

plt.legend(frameon=False, fontsize=10, title='Accuracy')

# Remove grid
plt.grid(False)

plt.tight_layout(pad=0)

# Customize legend with font size 12
#plt.legend(fontsize=12)
plt.savefig('Accuracy_Measured_Quality.svg', format='svg')

# Show the plot
plt.show()
