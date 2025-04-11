import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math
import pandas as pd


doe = 'DOE3'

df = pd.read_excel('DOE/DOE_Results.xlsx', sheet_name=doe)
df = df.dropna(axis=1, how='all')
#print(df)


baseline_values = {
    'Cost': 1395.0,
    'Lead Time': 113.1,
}

mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"


results = {
    'Cost': 'Norm. Cost',
    'Lead Time': 'Norm. LT',
}

limits = [
    (0.65, 1.3),     # Cost
    (0.65, 1.3),     # Lead Time

]

y_ticks_dict = {
    'Cost': [0.8, 1.0, 1.2],
    'Lead Time': [0.8, 1.0, 1.2],
}

additional_filter = None


x_ticks_dict = {
    'DOE3': [0.2, 0.5, 0.8],  # Example for DOE3
}


x_axis = 'Interoperability'
input_1 = 'Accuracy'
input_1_values = [0.9, 0.5, 0.1]
input_2 = 'Usability'
input_2_values = [1, 2, 3]
filter_value = 2
save_file = f'{doe}_Results_Interop.svg'



fig, axes = plt.subplots(1, 2, figsize=(6, 2.2))
axes = axes.flatten()

markers = ['.','.','.']
linestyles = ['dotted', 'dashed', 'solid']






for i, (result_string, label) in enumerate(results.items()):
    ax = axes[i]
    

    mean_col = f"{result_string} Mean"
    lower_col = f"{result_string} 95confidence Lower"
    upper_col = f"{result_string} 95confidence Upper"

    
    
    for j, (level1, level2) in enumerate(zip(input_1_values, input_2_values)):
        
        filtered_df = df[df[input_1] == level1]
        filtered_df = filtered_df[filtered_df[input_2] == level2].sort_values(x_axis)
        x = filtered_df[x_axis]
        
        
        baseline = 1.0
        y = np.array(filtered_df[mean_col]) / baseline_values[result_string]
        lower = np.array(filtered_df[lower_col]) / baseline_values[result_string]
        upper = np.array(filtered_df[upper_col]) / baseline_values[result_string]

        label_legend = f'$T_{{Acc}}={level1}$, $T_{{Usab}}={level2}$'
        
        ax.plot(x, y, label=label_legend, marker=markers[j], markersize=6, fillstyle='full', linestyle=linestyles[j], color='black', linewidth=1.2)
        ax.fill_between(x, lower, upper, alpha=0.2, color='grey')

    
    if i < len(limits) and limits[i]:
        ax.set_ylim(limits[i])
        
    ax.set_ylabel(label, fontsize=12)

    y_ticks = y_ticks_dict.get(result_string)
    if y_ticks:
        ax.set_yticks(y_ticks)
    
    
    ax.set_xlabel(r'Interoperability $T_{I}$', fontsize=12)
        
    x_ticks = x_ticks_dict.get(doe)
    ax.set_xticks(x_ticks)
    

plt.grid(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=11, frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.9])

plt.savefig(save_file, format='svg')

plt.show()
