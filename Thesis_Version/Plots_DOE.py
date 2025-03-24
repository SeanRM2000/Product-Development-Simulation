import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math
import pandas as pd


doe = 'DOE1'
use_value = None

df = pd.read_excel('Architecture/DOE_Results.xlsx', sheet_name=doe)
df = df.dropna(axis=1, how='all')
#print(df)



mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

results = {
    'Cost': 'Cost ($k)',
    'Lead Time': 'Lead Time (wks)',
    'Risk': 'Risk ($k)',
    'Effectivness': r'$\eta_{rework}$',
    'First Pass Yield': 'FPY',
    'Rel Cost Physical': r'$C_{physical}/C_{total}$',
    'Work Efficency': r'$\eta_{value}$',
    'Average Consitency': r'$\overline{I_C}$'
}

additional_filter = None

match doe:
    case 'DOE1':
        x_axis = 'Accuracy Change'
        input_2 = 'Digital Literacy'
        input_2_values = [1.0, 2.0, 3.0]
        label_addition = r'$A_{{DL,Eng}}$'
    
    case 'DOE2':
        x_axis = 'Interoperability'
        input_2 = 'Digital Literacy'
        input_2_values = [1.0, 2.0, 3.0]
        label_addition = r'$A_{{DL,EKM}}$'

    case 'DOE3':
        if False: # filter by interop
            x_axis = 'Accuracy'
            input_2 = 'Usability'
            input_2_values = [1.0, 2.0, 3.0]
            additional_filter = 'Interoperability'
            filter_value = 0.5
            label_addition = r'$T_{{Usab}}$'
        else: # filter by usability
            x_axis = 'Accuracy'
            input_2 = 'Interoperability'
            input_2_values = [0.2, 0.5, 0.8]
            additional_filter = 'Usability'
            filter_value = 1.75
            label_addition = r'$T_I$'

if additional_filter:
    df = df[df[additional_filter] == filter_value]

if use_value:
    input_2_values = [use_value]

fig, axes = plt.subplots(4, 2, figsize=(6, 8))
axes = axes.flatten()

markers = ['o', 's', 'x']
linestyles = ['dotted', 'solid', 'dashed']




for i, (result_string, label) in enumerate(results.items()):
    ax = axes[i]
    
    if True:
        mean_col = f"{result_string} Mean"
        lower_col = f"{result_string} 95confidence Lower"
        upper_col = f"{result_string} 95confidence Upper"
    else:
        mean_col = f"{result_string} Median"
        lower_col = f"{result_string} Q25"
        upper_col = f"{result_string} Q75"
    
    
    
    for j, level in enumerate(input_2_values):
        filtered_df = df[df[input_2] == level].sort_values(x_axis)
        x = filtered_df[x_axis]
        
        if result_string != 'Risk':

            y = filtered_df[mean_col]
            lower = filtered_df[lower_col]
            upper = filtered_df[upper_col]
            
            ax.plot(x, y, label=fr'{label_addition} = {level}', marker=markers[j], fillstyle='none', linestyle=linestyles[j], color='black')
            ax.fill_between(x, lower, upper, alpha=0.2, color='grey')
            
        else:
            y = filtered_df[result_string]
            ax.plot(x, y, label=f'{label_addition} = {level}', marker=markers[j], fillstyle='none', linestyle=linestyles[j], color='black')
    
    if i == 4:
        ax.set_ylim(0, 0.4)
    elif i == 3:
        ax.set_ylim(0, 0.5)
    
    elif i >= 3:
        ax.set_ylim(0, 1)    
    ax.set_ylabel(label)
    
    if i in {6,7}:
        ax.set_xlabel(x_axis)
    


plt.grid(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=11, frameon=False)
plt.tight_layout(rect=[0, 0.03, 1, 1])

plt.savefig(f'{doe}_Results.svg', format='svg')

plt.show()
