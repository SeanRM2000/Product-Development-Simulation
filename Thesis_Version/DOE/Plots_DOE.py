import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math
import pandas as pd


doe = 'DOE3'
use_value = None

df = pd.read_excel('DOE/DOE_Results.xlsx', sheet_name=doe)
df = df.dropna(axis=1, how='all')
#print(df)


baseline_values = {
    'Cost': 1395.0,
    'Lead Time': 113.1,
    'Risk': 18763.4,
    'Quality': 0.804,
    'Effectivness': 0.275,
    'Average Iterations': 7.5,
    'First Pass Yield': 0.075,
    'Rel Cost Physical': 0.72,
    'Work Efficency': 0.49,
    'Average Consitency': 0.66
}

mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

results = {
    'Cost': 'Norm. Cost',
    'Lead Time': 'Norm. LT',
    'Risk': 'Norm. Risk',
    'Quality': 'Quality',
    'Effectivness': r'$\eta_{rework}$',
    'Average Iterations': r'$\overline{n_{iter}}$',
    'First Pass Yield': 'FPY',
    'Rel Cost Physical': r'$C_{physical}$',
    'Work Efficency': r'$\eta_{value}$',
    'Average Consitency': r'$\overline{I_C}$'
}

limits = [
    (0.68, 1.6),     # Cost
    (0.7, 1.6),     # Lead Time
    (0.0, 9),     # Risk
    (0.7, 0.9),     # Quality
    (0.2, 0.42),     # Effectiveness
    (5.0, 11.0),        # Average Iterations
    (0.05, 0.35),     # First Pass Yield
    (0.45, 0.8),     # Rel Cost Physical
    (0.25, 0.65),     # Work Efficiency
    (0.43, 0.83)      # Average Consistency
]

y_ticks_dict = {
    'Cost': [0.7, 1.0, 1.3, 1.6],
    'Lead Time': [0.7, 1.0, 1.3, 1.6],
    'Risk': [0.0, 3.0, 6.0, 9.0],
    'Quality': [0.7, 0.8, 0.9],
    'Effectivness': [0.2, 0.3, 0.4],
    'Average Iterations': [5.0, 7.0, 9.0, 11.0],
    'First Pass Yield': [0.1, 0.2, 0.3],
    'Rel Cost Physical': [0.5, 0.6, 0.7, 0.8],
    'Work Efficency': [0.3, 0.4, 0.5, 0.6],
    'Average Consitency': [0.5, 0.6, 0.7, 0.8],
}

additional_filter = None


x_ticks_dict = {
    'DOE1': [-0.8, -0.4, 0.0, 0.4, 0.8],  # Example for DOE1
    'DOE2': [0, 0.25, 0.5, 0.75, 1.0],  # Example for DOE2
    'DOE3': [0.1, 0.3, 0.5, 0.7, 0.9],  # Example for DOE3
}


match doe:
    case 'DOE1':
        x_axis = 'Accuracy Change'
        x_label = 'Accuracy Change'
        input_2 = 'Digital Literacy'
        input_2_values = [1.0, 2.0, 3.0]
        label_addition = r'$A_{{DL,Eng}}$'
        save_file = f'{doe}_Results.svg'
    
    case 'DOE2':
        x_axis = 'Interoperability'
        x_label = r'Interoperability $T_I$'
        input_2 = 'Digital Literacy'
        input_2_values = [1.0, 2.0, 3.0]
        label_addition = r'$A_{{DL,EKM}}$'
        save_file = f'{doe}_Results.svg'

    case 'DOE3':
        if False: # filter by interop
            x_axis = 'Accuracy'
            x_label = r'Accuracy $T_{Acc}$'
            input_2 = 'Usability'
            input_2_values = [1.0, 2.0, 3.0]
            additional_filter = 'Interoperability'
            filter_value = 0.8
            label_addition = r'$T_{{Usab}}$'
            save_file = f'{doe}_Results_Interop_{filter_value}.svg'
            
            
        else: # filter by usability
            x_axis = 'Accuracy'
            x_label = r'Accuracy $T_{Acc}$'
            input_2 = 'Interoperability'
            input_2_values = [0.2, 0.5, 0.8]
            additional_filter = 'Usability'
            filter_value = 2
            label_addition = r'$T_I$'
            save_file = f'{doe}_Results_Usab_{filter_value}.svg'

if additional_filter:
    df = df[df[additional_filter] == filter_value]

if use_value:
    input_2_values = [use_value]

fig, axes = plt.subplots(5, 2, figsize=(6, 8.5))
axes = axes.flatten()

markers = ['o', 's', 'x']
markers = ['.','.','.']
linestyles = ['dotted', 'dashed', 'solid']




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
            
            if result_string == 'Cost':
                baseline = 1.0
                y = np.array(filtered_df[mean_col]) / baseline_values['Cost']
                lower = np.array(filtered_df[lower_col]) / baseline_values['Cost']
                upper = np.array(filtered_df[upper_col]) / baseline_values['Cost']
                
            elif result_string == 'Lead Time':
                baseline = 1.0
                y = np.array(filtered_df[mean_col]) / baseline_values['Lead Time']
                lower = np.array(filtered_df[lower_col]) / baseline_values['Lead Time']
                upper = np.array(filtered_df[upper_col]) / baseline_values['Lead Time']

            else:
                baseline = baseline_values[result_string]
                y = filtered_df[mean_col]
                lower = filtered_df[lower_col]
                upper = filtered_df[upper_col]
                baseline_tick = baseline_values[result_string]
            
            ax.plot(x, y, label=fr'{label_addition} = {level}', marker=markers[j], markersize=6, fillstyle='full', linestyle=linestyles[j], color='black', linewidth=1.2)
            ax.fill_between(x, lower, upper, alpha=0.2, color='grey')
            
        else:
            baseline = 1.0
            y = np.array(filtered_df[result_string]) / baseline_values['Risk']
            ax.plot(x, y, label=f'{label_addition} = {level}', marker=markers[j], markersize=6, fillstyle='full', linestyle=linestyles[j], color='black', linewidth=1.2)
            
    ax.axhline(baseline, color='grey', linestyle='--', linewidth=0.8)
    ax.text(1.02, baseline, 'B', color='black', fontsize=8, ha='left', va='center', transform=ax.get_yaxis_transform())
    
    
    if i < len(limits) and limits[i]:
        ax.set_ylim(limits[i])
        
    ax.set_ylabel(label, fontsize=12)
    #if result_string == 'Lead Time':
    #    ax.yaxis.set_label_coords(-0.15, 0.4)
    if result_string == 'Risk':
        ax.yaxis.set_label_coords(-0.15, 0.5)
    
    y_ticks = y_ticks_dict.get(result_string)
    if y_ticks:
        ax.set_yticks(y_ticks)
    
    
    if i in {8,9}:
        ax.set_xlabel(x_label, fontsize=12)
        
    x_ticks = x_ticks_dict.get(doe)
    ax.set_xticks(x_ticks)
    
    
    subplot_labels = list('abcdefghijklmnopqrstuvwxyz')

for i, ax in enumerate(axes):
    ax_label = f"({subplot_labels[i]})"
    ax.text(-0.15, 1, ax_label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='center', ha='right')


plt.grid(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=11, frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(save_file, format='svg')

plt.show()
