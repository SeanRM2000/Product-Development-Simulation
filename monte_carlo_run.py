import random
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np

import os

# Classes
from sim_run import PDsim

# Functions
from sim_helper_functions import convert_hours_to_ymd

# Parameters
from Inputs.sim_settings import *
from Inputs.tuning_params import *
from Inputs.sim_inputs import *
    
    
class MonteCarlo():
    def __init__(self):
        
        timestamp = time.time()
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        self.formatted_time = dt_object.strftime("%Y-%m-%d_%H-%M-%S")
        self.folder_name = 'sim_runs/monte_carlo_at_', + self.formatted_time
        os.makedirs(self.folder_name)
        
        self.convergence_data = {
            'Lead Time Average': [],
            'Lead Time Variance': [],
            'Cost Average': [],
            'Cost Variance': []
        }
        
        self.lead_times = []
        self.dev_costs = []
        self.total_efforts = {}
        self.average_utilizations = {}
        
        self.run_montecarlo()

    def run_montecarlo(self):
        stability = [float('inf')] * 4
        start_time = time.time()
        
        sim_runs_left = max_sim_runs
        total_sim_runs = max_sim_runs
        sim_run = 0
        while sim_runs_left != 0:
            sim_run_start_time = time.time()
            sim_run += 1
            
            sim = PDsim(montecarlo=True)
            lead_time, dev_cost, resource_util, total_effort  = sim.sim_run()
            
            # collect data
            self.lead_times.append(lead_time)
            self.dev_costs.append(dev_cost)
            for entry, util in resource_util.items():
                if entry not in self.average_utilizations:
                    self.average_utilizations[entry] = []
                self.average_utilizations[entry].append(util)
            for entry, effort in total_effort.items():
                if entry not in self.total_efforts:
                    self.total_efforts[entry] = []
                self.total_efforts[entry].append(effort) 
            
            # convergence data
            self.convergence_data['Lead Time Average'].append(np.mean(self.lead_times))
            self.convergence_data['Lead Time Variance'].append(np.var(self.lead_times))
            self.convergence_data['Cost Average'].append(np.mean(self.dev_costs))
            self.convergence_data['Cost Variance'].append(np.var(self.dev_costs))
            
            
            # Update Message
            time_passed = time.time() - start_time
            sim_run_time = time.time() - sim_run_start_time
            print(f'[{(time_passed / 60):.1f} min ({sim_run_time:.1f} s)]: Run {sim_run}/{total_sim_runs} completed.')
            
            # stability check
            if check_stability and sim_run >= 2 * stability_interval and sim_run % stability_interval == 0:
                stability = self.check_stability()
                print(f'           Stability Check (Req: <{(stability_criteria * 100)}%): {(stability[0] * 100):.2f}% (Lead Time Mean); {(stability[1] * 100):.2f}% (Lead Time Var); {(stability[2] * 100):.2f}% (Cost Mean); {(stability[3] * 100):.2f}% (Cost Var)')
                if all(stability_value < stability_criteria for stability_value in stability):
                    print(f'Distribution stability reached after {sim_run} runs. Simulation is being stopped.')
                    break
            

            sim_runs_left -= 1
            # after runs ask if continuation is wanted
            if sim_runs_left == 0:
                if check_stability:
                    if any(stability_value > stability_criteria for stability_value in stability):
                        print(f'Max simulation runs ({total_sim_runs}) but stability criteria has not yet been reached.')
                    elif all(stability_value < stability_criteria for stability_value in stability):
                        print(f'Max simulation runs ({total_sim_runs}) and stability criteria has been achieved.')
                else:
                    print('Max simulation runs reached.')
                self.convergence_plot()
                additional_runs = input('Add more runs? ')
                if additional_runs:
                    print(f'{additional_runs} additional runs added.')
                    sim_runs_left += int(additional_runs)
                    total_sim_runs += int(additional_runs)
        
        
        mean_lead_time_ymd = convert_hours_to_ymd(np.mean(self.lead_times))
        
        # Sim results 
        print('\nMonte Carlo Simulation completed.\n')
        
        print('__________________________________________________\n')
        print(f'Average Lead Time: {mean_lead_time_ymd[0]} year(s), {mean_lead_time_ymd[1]} month(s), {mean_lead_time_ymd[2]} day(s)')
        print(f'Average Development Cost: ${round(np.mean(self.dev_costs) / 1000, 1)}k')
        
        if include_noise_in_results:
            print('\nResource Utilizations (including noise) and Effort:')
        else:
            print('\nResource Utilizations and Effort:')
        for entry, utilizations in self.average_utilizations.items():
            if split_plots != 'profession':
                print(f'     {entry}: Utilization: {(np.mean(utilizations) * 100):.1f}%; Effort: {round(np.mean(self.total_efforts[entry]) / work_hours_per_day, 1)} person-days')
            else:
                print(f'     {entry}s: Utilization: {(np.mean(utilizations) * 100):.1f}%; Effort: {round(np.mean(self.total_efforts[entry]) / work_hours_per_day, 1)} person-days')
        
        
        self.convergence_plot()
        self.montecarlo_results_plot()
          

    def check_stability(self):
        previous_lead_times = self.lead_times[:-stability_interval]
        previous_dev_costs = self.dev_costs[:-stability_interval]
        
        prev_lead_times_mean = np.mean(previous_lead_times)
        prev_lead_times_var = np.var(previous_lead_times)
        prev_dev_costs_mean = np.mean(previous_dev_costs)
        prev_dev_costs_var = np.var(previous_dev_costs)
        
        stability = []
        stability.append(abs(self.convergence_data['Lead Time Average'][-1] - prev_lead_times_mean) / prev_lead_times_mean)
        stability.append(abs(self.convergence_data['Lead Time Variance'][-1] - prev_lead_times_var) / prev_lead_times_var)
        stability.append(abs(self.convergence_data['Cost Average'][-1] - prev_dev_costs_mean) / prev_dev_costs_mean)
        stability.append(abs(self.convergence_data['Cost Variance'][-1] - prev_dev_costs_var) / prev_dev_costs_var)
        
        return stability
    
    def montecarlo_results_plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Distribution of Lead Time and Cost', fontsize=16)

        # Lead time in weeks
        lead_times_in_weeks = [lt / (7 * 24) for lt in self.lead_times]
        axes[0].hist(lead_times_in_weeks, bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_title('Lead Time Distribution (Weeks)')
        axes[0].set_xlabel('Lead Time (Weeks)')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True)

        # Cost in thousands
        costs_in_thousands = [c / 1000 for c in self.dev_costs]
        axes[1].hist(costs_in_thousands, bins=20, edgecolor='black', alpha=0.7)
        axes[1].set_title('Cost Distribution (Thousands $)')
        axes[1].set_xlabel('Cost (Thousands $)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.folder_name +  '/monte_carlo_results_' + self.formatted_time + '.png')
        plt.show()
    
    def convergence_plot(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Convergence of Monte Carlo Simulation Metrics', fontsize=16)

        # Convert lead time to weeks and cost to thousands for plotting
        lead_time_avg = [lt / (7 * 24) for lt in self.convergence_data['Lead Time Average']]
        lead_time_var = [lv / ((7 * 24)**2) for lv in self.convergence_data['Lead Time Variance']]
        cost_avg = [c / 1000 for c in self.convergence_data['Cost Average']]
        cost_var = [cv / (1000**2) for cv in self.convergence_data['Cost Variance']]

        metrics = [
            (lead_time_avg, 'Lead Time Average (weeks)', 0, 0),
            (lead_time_var, 'Lead Time Variance (weeks²)', 0, 1),
            (cost_avg, 'Cost Average ($k)', 1, 0),
            (cost_var, 'Cost Variance ($k²)', 1, 1)
        ]

        # Plot each metric in its respective subplot
        for values, title, row, col in metrics:
            axes[row, col].plot(values, label=title)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Simulation Run')
            axes[row, col].set_ylabel(title)
            axes[row, col].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
        plt.savefig(self.folder_name +  '/convergence_' + self.formatted_time + '.png')
        plt.show()


if __name__ == "__main__":
    MonteCarlo()