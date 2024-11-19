import random
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
import pandas as pd
import os
import json

# Classes
from sim_run import PDsim

# Functions
from sim_helper_functions import convert_hours_to_ymd, clear_folder

# Parameters
from Inputs.sim_settings import *
from Inputs.tuning_params import *
from Inputs.sim_inputs import *
    
    
class MonteCarlo():
    def __init__(self, max_sim_runs, 
                 check_stability=False, stability_criteria=0.01, stability_interval=100, # stability
                 inital_seed=None, use_seeds=False, # monte carlo repeatability and trackability
                 architecture_config_name=None, # is provided if simulation is run based on architecture model inputs
                 ):
        
        # sim settings
        self.max_sim_runs = max_sim_runs
        self.check_stability = check_stability
        self.stability_criteria = stability_criteria
        self.stability_interval =  stability_interval
        
        # targets
        file_path = ('Architecture/Inputs/goals.json') if architecture_config_name else 'Inputs/test_data/test_goals.json'
        with open(file_path, 'r') as file:
            goal_data = json.load(file)
        self.overall_quality_goal = goal_data['overall_quality_goal']
        self.cost_target = goal_data['cost_target']
        self.lead_time_target = goal_data['lead_time_target']
        self.risk_calc_settings = goal_data['risk_calc_settings']
        
        self.architecture_config_name = architecture_config_name
        if architecture_config_name:
            self.load_folder = 'Architecture/Inputs/' + self.architecture_config_name
            self.save_folder = 'Architecture/Outputs/' + self.architecture_config_name
        else:
            timestamp = time.time()
            dt_object = datetime.datetime.fromtimestamp(timestamp)
            self.formatted_time = dt_object.strftime("%Y-%m-%d_%H-%M-%S")
            self.save_folder = 'sim_runs/monte_carlo_at_' + self.formatted_time
            self.load_folder = None
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        elif self.architecture_config_name:
            clear_folder(self.save_folder)
        
        
        self.convergence_data = {
            'Lead Time Mean': [],
            'Lead Time Variance': [],
            'Cost Mean': [],
            'Cost Variance': []
        }
        
        self.lead_times = []
        self.dev_costs = []
        self.overall_utils = []
        self.total_efforts = {}
        self.mean_utilizations = {}
        
        
        self.use_seeds = use_seeds
        if inital_seed and self.use_seeds:
            random.seed(inital_seed)
        
        if self.use_seeds:
            self.seeds = []


    def run_montecarlo(self, print_additional_information=True, show_plots=True):
        stability = [float('inf')] * 4
        start_time = time.time()
        
        sim_runs_left = self.max_sim_runs
        total_sim_runs = self.max_sim_runs
        sim_run = 0
        while sim_runs_left != 0:
            sim_run_start_time = time.time()
            sim_run += 1
            
            if self.use_seeds:
                seed = random.randint(0, 2**32 - 1)
                random.seed(seed)
                np.random.seed(seed)
                self.seeds.append(seed)
            
            sim = PDsim(overall_quality_goal=self.overall_quality_goal, montecarlo=True, folder=self.load_folder)
            try:
                lead_time, dev_cost, resource_util, overall_util, total_effort = sim.sim_run()
            except Exception as e:
                error = str(e)
                if self.use_seeds:
                    raise ValueError(f'{self.architecture_config_name + ': ' if self.architecture_config_name else ''}Simulation run failed with the following random seed: {seed}. Error: {error}')
                else:
                    raise ValueError(f'{self.architecture_config_name + ': ' if self.architecture_config_name else ''}Simulation run failed. Error: {error}')
            
            # collect data
            self.lead_times.append(lead_time)
            self.dev_costs.append(dev_cost)
            self.overall_utils.append(overall_util)
            for entry, util in resource_util.items():
                if entry not in self.mean_utilizations:
                    self.mean_utilizations[entry] = []
                self.mean_utilizations[entry].append(util)
            for entry, effort in total_effort.items():
                if entry not in self.total_efforts:
                    self.total_efforts[entry] = []
                self.total_efforts[entry].append(effort) 
            
            # convergence data
            self.convergence_data['Lead Time Mean'].append(np.mean(self.lead_times))
            self.convergence_data['Lead Time Variance'].append(np.var(self.lead_times))
            self.convergence_data['Cost Mean'].append(np.mean(self.dev_costs))
            self.convergence_data['Cost Variance'].append(np.var(self.dev_costs))
            
            
            # Update Message
            time_passed = time.time() - start_time
            sim_run_time = time.time() - sim_run_start_time
            print(f'\r[{(time_passed / 60):.1f} min ({sim_run_time:.1f} s)]: Run {sim_run}/{total_sim_runs} completed. {'(' + self.architecture_config_name + ')' if self.architecture_config_name else ''}', end='', flush=True)
            
            # stability check
            if self.check_stability and sim_run >= 2 * self.stability_interval and sim_run % self.stability_interval == 0:
                stability = self.stability_check()
                print(f'           Stability Check (Req: <{(self.stability_criteria * 100)}%): {(stability[0] * 100):.2f}% (Lead Time Mean); {(stability[1] * 100):.2f}% (Lead Time Var); {(stability[2] * 100):.2f}% (Cost Mean); {(stability[3] * 100):.2f}% (Cost Var)')
                if all(stability_value < self.stability_criteria for stability_value in stability):
                    print(f'Distribution stability reached after {sim_run} runs. Simulation is being stopped.')
                    break
            

            sim_runs_left -= 1
            # after runs ask if continuation is wanted
            if sim_runs_left == 0 and show_plots:
                print('Max simulation runs reached.')
                self.convergence_plot()
                additional_runs = input('Add more runs? ')
                if additional_runs and isinstance(additional_runs, int):
                    print(f'{additional_runs} additional runs added.')
                    sim_runs_left += int(additional_runs)
                    total_sim_runs += int(additional_runs)
                else:
                    print('No runs added.')
        
        
        mean_lead_time_ymd = convert_hours_to_ymd(np.mean(self.lead_times))
        
        lead_times_in_weeks = np.array(self.lead_times) / (7 * 24)
        costs_in_thousands = np.array(self.dev_costs) / 1000
        
        mean_lead_time = np.mean(lead_times_in_weeks)
        mean_cost = np.mean(costs_in_thousands)
        mean_util = np.mean(self.overall_utils) * 100
        
        # Sim results 
        print(f'\nMonte Carlo Simulation completed {'(' + self.architecture_config_name + ')' if self.architecture_config_name else ''}.\n')
        
        if sim_run > 2 * self.stability_interval:
            stability = self.stability_check()
            print(f'Stability Check (Req: <{(self.stability_criteria * 100)}%): {(stability[0] * 100):.2f}% (Lead Time Mean); {(stability[1] * 100):.2f}% (Lead Time Var); {(stability[2] * 100):.2f}% (Cost Mean); {(stability[3] * 100):.2f}% (Cost Var)\n')
        else:
            stability = None

        print('Results:')
        print(f'     Mean Lead Time: {mean_lead_time:.1f} weeks ({mean_lead_time_ymd[0]} year(s), {mean_lead_time_ymd[1]} month(s), {mean_lead_time_ymd[2]} day(s))')
        print(f'     Mean Development Cost: ${mean_cost:.1f}k')
        print(f'     Mean Overall Utilization: {(mean_util):.1f}%')
        
        if print_additional_information:
            if include_noise_in_results and simulate_noise:
                print('\nResource Utilizations (including noise) and Effort:')
            else:
                print('\nResource Utilizations and Effort:')
            for entry, utilizations in self.mean_utilizations.items():
                if split_plots != 'overall':
                    if split_plots != 'profession':
                        print(f'     {entry}: Utilization: {(np.mean(utilizations) * 100):.1f}%; Effort: {(np.mean(self.total_efforts[entry]) / work_hours_per_day):.1f} person-days')
                    else:
                        print(f'     {entry}s: Utilization: {(np.mean(utilizations) * 100):.1f}%; Effort: {(np.mean(self.total_efforts[entry]) / work_hours_per_day):.1f} person-days')

        
        
        
        risk_results = calculate_risk(lead_times_in_weeks, costs_in_thousands, lead_time_target, cost_target, self.risk_calc_settings)
        print('\n     Development Risk:')
        print(f'          Probability of Cost Overrun: {(risk_results['prob_cost_overrun']*100):.1f} %')
        print(f'          Risk of Cost Overrun: ${risk_results['cost_risk']:.1f}k')
        print(f'          Probability of Lead Time Overrun: {(risk_results['prob_lead_time_overrun']*100):.1f} %')
        print(f'          Risk of Lead Time Overrun: ${risk_results['lead_time_risk']:.1f}k')
        print(f'          Overall Risk: ${risk_results['combined_risk']:.1f}k\n')
        
        # save data points for single runs
        df_lead_times = pd.DataFrame({'Lead Times (weeks)': lead_times_in_weeks})
        df_costs = pd.DataFrame({'Development Costs (thousands)': costs_in_thousands})
        df_lead_times.to_csv(self.save_folder + '/lead_times_weeks.csv', index=False)
        df_costs.to_csv(self.save_folder + '/dev_costs_thousands.csv', index=False)
        
        
        self.convergence_plot(stability=stability, show_plot=show_plots)
        
        montecarlo_results_plots(lead_times_in_weeks, costs_in_thousands, self.lead_time_target, self.cost_target, self.save_folder, show_plots=show_plots)
        
        if self.architecture_config_name:
            self.save_simulation_results(round(mean_lead_time, 1), round(mean_cost, 1),
                                         round(mean_util, 1), 
                                         round(risk_results['combined_risk'], 1), round(risk_results['cost_risk'], 1), round(risk_results['lead_time_risk'], 1))
            
        if self.use_seeds:
            self.get_representative_samples(lead_times_in_weeks, costs_in_thousands, self.overall_utils, lead_time_target, cost_target)

        

    def stability_check(self):
        previous_lead_times = self.lead_times[:-self.stability_interval]
        previous_dev_costs = self.dev_costs[:-self.stability_interval]
        
        prev_lead_times_mean = np.mean(previous_lead_times)
        prev_lead_times_var = np.var(previous_lead_times)
        prev_dev_costs_mean = np.mean(previous_dev_costs)
        prev_dev_costs_var = np.var(previous_dev_costs)
        
        stability = []
        stability.append(abs(self.convergence_data['Lead Time Mean'][-1] - prev_lead_times_mean) / prev_lead_times_mean)
        stability.append(abs(self.convergence_data['Lead Time Variance'][-1] - prev_lead_times_var) / prev_lead_times_var)
        stability.append(abs(self.convergence_data['Cost Mean'][-1] - prev_dev_costs_mean) / prev_dev_costs_mean)
        stability.append(abs(self.convergence_data['Cost Variance'][-1] - prev_dev_costs_var) / prev_dev_costs_var)
        
        return stability
    
    def convergence_plot(self, stability=None, show_plot=True):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Convergence of Monte Carlo Simulation Metrics', fontsize=16)

        # Convert lead time to weeks and cost to thousands for plotting
        lead_time_avg = [lt / (7 * 24) for lt in self.convergence_data['Lead Time Mean']]
        lead_time_var = [lv / ((7 * 24)**2) for lv in self.convergence_data['Lead Time Variance']]
        cost_avg = [c / 1000 for c in self.convergence_data['Cost Mean']]
        cost_var = [cv / (1000**2) for cv in self.convergence_data['Cost Variance']]

        metrics = [
            (lead_time_avg, 'Lead Time Mean (weeks)', 0, 0),
            (lead_time_var, 'Lead Time Variance (weeks²)', 0, 1),
            (cost_avg, 'Cost Mean ($k)', 1, 0),
            (cost_var, 'Cost Variance ($k²)', 1, 1)
        ]

        # Plot each metric in its respective subplot
        for i, (values, title, row, col) in enumerate(metrics):
            axes[row, col].plot(values, label=title)
            axes[row, col].set_title(title)
            axes[row, col].set_xlim(left=0)
            axes[row, col].set_ylim(bottom=max(0, 0.8*min(values)), top=1.2*max(values))
            axes[row, col].set_xlabel('Simulation Run')
            axes[row, col].set_ylabel(title)
            axes[row, col].grid(True)
            
            if stability:
                last_x = len(values) - 1
                last_y = values[-1]
                stability_text = f'{(stability[i] * 100):.2f}%'
                axes[row, col].annotate(
                    stability_text,
                    xy=(last_x, last_y),
                    xytext=(last_x, last_y * 1.05),
                    ha='center',
                    fontsize=10,
                    color='blue',
                    arrowprops=dict(arrowstyle="->", color='gray', lw=0.5)
                )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.save_folder +  '/convergence.png')
        if show_plot:
            plt.show()
            

    def get_representative_samples(self, lead_times_in_weeks, costs_in_thousands, utilizations, lead_time_target, cost_target):
        print('Generating Most Likely, Worst Case, and Best Case...')
        
        mean_lead_time = np.mean(lead_times_in_weeks)
        mean_dev_cost = np.mean(costs_in_thousands)
        mean_utilization = np.mean(utilizations)

        most_likely_seed = None
        best_case_seed = None
        worst_case_seed = None
        risks = []
        distances = []
        
        for lead_time, dev_cost, util in zip(lead_times_in_weeks, costs_in_thousands, utilizations):
            distance = np.sqrt(((lead_time - mean_lead_time) / mean_lead_time) ** 2 +
                               ((dev_cost - mean_dev_cost) / mean_dev_cost) ** 2 +
                               ((util - mean_utilization) / mean_utilization) ** 2)
            distances.append(distance)
            
            risk = calculate_risk([lead_time], [dev_cost], lead_time_target,cost_target, self.risk_calc_settings)
            risks.append(risk)
        
        
        most_likely_seed = self.seeds[np.argmin(distances)]
        print(f'\nMost Likely Case Seed: {most_likely_seed}')
        sim = PDsim(folder=self.load_folder, file_name_extention=f'MostLikely_with_{most_likely_seed}', random_seed=most_likely_seed)
        sim.sim_run()
        
        best_case_seed = self.seeds[np.argmin(risks)]
        print(f'\nBest Case Seed: {best_case_seed}')
        sim = PDsim(folder=self.load_folder, file_name_extention=f'Best_with_{best_case_seed}', random_seed=best_case_seed)
        sim.sim_run()
        
        worst_case_seed = self.seeds[np.argmax(risks)]
        print(f'\nWorst Case Seed: {worst_case_seed}')
        sim = PDsim(folder=self.load_folder, file_name_extention=f'Worst_with_{worst_case_seed}', random_seed=worst_case_seed)
        sim.sim_run()
        

    
    def save_simulation_results(self, 
                                median_time_to_market_risk, median_development_cost,
                                human_resource_utilization, 
                                overall_development_risk, schedule_risk, development_cost_risk):
        
        results_file = 'Architecture/Outputs/Simulation_Results.csv'
        
        # Create a dictionary for the new row
        results_data = {
            'Name': self.architecture_config_name,
            'Mean Lead Time : weeks': median_time_to_market_risk,
            'Mean Development Cost : $k': median_development_cost,
            'Human Resource Utilization : percent': human_resource_utilization,
            'Overall Development Risk : $k': overall_development_risk,
            'Schedule Risk : $k': schedule_risk,
            'Development Cost Risk : $k': development_cost_risk
        }

        if os.path.isfile(results_file):
            df = pd.read_csv(results_file)
            
            # update existing
            if self.architecture_config_name in df['Name'].values:
                df.set_index('Name', inplace=True)
                new_data = pd.DataFrame([results_data]).set_index('Name')
                df.update(new_data)
                df.reset_index(inplace=True)
                print(f'Results file for the architecture model was updated (Row: {self.architecture_config_name}).\n')
            else:
                # Append the new data as a new row
                df = pd.concat([df, pd.DataFrame([results_data])], ignore_index=True)
                print(f'Warning: New row for {self.architecture_config_name} was added.\n')
        else:
            # Create a new DataFrame and save to file
            df = pd.DataFrame([results_data])
            print('Warning: The results file did not exist. It was created.\n')
        
        # Save the updated DataFrame back to the CSV file
        df.to_csv(results_file, index=False)


def calculate_risk(lead_times, costs, lead_time_target, cost_target, risk_calc_settings):
    
    def impact(overrun, risk_factor_factor, impact_type):
        if impact_type == 'linear':
            return overrun * risk_factor_factor
        elif impact_type == 'quadratic':
            if overrun < 0:
                return -(overrun ** 2) * risk_factor_factor
            else:
                return (overrun ** 2) * risk_factor_factor
        elif impact_type == 'constant':
            return risk_factor_factor
        else:
            raise ValueError("Invalid impact_type. Choose 'linear', 'quadratic', or 'constant'.")
    
    
    cost_risk_factor = risk_calc_settings['cost_risk_factor']
    cost_impact_type = risk_calc_settings['cost_impact_type']
    lead_time_risk_factor = risk_calc_settings['lead_time_risk_factor']
    lead_time_impact_type = risk_calc_settings['lead_time_impact_type']

    if len(lead_times) > 1:
        # Calculate probabilities of overruns
        prob_cost_overrun = np.mean(costs > cost_target)
        prob_lead_time_overrun = np.mean(lead_times > lead_time_target)

        # Filter and calculate cost overruns
        cost_overruns = [cost - cost_target for cost in costs if cost > cost_target]
        total_cost_impact = sum(impact(overrun, cost_risk_factor, cost_impact_type) for overrun in cost_overruns)
        risk_cost = total_cost_impact / len(costs)

        # Filter and calculate lead time overruns
        lead_time_overruns = [lead_time - lead_time_target for lead_time in lead_times if lead_time > lead_time_target]
        total_lead_time_impact = sum(impact(overrun, lead_time_risk_factor, lead_time_impact_type) for overrun in lead_time_overruns)
        risk_lead_time = total_lead_time_impact / len(lead_times) 

        # Calculate risks as probability * mean impact
        combined_risk = risk_cost + risk_lead_time

        # Results
        results = {
            "prob_cost_overrun": prob_cost_overrun,
            "prob_lead_time_overrun": prob_lead_time_overrun,
            "cost_risk": risk_cost,
            "lead_time_risk": risk_lead_time,
            "combined_risk": combined_risk
        }

        return results
    
    # single value
    else:
        cost_risk = impact((costs[0] - cost_target), cost_risk_factor, cost_impact_type)
        lead_time_risk = impact((lead_times[0] - lead_time_target), lead_time_risk_factor, lead_time_impact_type)
        
        return cost_risk + lead_time_risk



def montecarlo_results_plots(lead_times_in_weeks, costs_in_thousands, lead_time_target, cost_target, folder_name, show_plots=True):
    def two_decimals(x, pos):
        return f'{x:.2f}'
    
    def no_decimal(x, pos):
        return f'{x:.0f}'
    
    plt.rcParams["font.family"] = "Times New Roman"
    ticksize = 12
    labelsize = 14
    nbins = 20
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.4)

    # Lead time in weeks
    counts, bins = np.histogram(lead_times_in_weeks, bins=nbins)
    relative_probabilities = counts / len(lead_times_in_weeks)
    axes[0].bar(bins[:-1], relative_probabilities, width=np.diff(bins), color='lightgray',edgecolor="black", align="edge")
    axes[0].set_xlabel('Lead Time (weeks)', fontsize=labelsize)
    axes[0].set_ylabel('Relative Probability', fontsize=labelsize)
    axes[0].yaxis.set_major_formatter(FuncFormatter(two_decimals))
    axes[0].xaxis.set_major_formatter(FuncFormatter(no_decimal))
    axes[0].tick_params(axis='both', which='major', labelsize=ticksize)

    # CDF
    ax2 = axes[0].twinx()
    cdf = np.cumsum(counts) / np.sum(counts)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    cdf = np.insert(cdf, 0, 0)
    cdf = np.append(cdf, 1)
    bin_centers = np.insert(bin_centers, 0, bins[0])
    bin_centers = np.append(bin_centers, bins[-1])
    ax2.plot(bin_centers, cdf, color='black')
    ax2.set_ylabel('Cumulative Probability', fontsize=labelsize)
    ax2.tick_params(axis='both', which='major', labelsize=ticksize)

    # ticks and grid
    bin_width = bin_centers[1] - bin_centers[0]
    time_limits = (min(bin_centers[0], lead_time_target - bin_width), max(bin_centers[-1], lead_time_target + bin_width))
    time_ticks = np.arange(time_limits[0], time_limits[1] + (time_limits[1] - time_limits[0]) / 7, (time_limits[1] - time_limits[0]) / 7)
    axes[0].set_xlim(time_limits[0], time_limits[1])
    cdf_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    primary_ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    ax2.set_ylim(0, 1)
    ax2.set_yticks(cdf_ticks)
    axes[0].set_yticks(primary_ticks)
    axes[0].set_xticks(time_ticks)
    axes[0].grid(True, axis='y', which='both', linestyle='-', linewidth=1, color='black')
    ax2.grid(False)
    
    # Target 
    axes[0].axvline(x=lead_time_target, color='dimgray', linestyle='--', linewidth=2)
    axes[0].text(lead_time_target, axes[0].get_ylim()[1], f'Target Lead Time: {lead_time_target} weeks', 
            color='black', ha='center', va='bottom', fontsize=ticksize)


    # Cost in thousands
    counts, bins = np.histogram(costs_in_thousands, bins=nbins) 
    relative_probabilities = counts / len(costs_in_thousands)
    axes[1].bar(bins[:-1], relative_probabilities, width=np.diff(bins), color='lightgray',edgecolor="black", align="edge")
    axes[1].set_xlabel('Cost ($k)', fontsize=labelsize)
    axes[1].set_ylabel('Relative Probability', fontsize=labelsize)
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    axes[1].tick_params(axis='both', which='major', labelsize=ticksize)

    # CDF
    ax2 = axes[1].twinx()
    cdf = np.cumsum(counts) / np.sum(counts)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    cdf = np.insert(cdf, 0, 0)
    cdf = np.append(cdf, 1)
    bin_centers = np.insert(bin_centers, 0, bins[0])
    bin_centers = np.append(bin_centers, bins[-1])
    ax2.plot(bin_centers, cdf, color='black')
    ax2.set_ylabel('Cumulative Probability', fontsize=labelsize)
    ax2.tick_params(axis='both', which='major', labelsize=ticksize)

    # limits, ticks, and grid
    bin_width = bin_centers[1] - bin_centers[0]
    cost_limits = (min(bin_centers[0], cost_target - bin_width), max(bin_centers[-1], cost_target + bin_width))
    cost_ticks = np.arange(cost_limits[0], cost_limits[1] + (cost_limits[1] - cost_limits[0]) / 7, (cost_limits[1] - cost_limits[0]) / 7)
    axes[1].set_xlim(cost_limits[0], cost_limits[1])
    ax2.set_ylim(0, 1)
    ax2.set_yticks(cdf_ticks)
    axes[1].set_yticks(primary_ticks)
    axes[1].set_xticks(cost_ticks)
    axes[1].grid(True, axis='y', which='both', linestyle='-', linewidth=1, color='black')
    ax2.grid(False)
    
    # Target 
    axes[1].axvline(x=cost_target, color='dimgray', linestyle='--', linewidth=2)
    axes[1].text(cost_target, axes[1].get_ylim()[1], f'Target Cost: ${cost_target}k', 
            color='black', ha='center', va='bottom', fontsize=ticksize)
    
    #plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(folder_name + '/monte_carlo_pdf_cdf.png')
    plt.savefig(folder_name + '/monte_carlo_pdf_cdf.svg', format='svg')
    if show_plots:
        plt.show()


    plt.figure(figsize=(10, 8))
    
    nbins = nbins ** 2
    k = gaussian_kde([lead_times_in_weeks, costs_in_thousands])
    xi, yi = np.mgrid[lead_times_in_weeks.min():lead_times_in_weeks.max():nbins*1j, 
                    costs_in_thousands.min():costs_in_thousands.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    # Plot contours for different probability density levels
    plt.contour(xi, yi, zi.reshape(xi.shape), levels=4, colors='dimgray')
    cmap = plt.get_cmap('Greys')
    lower_half_cmap = mcolors.LinearSegmentedColormap.from_list('lower_half', cmap(np.linspace(0, 0.4, 256)))
    c1 = plt.contourf(xi, yi, zi.reshape(xi.shape), levels=4, cmap=lower_half_cmap)
    cbar = plt.colorbar(c1)
    cbar.set_label('Relative Probability', fontsize=ticksize)
    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=ticksize-1)
    cbar.ax.yaxis.offsetText.set_x(3.2) 
    
    # Labeling and styling
    plt.xlabel('Lead Time (weeks)', fontsize=labelsize)
    plt.ylabel('Cost ($k)', fontsize=labelsize)
    plt.xlim(time_limits[0], time_limits[1])
    plt.ylim(cost_limits[0], cost_limits[1])
    plt.xticks(ticks=time_ticks, fontsize=ticksize)
    plt.yticks(ticks=cost_ticks, fontsize=ticksize)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
    plt.grid(True, linestyle='-', linewidth=1, color='black')
    
    # Targets
    plt.axvline(x=lead_time_target, color='dimgray', linestyle='--', linewidth=2)
    plt.text(lead_time_target, plt.ylim()[1], f'Target Lead Time: {lead_time_target} weeks', 
            color='black', ha='center', va='bottom', fontsize=ticksize)
    plt.axhline(y=cost_target, color='dimgray', linestyle='--', linewidth=2)
    plt.text(plt.xlim()[1]+(plt.xlim()[1]-(plt.xlim()[0]))*0.01, cost_target, f'Target Cost: ${cost_target}k', 
            color='black', ha='left', va='center', rotation=90, fontsize=ticksize)
    
    plt.savefig(folder_name + '/monte_carlo_combined_pdf.png')
    plt.savefig(folder_name + '/monte_carlo_combined_pdf.svg', format='svg')
    if show_plots:
        plt.show()




def montecarlo_box_plots(cost_target, lead_time_target):
    
    def plot_max_min_outliers(data, axis, position):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [y for y in data if y < lower_bound or y > upper_bound]
        
        if outliers:
            if min(outliers) < lower_bound:
                axis.plot(position, min(outliers), 'o', color='black', markerfacecolor='none')
            if max(outliers) > upper_bound:
                axis.plot(position, max(outliers), 'o', color='black', markerfacecolor='none')
    
    
    
    plt.rcParams["font.family"] = "Times New Roman"
    ticksize = 12
    labelsize = 14

    lead_times_data = []
    costs_data = []
    labels = []
    
    for config in sorted(os.path.basename(f.path) for f in os.scandir('Architecture/Outputs') if f.is_dir()):
        lead_time_file = 'Architecture/Outputs/' + config + '/lead_times_weeks.csv'
        cost_file = 'Architecture/Outputs/' + config + '/dev_costs_thousands.csv'
        
        lead_times_data.append(pd.read_csv(lead_time_file)['Lead Times (weeks)'].values)
        costs_data.append(pd.read_csv(cost_file)['Development Costs (thousands)'].values)
        labels.append(config)


    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Lead Time Box Plot
    axes[0].boxplot(lead_times_data, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='lightgray', color="black"), medianprops=dict(color="black"))
    axes[0].set_ylabel('Lead Time (weeks)', fontsize=labelsize)
    axes[0].tick_params(axis='y', which='major', labelsize=ticksize)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    for i, data in enumerate(lead_times_data, start=1):
        plot_max_min_outliers(data, axes[0], i)

    # Target line for lead time
    axes[0].axhline(y=lead_time_target, color='dimgray', linestyle='--', linewidth=2)
    axes[0].text(axes[0].get_xlim()[1], lead_time_target, f'Target', 
                 color='black', ha='right', va='bottom', fontsize=ticksize)

    # Cost Box Plot
    axes[1].boxplot(costs_data, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='lightgray', color="black"), medianprops=dict(color="black"))
    axes[1].set_ylabel('Cost ($k)', fontsize=labelsize)
    axes[1].tick_params(axis='y', which='major', labelsize=ticksize)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
    for i, data in enumerate(costs_data, start=1):
        plot_max_min_outliers(data, axes[1], i)

    # Target line for cost
    axes[1].axhline(y=cost_target, color='dimgray', linestyle='--', linewidth=2)
    axes[1].text(axes[1].get_xlim()[1], cost_target, f'Target', 
                 color='black', ha='right', va='bottom', fontsize=ticksize)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.savefig('Architecture/Outputs/monte_carlo_box_plots.png')
    plt.savefig('Architecture/Outputs/monte_carlo_box_plots.svg', format='svg')



def run_all_architecture_configurations(n_runs, use_seeds=True, configurations=None):
    if not configurations:
        configurations = [os.path.basename(f.path) for f in os.scandir('Architecture/Inputs') if f.is_dir()]
        
    for config in configurations:
        # skip product folder
        if config == 'Product':
            continue
        
        print('\n\n------------------------------------------------------------')
        print(f'Running Monte Carlo for {config}.')
        print('------------------------------------------------------------')
        
        sim = MonteCarlo(
            max_sim_runs=n_runs,
            use_seeds=use_seeds,
            architecture_config_name=config
        )
        sim.run_montecarlo(print_additional_information=False, show_plots=False)
     
        with open('Architecture/Inputs/goals.json', 'r') as file:
            goal_data = json.load(file)

    montecarlo_box_plots(cost_target=goal_data['cost_target'], lead_time_target=goal_data['lead_time_target'])
    
    
def plot_from_csv(folder_name, lead_time_target, cost_target, risk_calc_settings):
    lead_times_df = pd.read_csv(folder_name + '/Lead_Time_Data_Baseline.csv')
    dev_costs_df = pd.read_csv(folder_name + '/Cost_Data_Baseline.csv')
    
    lead_times_weeks = lead_times_df['Lead Times (weeks)'].values
    dev_costs_thousands = dev_costs_df['Development Costs (thousands)'].values

    montecarlo_results_plots(lead_times_weeks, dev_costs_thousands, lead_time_target, cost_target, folder_name)
    
    risk_results = calculate_risk(lead_times_weeks, dev_costs_thousands, lead_time_target, cost_target, risk_calc_settings)
    print('Development Risk:')
    print(f'     Probability of Cost Overrun: {(risk_results['prob_cost_overrun']*100):.1f} %')
    print(f'     Risk of Cost Overrun: ${risk_results['cost_risk']:.1f}k')
    print(f'     Probability of Lead Time Overrun: {(risk_results['prob_lead_time_overrun']*100):.1f} %')
    print(f'     Risk of Lead Time Overrun: ${risk_results['lead_time_risk']:.1f}k')
    print(f'     Overall Risk: ${risk_results['combined_risk']:.1f}k')
    
    


if __name__ == "__main__":
    overall_quality_goal = 0.9
    cost_target = 1600
    lead_time_target = 130
    risk_calc_settings = {
        'cost_risk_factor': 2, # lost revenue ($k) per $k or $k^2 overrun
        'cost_impact_type': 'quadratic',
        'lead_time_risk_factor': 150, # lost revenue ($k) per week or week^2 overrun (depending on impact function)
        'lead_time_impact_type': 'quadratic'
    }
    
    
    if False:
        plot_from_csv('sim_runs/plots', cost_target=cost_target, lead_time_target=lead_time_target, risk_calc_settings=risk_calc_settings)
        montecarlo_box_plots(cost_target=cost_target, lead_time_target=lead_time_target)
    else:
        if True:
            run_all_architecture_configurations(n_runs=1000)
        else:
            sim = MonteCarlo(
                # Sim runs
                max_sim_runs = 10,
                
                # Stability
                check_stability = False,
                stability_criteria = 0.01, # % 
                stability_interval = 100,
                
                # Seeding
                inital_seed=None,
                use_seeds=True,
                
                architecture_config_name=None#'Baseline'
            )
            sim.run_montecarlo(print_additional_information=False)