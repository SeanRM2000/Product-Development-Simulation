import random
import time
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import matplotlib.colors as mcolors
import matplotlib as mpl
from scipy import stats
from scipy.stats import gaussian_kde
import pandas as pd
import os
import shutil
import json
import warnings

# Classes
from sim_run import PDsim

# Functions
from sim_helper_functions import convert_hours_to_ymd, clear_folder

# Parameters
from Inputs.sim_settings import *
from Inputs.tuning_params import *
    
    
class MonteCarlo():
    def __init__(self, max_sim_runs, 
                 check_stability=False, stability_criteria=0.05, stability_interval=100, # stability
                 inital_seed=None, use_seeds=False, # monte carlo repeatability and trackability
                 architecture_config_name=None, # provid if simulation is run based on architecture model inputs
                 skip_errors=False,
                 folder_extention=''
                 ):
        
        # sim settings
        self.max_sim_runs = max_sim_runs
        self.check_stability = check_stability
        self.stability_criteria = stability_criteria
        self.stability_interval =  stability_interval
        self.skip_errors=skip_errors
        
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
            if folder_extention:
                self.load_folder = 'Architecture/Inputs/' + folder_extention + '/' + self.architecture_config_name
                self.save_folder = 'Architecture/Outputs/' + folder_extention + '/' + self.architecture_config_name
            else:
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
        self.final_quality = []
        self.effectivness = []
        self.average_iterations = []
        self.fp_yield = []
        self.cost_from_physical = []
        self.work_efficiency = []
        self.consitency = []
        
        
        self.use_seeds = use_seeds
        if inital_seed and self.use_seeds:
            random.seed(inital_seed)
        
        if self.use_seeds:
            self.seeds = []


    def run_montecarlo(self, show_plots=True, create_samples=False):
        stability = [float('inf')] * 4
        start_time = time.time()
        
        sim_runs_left = self.max_sim_runs
        total_sim_runs = self.max_sim_runs
        skipped_runs = 0
        sim_run = 0
        while sim_runs_left != 0:
            sim_run_start_time = time.time()
            sim_run += 1
            sim_runs_left -= 1
            
            if self.use_seeds:
                seed = random.randint(0, 2**32 - 1)
                random.seed(seed)
                np.random.seed(seed)
                self.seeds.append(seed)
            
            # simulation
            sim = PDsim(overall_quality_goal=self.overall_quality_goal, montecarlo=True, folder=self.load_folder)
            try: 
                results = sim.sim_run()
            
            # error handeling
            except Exception as e:
                error = str(e)
                
                if self.skip_errors:
                    time_passed = time.time() - start_time
                    sim_run_time = time.time() - sim_run_start_time
                    print(f'\n[{(time_passed / 60):.1f} min ({sim_run_time:.1f} s)]: Run {sim_run}/{total_sim_runs}{f' (Seed: {seed})' if self.use_seeds else ''} skipped due to error: {error}')
                    
                    # adjust sim run counters
                    skipped_runs += 1
                    sim_runs_left += 1
                    total_sim_runs += 1
                    
                    if skipped_runs > 20:
                        print('\nMonte Carlo was stopped due to too many errors.\n')
                        return 'skipped'
                    else:
                        continue
                else:
                    if self.use_seeds:
                        raise RuntimeError(f'{self.architecture_config_name + ': ' if self.architecture_config_name else ''}Simulation run failed with the following random seed: {seed}. Error: {error}')
                    else:
                        raise RuntimeError(f'{self.architecture_config_name + ': ' if self.architecture_config_name else ''}Simulation run failed. Error: {error}')
            
            # collect data
            self.dev_costs.append(results[0])
            self.lead_times.append(results[1])
            self.final_quality.append(results[2])
            self.effectivness.append(results[3])
            self.average_iterations.append(results[4])
            self.fp_yield.append(results[5])
            self.cost_from_physical.append(results[6])
            self.work_efficiency.append(results[7])
            self.consitency.append(results[8])


            # convergence data
            self.convergence_data['Lead Time Mean'].append(np.mean(self.lead_times))
            self.convergence_data['Lead Time Variance'].append(np.var(self.lead_times))
            self.convergence_data['Cost Mean'].append(np.mean(self.dev_costs))
            self.convergence_data['Cost Variance'].append(np.var(self.dev_costs))
            
            
            # Update Message
            time_passed = time.time() - start_time
            sim_run_time = time.time() - sim_run_start_time
            print(f'\r[{(time_passed / 60):.1f} min ({sim_run_time:.1f} s)]: Run {sim_run}/{total_sim_runs} {f' (Seed: {seed})' if self.use_seeds else ''} completed.', end='', flush=True)
            
            # stability check
            if self.check_stability and sim_run >= 2 * self.stability_interval and sim_run % self.stability_interval == 0:
                stability = self.stability_check()
                print(f'           Stability Check (Req: <{(self.stability_criteria * 100)}%): {(stability[0] * 100):.2f}% (Lead Time Mean); {(stability[1] * 100):.2f}% (Lead Time Var); {(stability[2] * 100):.2f}% (Cost Mean); {(stability[3] * 100):.2f}% (Cost Var)')
                if all(stability_value < self.stability_criteria for stability_value in stability):
                    print(f'Distribution stability reached after {sim_run} runs. Simulation is being stopped.')
                    break
            

            # after runs ask if continuation is wanted
            if sim_runs_left == 0 and show_plots:
                print('\nMax simulation runs reached.')
                self.convergence_plot()
                additional_runs = input('Add more runs? ')
                try:
                    additional_runs = int(additional_runs)  # Convert to integer
                    print(f'{additional_runs} additional runs added.')
                    sim_runs_left += additional_runs
                    total_sim_runs += additional_runs
                except ValueError:
                    print('No runs added.')
        
        # Sim results 
        end_time = time.time() - start_time
        print(f'\n\nMonte Carlo Simulation completed {'(' + self.architecture_config_name + ')' if self.architecture_config_name else ''} ({(end_time / 60):.1f} min).\n')
        
        if self.skip_errors and skipped_runs > 0:
            print(f'------->Warning: {skipped_runs} runs skipped due to errors\n')
        
        if sim_run > 2 * self.stability_interval:
            stability = self.stability_check()
            print(f'Stability Check (Req: <{(self.stability_criteria * 100)}%): {(stability[0] * 100):.2f}% (Lead Time Mean); {(stability[1] * 100):.2f}% (Lead Time Var); {(stability[2] * 100):.2f}% (Cost Mean); {(stability[3] * 100):.2f}% (Cost Var)\n')
        else:
            stability = None
            
                
        mean_lead_time_ymd = convert_hours_to_ymd(np.mean(self.lead_times))
        
        lead_times_in_weeks = np.array(self.lead_times) / (7 * 24)
        costs_in_thousands = np.array(self.dev_costs) / 1000
        
        mean_lead_time = np.mean(lead_times_in_weeks)
        mean_cost = np.mean(costs_in_thousands)
        mean_quality = np.mean(self.final_quality)
        risk_results = calculate_risk(lead_times_in_weeks, costs_in_thousands, self.lead_time_target, self.cost_target, self.risk_calc_settings)
        mean_effectivness = np.mean(self.effectivness)
        mean_iterations = np.mean(self.average_iterations)
        mean_fp_yield = np.mean(self.fp_yield)
        mean_cost_from_physical = np.mean(self.cost_from_physical)
        mean_work_efficiency = np.mean(self.work_efficiency)
        mean_consitency = np.mean(self.consitency)
        
        

        #print('Results:')
        #print(f'     Mean Lead Time: {mean_lead_time:.1f} weeks ({mean_lead_time_ymd[0]} year(s), {mean_lead_time_ymd[1]} month(s), {mean_lead_time_ymd[2]} day(s))')
        #print(f'     Mean Development Cost: ${mean_cost:.1f}k')
        #print(f'     Overall Risk: ${risk_results['combined_risk']:.1f}k\n')

        
        # save data points for single runs
        df_combined = pd.DataFrame({
            'Development Costs (thousands)': costs_in_thousands,
            'Lead Times (weeks)': lead_times_in_weeks,
            'Effectivness': self.effectivness,
            'Average Iterations': self.average_iterations,
            'First Pass Yield': self.fp_yield,
            '% Cost from physical Prot. / Test': self.cost_from_physical,
            'Work Efficency': self.work_efficiency,
            'Average Consitency': self.consitency
        })
        df_combined.to_csv(self.save_folder + '/run_data.csv', index=False)
        
        self.convergence_plot(stability=stability, show_plot=show_plots)
        
        montecarlo_results_plots(lead_times_in_weeks, costs_in_thousands, self.lead_time_target, self.cost_target, self.save_folder, show_plots=show_plots)
        
        if self.architecture_config_name:
            self.save_statistical_data(costs_in_thousands,
                                       lead_times_in_weeks,
                                       self.final_quality,
                                       risk_results['combined_risk'],
                                       self.effectivness,
                                       self.average_iterations,
                                       self.fp_yield,
                                       self.cost_from_physical,
                                       self.work_efficiency,
                                       self.consitency
                                       )
            
            self.save_simulation_results(round(mean_cost, 1),
                                         round(mean_lead_time, 1), 
                                         round(mean_quality, 3),
                                         round(risk_results['combined_risk'], 1),
                                         round(mean_effectivness, 3),
                                         round(mean_iterations, 1),
                                         round(mean_fp_yield, 3),
                                         round(mean_cost_from_physical, 3),
                                         round(mean_work_efficiency, 3),
                                         round(mean_consitency, 3)
                                         )
            
        if self.use_seeds and create_samples:
            self.get_representative_samples(lead_times_in_weeks, costs_in_thousands, self.lead_time_target, self.cost_target)
            
        plt.close('all')
        
        return skipped_runs


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
        
        ticksize = 10
        labelsize = 12
        
        fig, ax = plt.subplots(figsize=(7.5, 1.8))

        # Convert lead time to weeks and cost to thousands for plotting
        lead_time_avg = [lt / (7 * 24) for lt in self.convergence_data['Lead Time Mean']]
        lead_time_var = [lv / ((7 * 24)**2) for lv in self.convergence_data['Lead Time Variance']]
        cost_avg = [c / 1000 for c in self.convergence_data['Cost Mean']]
        cost_var = [cv / (1000**2) for cv in self.convergence_data['Cost Variance']]

        # Normalize each metric such that the last value is 1
        lead_time_avg_norm = [val / lead_time_avg[-1] for val in lead_time_avg]
        lead_time_var_norm = [val / lead_time_var[-1] for val in lead_time_var]
        cost_avg_norm = [val / cost_avg[-1] for val in cost_avg]
        cost_var_norm = [val / cost_var[-1] for val in cost_var]

        metrics = [
            (lead_time_avg_norm, 'Lead Time Mean (relative)', 'royalblue', '-'),
            (lead_time_var_norm, 'Lead Time Variance (relative)', 'royalblue', '--'),
            (cost_avg_norm, 'Cost Mean (relative)', 'darkorange', '-'),
            (cost_var_norm, 'Cost Variance (relative)', 'darkorange', '--')
        ]

        # Plot each metric on the same subplot
        for values, label, color, line in metrics:
            ax.plot(values, label=label, color=color, linestyle=line, linewidth=1.5)

        #ax.set_title('Convergence of Monte Carlo Simulation Metrics (Relative)')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=round(max(lead_time_avg_norm + lead_time_var_norm + cost_avg_norm + cost_var_norm) * 1.1, 1))
        ax.set_xlabel('Simulation Run', fontsize=labelsize)
        ax.set_ylabel('Normalized Value', fontsize=labelsize)
        ax.tick_params(axis='both', labelsize=ticksize)
        ax.grid(True)
        #ax.legend(loc='upper left', fontsize='small')

        plt.tight_layout(pad=0)
        plt.savefig(self.save_folder + '/convergence_relative.svg', format='svg')
        #plt.savefig(self.save_folder + '/convergence_relative.png')
        if show_plot:
            plt.show()
            

    def get_representative_samples(self, lead_times_in_weeks, costs_in_thousands, lead_time_target, cost_target):
        print('Generating Most Likely, Worst Case, and Best Case...')
        
        mean_lead_time = np.mean(lead_times_in_weeks)
        mean_dev_cost = np.mean(costs_in_thousands)

        most_likely_seed = None
        best_case_seed = None
        worst_case_seed = None
        risks = []
        distances = []
        
        for lead_time, dev_cost in zip(lead_times_in_weeks, costs_in_thousands):
            distance = np.sqrt(((lead_time - mean_lead_time) / mean_lead_time) ** 2 +
                               ((dev_cost - mean_dev_cost) / mean_dev_cost) ** 2)
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
        

    def save_statistical_data(self,
                              cost, 
                              lead_time,
                              quality,
                              risk,
                              effectivness,
                              n_iterations,
                              fp_yield,
                              cost_physical,
                              work_eff,
                              consitency
                              ):
        
        n = len(cost)
        t_critical = stats.t.ppf((1 + 0.95) / 2, df=n-1)
        
        def confidence(data, upper=False):
            mean = np.mean(data)
            std_err = stats.sem(data)
            margin_of_error = t_critical * std_err
            
            if upper:
                return mean + margin_of_error
            else:
                return mean - margin_of_error
        
        results_file = 'Architecture/Simulation_Results.csv'
        
        if 'Baseline' in self.architecture_config_name:
            doe = 'Baseline'
            number = ''
        else:
            doe, number = self.architecture_config_name.split('-')
        
        results_data = {
            "Name": self.architecture_config_name,
            "DOE": doe,
            "Number": number,
            
            "Cost Mean": round(np.mean(cost), 1),
            "Cost Median": round(np.median(cost), 1),
            "Cost 95confidence Lower": round(confidence(cost), 1),
            "Cost 95confidence Upper": round(confidence(cost, upper=True), 1),
            "Cost Q25": round(np.percentile(cost, [25, 75])[0], 1),
            "Cost Q75": round(np.percentile(cost, [25, 75])[1], 1),
            
            "Lead Time Mean": round(np.mean(lead_time), 1),
            "Lead Time Median": round(np.median(lead_time), 1),
            "Lead Time 95confidence Lower": round(confidence(lead_time), 1),
            "Lead Time 95confidence Upper": round(confidence(lead_time, upper=True), 1),
            "Lead Time Q25": round(np.percentile(lead_time, [25, 75])[0], 1),
            "Lead Time Q75": round(np.percentile(lead_time, [25, 75])[1], 1),
            
            "Quality Mean": round(np.mean(quality), 3),
            "Quality Median": round(np.median(quality), 3),
            "Quality 95confidence Lower": round(confidence(quality), 3),
            "Quality 95confidence Upper": round(confidence(quality, upper=True), 3),
            "Quality Q25": round(np.percentile(quality, [25, 75])[0], 3),
            "Quality Q75": round(np.percentile(quality, [25, 75])[1], 3),
            
            "Risk": round(risk, 1),
            
            "Effectivness Mean": round(np.mean(effectivness), 3),
            "Effectivness Median": round(np.median(effectivness), 3),
            "Effectivness 95confidence Lower": round(confidence(effectivness), 3),
            "Effectivness 95confidence Upper": round(confidence(effectivness, upper=True), 3),
            "Effectivness Q25": round(np.percentile(effectivness, [25, 75])[0], 3),
            "Effectivness Q75": round(np.percentile(effectivness, [25, 75])[1], 3),
            
            "Average Iterations Mean": round(np.mean(n_iterations), 1),
            "Average Iterations Median": round(np.median(n_iterations), 1),
            "Average Iterations 95confidence Lower": round(confidence(n_iterations), 1),
            "Average Iterations 95confidence Upper": round(confidence(n_iterations, upper=True), 1),
            "Average Iterations Q25": round(np.percentile(n_iterations, [25, 75])[0], 1),
            "Average Iterations Q75": round(np.percentile(n_iterations, [25, 75])[1], 1),
            
            "First Pass Yield Mean": round(np.mean(fp_yield), 3),
            "First Pass Yield Median": round(np.median(fp_yield), 3),
            "First Pass Yield 95confidence Lower": round(confidence(fp_yield), 3),
            "First Pass Yield 95confidence Upper": round(confidence(fp_yield, upper=True), 3),
            "First Pass Yield Q25": round(np.percentile(fp_yield, [25, 75])[0], 3),
            "First Pass Yield Q75": round(np.percentile(fp_yield, [25, 75])[1], 3),
            
            "Rel Cost Physical Mean": round(np.mean(cost_physical), 3),
            "Rel Cost Physical Median": round(np.median(cost_physical), 3),
            "Rel Cost Physical 95confidence Lower": round(confidence(cost_physical), 3),
            "Rel Cost Physical 95confidence Upper": round(confidence(cost_physical, upper=True), 3),
            "Rel Cost Physical Q25": round(np.percentile(cost_physical, [25, 75])[0], 3),
            "Rel Cost Physical Q75": round(np.percentile(cost_physical, [25, 75])[1], 3),
            
            "Work Efficency Mean": round(np.mean(work_eff), 3),
            "Work Efficency Median": round(np.median(work_eff), 3),
            "Work Efficency 95confidence Lower": round(confidence(work_eff), 3),
            "Work Efficency 95confidence Upper": round(confidence(work_eff, upper=True), 3),
            "Work Efficency Q25": round(np.percentile(work_eff, [25, 75])[0], 3),
            "Work Efficency Q75": round(np.percentile(work_eff, [25, 75])[1], 3),
            
            "Average Consitency Mean": round(np.mean(consitency), 3),
            "Average Consitency Median": round(np.median(consitency), 3),
            "Average Consitency 95confidence Lower": round(confidence(consitency), 3),
            "Average Consitency 95confidence Upper": round(confidence(consitency, upper=True), 3),
            "Average Consitency Q25": round(np.percentile(consitency, [25, 75])[0], 3),
            "Average Consitency Q75": round(np.percentile(consitency, [25, 75])[1], 3)
        }

        
        self.save_results_file(results_file, results_data)

    
    
    def save_simulation_results(self, 
                                cost, 
                                lead_time,
                                quality,
                                risk,
                                effectivness,
                                iterations,
                                fp_yield,
                                cost_physical,
                                work_eff,
                                consitency
                                ):
        
        results_file = 'Architecture/Simulation_Results_means.csv'

        if 'Baseline' in self.architecture_config_name:
            doe = 'Baseline'
            number = ''
        else:
            doe, number = self.architecture_config_name.split('-')
        
        results_data = {
            "Name": self.architecture_config_name,
            "DOE": doe,
            "Number": number,
            'Mean Cost': cost,
            'Mean Lead Time': lead_time,
            'Mean Quality': quality,
            'Risk': risk,
            'Effectivness': effectivness,
            'Average Iterations': iterations,
            'First Pass Yield': fp_yield,
            'Rel Cost Physical': cost_physical,
            'Work Efficency': work_eff,
            'Average Consitency': consitency
        }
        
        self.save_results_file(results_file, results_data)


    def save_results_file(self, results_file, results_data):
        if os.path.isfile(results_file):
            df = pd.read_csv(results_file)
            
            # update existing
            if self.architecture_config_name in df['Name'].values:
                df.set_index('Name', inplace=True)
                new_data = pd.DataFrame([results_data]).set_index('Name')
                df.update(new_data)
                df.reset_index(inplace=True)
            else:
                # Append the new data as a new row
                df = pd.concat([df, pd.DataFrame([results_data])], ignore_index=True)
        else:
            # Create a new DataFrame and save to file
            df = pd.DataFrame([results_data])
        
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
    def one_decimals(x, pos):
        return f'{x:.1f}'
    def two_decimals(x, pos):
        return f'{x:.2f}'
    
    def no_decimal(x, pos):
        return f'{x:.0f}'
    
    ticksize = 10
    labelsize = 12
    nbins = 20
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.2))
    

    # Lead time in weeks
    lead_times_in_weeks = lead_times_in_weeks / 52.14 * 12
    lead_time_target = lead_time_target / 52.14 * 12
    
    counts, bins = np.histogram(lead_times_in_weeks, bins=nbins)
    relative_probabilities = counts / len(lead_times_in_weeks)
    axes[0].bar(bins[:-1], relative_probabilities, width=np.diff(bins), color='lightgray',edgecolor="black", align="edge", linewidth=0.7)
    axes[0].set_xlabel('Lead Time (months)', fontsize=labelsize)
    axes[0].set_ylabel('Relative Probability', fontsize=11)
    axes[0].yaxis.set_major_formatter(FuncFormatter(one_decimals))
    axes[0].xaxis.set_major_formatter(FuncFormatter(one_decimals))
    axes[0].tick_params(axis='both', which='major', labelsize=ticksize)

    # CDF
    ax2 = axes[0].twinx()
    cdf = np.cumsum(counts) / np.sum(counts)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    cdf = np.insert(cdf, 0, 0)
    cdf = np.append(cdf, 1)
    bin_centers = np.insert(bin_centers, 0, bins[0])
    bin_centers = np.append(bin_centers, bins[-1])
    ax2.plot(bin_centers, cdf, color='black', linewidth=1.0) # black
    ax2.set_ylabel('Cumulative Probability', fontsize=11) # black
    ax2.tick_params(axis='both', which='major', labelsize=ticksize)

    # ticks and grid
    bin_width = bin_centers[1] - bin_centers[0]
    time_limits = (min(bin_centers[0], lead_time_target - bin_width), max(bin_centers[-1], lead_time_target + bin_width))
    time_ticks = np.linspace(time_limits[0], time_limits[1], 5)
    axes[0].set_xlim(time_limits[0], time_limits[1])
    cdf_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    primary_ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    ax2.set_ylim(0, 1)
    ax2.set_yticks(cdf_ticks)
    if max(relative_probabilities) > primary_ticks[-1]:
        factor = max(relative_probabilities)*1.2 / primary_ticks[-1]
        for i, tick in enumerate(primary_ticks):
            primary_ticks[i] = round(tick * factor, 2)
    axes[0].set_yticks(primary_ticks)
    axes[0].set_xticks(time_ticks)
    ax2.grid(False)
    
    # Target 
    axes[0].axvline(x=lead_time_target, color='dimgray', linestyle='--', linewidth=1.5) # dimgray
    axes[0].text(lead_time_target, axes[0].get_ylim()[1], f'Target', # Lead Time: {lead_time_target} weeks', 
            color='black', ha='center', va='bottom', fontsize=ticksize)


    # Cost in thousands
    costs_in_thousands = costs_in_thousands / 1000
    cost_target = cost_target / 1000
    counts, bins = np.histogram(costs_in_thousands, bins=nbins) 
    relative_probabilities = counts / len(costs_in_thousands)
    axes[1].bar(bins[:-1], relative_probabilities, width=np.diff(bins), color='lightgray',edgecolor="black", align="edge", linewidth=0.7)
    axes[1].set_xlabel('Cost ($m)', fontsize=labelsize)
    axes[1].set_ylabel('Relative Probability', fontsize=11)
    axes[1].yaxis.set_major_formatter(FuncFormatter(one_decimals))
    axes[1].xaxis.set_major_formatter(FuncFormatter(two_decimals))
    axes[1].tick_params(axis='both', which='major', labelsize=ticksize)

    # CDF
    ax2 = axes[1].twinx()
    cdf = np.cumsum(counts) / np.sum(counts)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    cdf = np.insert(cdf, 0, 0)
    cdf = np.append(cdf, 1)
    bin_centers = np.insert(bin_centers, 0, bins[0])
    bin_centers = np.append(bin_centers, bins[-1])
    ax2.plot(bin_centers, cdf, color='black', linewidth=1.0) # black
    ax2.set_ylabel('Cumulative Probability', fontsize=11) # black
    ax2.tick_params(axis='both', which='major', labelsize=ticksize)

    # limits, ticks, and grid
    bin_width = bin_centers[1] - bin_centers[0]
    cost_limits = (min(bin_centers[0], cost_target - bin_width), max(bin_centers[-1], cost_target + bin_width))
    cost_ticks = np.linspace(cost_limits[0], cost_limits[1], 5)
    axes[1].set_xlim(cost_limits[0], cost_limits[1])
    ax2.set_ylim(0, 1)
    ax2.set_yticks(cdf_ticks)
    if max(relative_probabilities) > primary_ticks[-1]:
        factor = max(relative_probabilities)*1.2 / primary_ticks[-1]
        for i, tick in enumerate(primary_ticks):
            primary_ticks[i] = round(tick * factor, 2)
        axes[0].set_yticks(primary_ticks)
    axes[1].set_yticks(primary_ticks)
    axes[1].set_xticks(cost_ticks)
    ax2.grid(False)
    
    # Target 
    axes[1].axvline(x=cost_target, color='dimgray', linestyle='--', linewidth=1.5) # dimgray
    axes[1].text(cost_target, axes[1].get_ylim()[1], f'Target', # Cost: ${cost_target}k', 
            color='black', ha='center', va='bottom', fontsize=ticksize)
    
    #axes[0].text(0.5, -0.25, '(a)', transform=axes[0].transAxes, fontsize=labelsize, ha='center', va='center')
    #axes[1].text(0.5, -0.25, '(b)', transform=axes[1].transAxes, fontsize=labelsize, ha='center', va='center')
    
    plt.tight_layout(pad=0)#rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(wspace=0.6)
    #plt.savefig(folder_name + '/monte_carlo_pdf_cdf.png')
    plt.savefig(folder_name + '/monte_carlo_pdf_cdf.svg', format='svg')
    if show_plots:
        plt.show()

    return


def plot_cost_lead_time_conture(lead_times_in_weeks, costs_in_thousands, lead_time_target, cost_target, folder_name, show_plots=True):
    labelsize = 12
    ticksize = 10
    
    
    plt.figure(figsize=(7.5, 7.5))
    
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
    cbar.set_label('Relative Probability', fontsize=labelsize)
    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.ax.yaxis.offsetText.set_x(3.2) 
    
    # Labeling and styling
    plt.xlabel('Lead Time (weeks)', fontsize=labelsize)
    plt.ylabel('Cost ($k)', fontsize=labelsize)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
    plt.grid(True, linestyle='-', linewidth=1, color='black')
    
    # Targets
    plt.axvline(x=lead_time_target, color='red', linestyle='--', linewidth=2) # dimgray
    plt.text(lead_time_target, plt.ylim()[1], f'Target Lead Time', #: {lead_time_target} weeks', 
            color='black', ha='center', va='bottom', fontsize=ticksize)
    plt.axhline(y=cost_target, color='red', linestyle='--', linewidth=2) # dimgray
    plt.text(plt.xlim()[1]+(plt.xlim()[1]-(plt.xlim()[0]))*0.01, cost_target, f'Target Cost', #: ${cost_target}k', 
            color='black', ha='left', va='center', rotation=90, fontsize=ticksize)
    
    plt.tight_layout(pad=0)
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
    ticksize = 10
    labelsize = 12

    lead_times_data = []
    costs_data = []
    labels = []
    
    abbreviations={
        'Baseline': 'BL',
        'Config1': 'C1',
        'Config2': 'C2',
        'Config3': 'C3',
        'Config4': 'C4'
    }
    rot=0
    if rot==0:
        orientation='center'
    else:
        orientation='right'
    
    for config in sorted(os.path.basename(f.path) for f in os.scandir('Architecture/Outputs') if f.is_dir()):
        lead_time_file = 'Architecture/Outputs/' + config + '/lead_times_weeks.csv'
        cost_file = 'Architecture/Outputs/' + config + '/dev_costs_thousands.csv'
        
        lead_times_data.append(pd.read_csv(lead_time_file)['Lead Times (weeks)'].values)
        costs_data.append(pd.read_csv(cost_file)['Development Costs (thousands)'].values)
        
        # if config in abbreviations:
        #     labels.append(abbreviations[config])
        # else:
        labels.append(config)


    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))

    # Lead Time Box Plot
    axes[0].boxplot(lead_times_data, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='lightgray', color="black"), medianprops=dict(color="black"))
    axes[0].set_ylabel('Lead Time (weeks)', fontsize=labelsize, fontweight='bold')
    axes[0].tick_params(axis='y', which='major', labelsize=ticksize)
    axes[0].set_xticklabels(labels, rotation=rot, ha=orientation)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    #for i, data in enumerate(lead_times_data, start=1):
    #    plot_max_min_outliers(data, axes[0], i)

    # Target line for lead time
    axes[0].axhline(y=lead_time_target, color='red', linestyle='--', linewidth=2) # dimgray
    #axes[0].text(axes[0].get_xlim()[1], lead_time_target, f'Target', 
    #             color='black', ha='right', va='bottom', fontsize=ticksize)

    # Cost Box Plot
    axes[1].boxplot(costs_data, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='lightgray', color="black"), medianprops=dict(color="black"))
    axes[1].set_ylabel('Cost ($k)', fontsize=labelsize, fontweight='bold')
    axes[1].tick_params(axis='y', which='major', labelsize=ticksize)
    axes[1].set_xticklabels(labels, rotation=rot, ha=orientation)
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
    #for i, data in enumerate(costs_data, start=1):
    #    plot_max_min_outliers(data, axes[1], i)

    # Target line for cost
    axes[1].axhline(y=cost_target, color='red', linestyle='--', linewidth=2) # dimgray
    #axes[1].text(axes[1].get_xlim()[1], cost_target, f'Target', 
    #             color='black', ha='right', va='bottom', fontsize=ticksize)

    plt.tight_layout(pad=0)#rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    fig.subplots_adjust(wspace=0.23)
    plt.savefig('Architecture/Outputs/monte_carlo_box_plots.png')
    plt.savefig('Architecture/Outputs/monte_carlo_box_plots.svg', format='svg')



def run_all_architecture_configurations(n_runs, move_folders=False, use_seeds=True, configurations=None, create_samples=False, skip_errors=False, folder_extention=''):
    warnings.filterwarnings("ignore")
    if not configurations:
        configurations = [os.path.basename(f.path) for f in os.scandir('Architecture/Inputs/' + folder_extention) if f.is_dir()]
    
    start_time = time.time()
    n_total_errors = 0
    n_configs = 0
    configs_skipped = []
    
    for config in list(configurations):
        if config in {'Product', 'completed', 'skip'}:
            configurations.remove(config)
    
    if not configurations:
        print('\nNo configuration folders found\n')
        return
    
    for config in configurations:
        
        n_configs += 1
        
        print('\n\n------------------------------------------------------------')
        print(f'Running Monte Carlo for {config}.')
        print('------------------------------------------------------------')
        
        sim = MonteCarlo(
            max_sim_runs=n_runs,
            use_seeds=use_seeds,
            architecture_config_name=config,
            skip_errors=skip_errors,
            folder_extention=folder_extention
        )
        result = sim.run_montecarlo(show_plots=False, create_samples=create_samples)
        if result == 'skipped':
            configs_skipped.append(config)
        else:
            n_total_errors += result
        
        # move folder
        if move_folders:
            destination = 'Architecture/Inputs/' + folder_extention + '/completed'
            if not os.path.exists(destination):
                os.makedirs(destination)
            shutil.move('Architecture/Inputs/' + folder_extention + '/' + config, destination)
            print('Configuration folder moved to \'completed\'')
    
    total_time = (time.time() - start_time) / 60 # min
    
    print('\n------------------------------------------------------------')
    print('All runs completed\n')
    print(f'Total time: {(total_time / 60):.1f} h')
    print(f'Average time per configuration: {(total_time / n_configs):.1f} min')
    print(f'Number of single runs skipped due to errors: {n_total_errors} ({(100 * n_total_errors / (n_total_errors + n_runs * (n_configs - len(configs_skipped)))):.3f} %)\n')
    print(f'Completely skipped configs: {configs_skipped if configs_skipped else None}')
    
    #with open('Architecture/Inputs/goals.json', 'r') as file:
    #    goal_data = json.load(file)

    #montecarlo_box_plots(cost_target=goal_data['cost_target'], lead_time_target=goal_data['lead_time_target'])
    
    
def plot_from_csv(folder_name, lead_time_target, cost_target):
    lead_times_df = pd.read_csv(folder_name + '/Lead_Time_Data_Baseline.csv')
    dev_costs_df = pd.read_csv(folder_name + '/Cost_Data_Baseline.csv')
    
    lead_times_weeks = lead_times_df['Lead Times (weeks)'].values
    dev_costs_thousands = dev_costs_df['Development Costs (thousands)'].values

    montecarlo_results_plots(lead_times_weeks, dev_costs_thousands, lead_time_target, cost_target, folder_name)



if __name__ == "__main__":
    plt.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams['svg.fonttype'] = 'none'
    
    if True:
        if True:
            run_all_architecture_configurations(
                
                n_runs=400, # 400
                                                
                move_folders=True, 
                skip_errors=True, 
                create_samples=False, 
                folder_extention='DOE4 - Full 1' # DOE4 - Full 2, DOE4 - Full 3
            ) 
            
        else:
            warnings.filterwarnings("ignore")
            sim = MonteCarlo(
                # Sim runs
                max_sim_runs = 1500,
                
                # Stability
                check_stability = False,
                stability_criteria = 0.01, # % 
                stability_interval = 100,
                
                # Seeding
                inital_seed=None,
                use_seeds=True,
                
                architecture_config_name='Baseline', # Baseline
                folder_extention='', # ''
                
                skip_errors=True
            )
            sim.run_montecarlo(create_samples=False, show_plots=False)
            
    else:
        # plotting from csv
        cost_target = 1500
        lead_time_target = 110
        plot_from_csv('sim_runs/plots', cost_target=cost_target, lead_time_target=lead_time_target)
        montecarlo_box_plots(cost_target=cost_target, lead_time_target=lead_time_target)