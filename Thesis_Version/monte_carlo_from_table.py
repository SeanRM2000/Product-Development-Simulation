import random
import time
import numpy as np
import pandas as pd
import os
import json
import warnings
import copy

# Classes
from sim_run import PDsim

# Parameters
from Inputs.sim_settings import *
from Inputs.tuning_params import *
    
    
class HypercubeSampleRun():
    def __init__(self, folder, n_sim_runs, inital_seed=None):
        
        self.folder = folder
        self.hypercube_inputs = pd.read_csv(self.folder + "/DOE_table.csv")
        
        # sim settings
        self.n_sim_runs = n_sim_runs

        
        # targets
        with open('Architecture/Inputs/goals.json', 'r') as file:
            goal_data = json.load(file)
        self.overall_quality_goal = goal_data['overall_quality_goal']
        
        with open('Architecture/Inputs/Baseline/organization.json', "r") as f:
            self.org_data_orig = json.load(f)
        with open('Architecture/Inputs/Baseline/tools_with_new.json', "r") as f:
            self.tools_data_with_new = json.load(f)
        with open('Architecture/Inputs/Baseline/tools.json', "r") as f:
            self.tools_data_orig = json.load(f)


        if inital_seed:
            random.seed(inital_seed)
            np.random.seed(inital_seed)
            

    def run_sim(self):
        start_time = time.time()
        skipped_runs = 0
        
        
        
        for index, row in self.hypercube_inputs.iterrows():
            
            print(f'\n\nNow simulating row {index+1}/{self.hypercube_inputs.shape[0]}:')

            self.change_inputfile(row)
            
            lead_times = []
            dev_costs = []
            effectivness = []
            average_iterations = []
            fp_yield = []
            cost_from_physical = []
            work_efficiency = []
            consitency = []
        
            sim_runs_left = self.n_sim_runs
            total_sim_runs = self.n_sim_runs
            sim_run = 0
            while sim_runs_left != 0:
                sim_run_start_time = time.time()
                sim_run += 1
                sim_runs_left -= 1
                
                # simulation
                sim = PDsim(overall_quality_goal=self.overall_quality_goal, montecarlo=True, folder=self.folder)
                try: 
                    results = sim.sim_run()
                    
                    # collect data
                    dev_costs.append(results[0])
                    lead_times.append(results[1])
                    effectivness.append(results[3])
                    average_iterations.append(results[4])
                    fp_yield.append(results[5])
                    cost_from_physical.append(results[6])
                    work_efficiency.append(results[7])
                    consitency.append(results[8])
                    
                    # Update Message
                    time_passed = time.time() - start_time
                    sim_run_time = time.time() - sim_run_start_time
                    print(f'\r   [{(time_passed / 60):.1f} min ({sim_run_time:.1f} s)]: Run {sim_run}/{total_sim_runs} of config {index} completed.', end='', flush=True)
                    
                    
                # error handeling
                except Exception as e:
                    error = str(e)

                    time_passed = time.time() - start_time
                    sim_run_time = time.time() - sim_run_start_time
                    print(f'\n   [{(time_passed / 60):.1f} min ({sim_run_time:.1f} s)]: Run {sim_run}/{total_sim_runs} of config {index} skipped due to error: {error}')
                    
                    # adjust sim run counters
                    skipped_runs += 1
                    sim_runs_left += 1
                    total_sim_runs += 1
            
            ######## end of while loop
                    
            lead_times_in_weeks = np.array(lead_times) / (7 * 24)
            costs_in_thousands = np.array(dev_costs) / 1000
            
            mean_lead_time = np.mean(lead_times_in_weeks)
            mean_cost = np.mean(costs_in_thousands)
            mean_effectivness = np.mean(effectivness)
            mean_iterations = np.mean(average_iterations)
            mean_fp_yield = np.mean(fp_yield)
            mean_cost_from_physical = np.mean(cost_from_physical)
            mean_work_efficiency = np.mean(work_efficiency)
            mean_consitency = np.mean(consitency)

            self.save_simulation_results(
                costs_in_thousands,
                lead_times_in_weeks,
                effectivness,
                average_iterations,
                fp_yield,
                cost_from_physical,
                work_efficiency,
                consitency,
                self.hypercube_inputs.iloc[index],
                means=False
            )

            self.save_simulation_results(
                [round(mean_cost, 1)],
                [round(mean_lead_time, 1)], 
                [round(mean_effectivness, 3)],
                [round(mean_iterations, 1)],
                [round(mean_fp_yield, 3)],
                [round(mean_cost_from_physical, 3)],
                [round(mean_work_efficiency, 3)],
                [round(mean_consitency, 3)],
                self.hypercube_inputs.iloc[index],
                means=True
            )

        # Sim results
        total_time = (time.time() - start_time) / 60 # min
        print('\n------------------------------------------------------------')
        print('All runs completed\n')
        print(f'Total time: {(total_time / 60):.1f} h')
        print(f'Average time per configuration: {(total_time / self.hypercube_inputs.shape[0]):.1f} min')
        print(f'Number of single runs skipped due to errors: {skipped_runs} ({(100 * skipped_runs / (skipped_runs + self.n_sim_runs * self.hypercube_inputs.shape[0])):.3f} %)\n')

    
    def save_simulation_results(self, 
                                cost, 
                                lead_time,
                                effectivness,
                                iterations,
                                fp_yield,
                                cost_physical,
                                work_eff,
                                consitency,
                                input_row,
                                means
                                ):
        
        if means:
            results_file = self.folder + '/Hypercube_Results_means.csv'
        else:
            results_file = self.folder + '/Hypercube_Results.csv'

        
        results_data = {
            'Cost': cost,
            'Lead Time': lead_time,
            'Effectivness': effectivness,
            'Average Iterations': iterations,
            'First Pass Yield': fp_yield,
            'Rel Cost Physical': cost_physical,
            'Work Efficency': work_eff,
            'Average Consitency': consitency
        }

        results = pd.DataFrame(results_data)
                
        input_data = pd.DataFrame([input_row])
        input_data = pd.concat([input_data]*results.shape[0], ignore_index=True)

        df_combined = pd.concat([input_data, results], axis=1)
        
        df = pd.read_csv(results_file)
        
        df = pd.concat([df, df_combined], ignore_index=True)

        df.to_csv(results_file, index=False)
        

    def change_inputfile(self, row):
        
        def update_dl(data, new_value, type):

            if isinstance(data, dict):
                for key, value in data.items():
                    if key == "digital_literacy" and isinstance(value, dict):
                        if type in value: 
                            value[type] = new_value
                    else:
                        update_dl(value, new_value, type)
            elif isinstance(data, list):
                for item in data:
                    update_dl(item, new_value, type)
            return data
        
        def update_tools(data, tool, accuracy, interop, tool_complexity=None):

            for tool_name, tool_data in data.items():
                if tool_name == tool:
                    tool_data["parameters"]["accuracy"] = accuracy
                    tool_data["parameters"]["interoperability"] = interop
                    
                    if tool_complexity:
                        tool_data["parameters"]["use_complexity"] = tool_complexity
                
            return data
            
        
        digital_lit_Eng = round(row["DL_Eng"], 4)
        digital_lit_EKM = round(row["DL_EKM"], 4)
        NewTool = row["New Tool"]
        new_tool_acc = round(row["T_Acc (new)"], 4)
        new_tool_inter = round(row["T_I (new)"], 4)
        new_tool_usab = round(row["T_Usab (new)"], 4)
        

        org_data = copy.deepcopy(self.org_data_orig)
        new_org_data = update_dl(org_data, digital_lit_EKM, 'EngineeringSupportTools')
        new_org_data = update_dl(new_org_data, digital_lit_Eng, 'EngineeringTools')
        
        with open(os.path.join(self.folder, "organization.json"), "w") as f:
            json.dump(new_org_data, f, indent=4)
        
        
        if NewTool == True:
            tools_data = copy.deepcopy(self.tools_data_with_new)
            new_tool_data = update_tools(tools_data, 'HFSystemSimulator', new_tool_acc, new_tool_inter, new_tool_usab)
        else:
            new_tool_data = copy.deepcopy(self.tools_data_orig)
        
        old_tools = ['MBSE', 'LFSystemSimulator', 'CAD', 'IDE', 'ECAD', 'FEM', 'CurcuitSimulation']
        for tool in old_tools:
            old_tool_acc = round(row[f"T_Acc ({tool})"], 4)
            old_tool_inter = round(row[f"T_I ({tool})"], 4)
            new_tool_data = update_tools(new_tool_data, tool, old_tool_acc, old_tool_inter)
        
        with open(os.path.join(self.folder, "tools.json"), "w") as f:
            json.dump(new_tool_data, f, indent=4)
            


        

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    sim = HypercubeSampleRun(
        folder = 'Architecture/Hypercube',

        n_sim_runs = 50,
        inital_seed=None,

    )

    sim.run_sim()