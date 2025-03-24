
import numpy as np
import os
import json
import warnings

from Inputs.tuning_params import *
from Inputs.sim_settings import *



def consitency_check(folder):
    # Not everything is checked:
    # all knowledge domains same?
    # no dublicate team responsibilities 
    # responsibilies and org capabilities match activities and architecture and fully covered (at least one responsibility)
    # digital literacy and org capabilities match
    # single decision makers??? --> not yet clear if this is needed
    # check if task times and step size are compatible (noise creation events)
    
    print('Consitency Check Started:')
    
    if fixed_assignments:
        if not check_unique_responsibilities(folder):
            print('    All responsibilities are unique: True')
        else: 
            raise ValueError('Not all responsibilities are unique.')
    
    
    print('Consitency Check finished!\n')
        
    

def check_unique_responsibilities(folder):
    """Checks if every responsibility (activity, architecture element) exists only once in the given JSON structure."""
    responsibility_tracker = {}
    duplicates_found = False

    def process_team(team):
        """Recursively processes a team, checking responsibilities of the manager and members."""
        # Check manager's responsibilities
        manager = team.get("Manager", {})
        if manager:
            process_responsibilities(manager.get("responsibilities", {}), manager["name"])

        # Check members' responsibilities
        for member in team.get("Members", []):
            process_responsibilities(member.get("responsibilities", {}), member["name"])

        # Recursively process subteams
        for subteam in team.get("Subteams", []):
            process_team(subteam)

    def process_responsibilities(responsibilities, assignee):
        """Processes and checks unique responsibilities."""
        nonlocal duplicates_found 
        for activity, elements in responsibilities.items():
            for element in elements:
                key = (activity, element)
                if key in responsibility_tracker:
                    print(f"Duplicate responsibility found: {key} assigned to both {responsibility_tracker[key]} and {assignee}")
                    duplicates_found = True
                else:
                    responsibility_tracker[key] = assignee

    file_path = (folder + '/organization.json') if folder else 'Inputs/test_data/test_organization.json'
    with open(file_path, "r") as file:
        org_data = json.load(file)
    
    # Start processing the organization from the top-level team
    project_team = org_data.get("Project Team", {})
    process_team(project_team)

    return duplicates_found






def convert_hours_to_ymd(hours, remainder=False):
    total_days = hours / 24
    years = int(total_days / 365)
    remaining_days = total_days % 365
    months = int(remaining_days / 30.42)
    days = remaining_days % 30
    remaining_hours = round((days - int(days)) * 24, 1)
    if remainder:
        return years, months, int(days), remaining_hours
    else:
        return years, months, round(days)



def calc_efficiency_competency(knowledge_req, expertise, digital_literacy=None, tool_complexity=None, tool_productivity=None):
    # ak: Agent knowledge vector (expertise)
    # kr: Task knowledge requirement vector
    # dc: Agent digital competency
    # tp: tool producttivity
    # tc: Tool complexity
    # se: scaling factor for excess knowledge
    se = scaling_factor_excess_knowledge
    ak = expertise
    kr = knowledge_req
    dc = digital_literacy
    tc = tool_complexity 
    tp = tool_productivity

    # relevant knowledge items
    k_rel = [i for i in range(len(kr)) if kr[i] > 0]

    # knowledge with higher requirement has a higher weight in the overall competency
    if use_knowledge_weight:
        efficiency = sum((kr[i] + se * (ak[i] - kr[i]) if ak[i] > kr[i] else ak[i]) for i in k_rel) / sum(kr[i] for i in k_rel)
        competency = sum(min(ak[i], kr[i]) for i in k_rel) / sum(kr[i] for i in k_rel)
    else:
        efficiency = sum((kr[i] + se * (ak[i] - kr[i]) if ak[i] > kr[i] else ak[i]) / kr[i] for i in k_rel) / len(k_rel)
        competency = sum(min(1, ak[i] / kr[i]) for i in k_rel) / len(k_rel)
    
    # problem probability
    problem_rate = problem_rate_factor * sum([max(0, (1 - ak[i] / kr[i])) for i in k_rel]) / len(k_rel) # average missmatch
    if problem_rate > 0:
        problem_probability = 1 - np.exp(-problem_rate * step_size)
    else:
        problem_probability = 0

    # impact of tool
    if not dc or not tc:
        tool_efficiency = tp
    else:
        tool_efficiency = (min(1, dc / tc) + max(0, se * (dc - tc))) * tp

    if not tp:
        overall_efficiency = efficiency
    else:
        overall_efficiency = efficiency * tool_efficiency
    
    return overall_efficiency, competency, problem_probability



def calc_knowledge_gain(inital_level, complexity, expert_level=1, knowledge_base_effectivness=None, additional_factor=1):
    if inital_level > expert_level:
        warnings.warn('Inital knowledge level is higher than expert knowledge, possibly due to two concurrent consultation requests.')
        return 0
    
    if knowledge_base_effectivness: # knowledge
        learning_efficiency = knowledge_base_effectivness * learning_efficiency_knowledge_base
        average_effort = knowledge_base_latency_average
    else:
        learning_efficiency = learning_efficiency_expert_consultation
        average_effort = consultation_effort_average
    
    new_knowledge_level = expert_level / (1 + ( (expert_level / inital_level) - 1) * np.exp(-learning_efficiency * average_effort * additional_factor / complexity) )
    if new_knowledge_level < inital_level:
        if new_knowledge_level - inital_level > 0.01:
            raise ValueError(f'Knowledge level decreased ({new_knowledge_level - inital_level})')
        else:
            warnings.warn(f'Knowledge level decreased ({new_knowledge_level - inital_level}). No change was made)')
            return 0
    else:
        return new_knowledge_level - inital_level



def interpolate_knowledge_base_completeness(list, knowledge_level):
    index = knowledge_level - 1
    if index.is_integer():
        return list[int(index)]
    else:
        lower_index = int(index)
        upper_index = lower_index + 1
        fraction = upper_index - lower_index
        return list[lower_index] + fraction * (list[upper_index]- list[lower_index])
    
    
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print(f"The folder {folder_path} does not exist.")