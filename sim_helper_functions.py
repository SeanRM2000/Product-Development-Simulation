
import numpy as np

from Inputs.tuning_params import *
from Inputs.sim_settings import *

def convert_hours_to_ymd(hours, remainder=False):
    total_days = hours / 24
    years = int(total_days / 365)
    remaining_days = total_days % 365
    months = int(remaining_days / 30)
    days = remaining_days % 30
    remaining_hours = round((days - int(days)) * 24, 1)
    if remainder:
        return years, months, int(days), remaining_hours
    else:
        return years, months, round(days)


def calc_efficiency_competency(knowledge_req, expertise):
    # ak: Agent knowledge vector (expertise)
    # kr: Task knowledge requirement vector
    # dc: Agent digital competency
    # dt: Tool required for task
    # se: scaling factor for excess knowledge
    se = scaling_factor_excess_knowledge

    ak = expertise
    kr = knowledge_req

    # relevant knowledge items
    k_rel = [i for i in range(len(kr)) if kr[i] > 0]

    # knowledge with higher requirement has a higher weight in the overall competency
    if use_knowledge_weight:
        efficiency = sum((kr[i] + se * (ak[i] - kr[i]) if ak[i] > kr[i] else ak[i]) for i in k_rel) / sum(kr[i] for i in k_rel)
        compompetency = sum(min(ak[i], kr[i]) for i in k_rel) / sum(kr[i] for i in k_rel)
    else:
        efficiency = sum((kr[i] + se * (ak[i] - kr[i]) if ak[i] > kr[i] else ak[i]) / kr[i] for i in k_rel) / len(k_rel)
        compompetency = sum(min(1, ak[i] / kr[i]) for i in k_rel) / len(k_rel)
    
    # problem probability
    problem_rate = sum([max(0, (1 - ak[i] / kr[i])) for i in k_rel]) / len(k_rel)
    if problem_rate > 0:
        problem_probability = 1 - np.exp(-problem_rate * step_size)
    else:
        problem_probability = 0

    return 1 / efficiency, compompetency, problem_probability


def calc_knowledge_gain(inital_level, effort, complexity, expert_level=1):
    if inital_level > expert_level:
        raise ValueError('Inital knowledge level is higher than expert knowledge')
    new_knowledge_level = expert_level / (1 + (expert_level / inital_level - 1) * np.exp(-consultation_effectiveness * effort / complexity))
    return new_knowledge_level - inital_level


def consitency_check():
    # all knowledge domains same?
    # no dublicate team responsibilities 
    # responsibilies and org capabilities match activities and architecture and fully covered (at least one responsibility)
    # digital literacy and org capabilities match
    # single decision makers??? --> not yet clear if this is needed
    # check if task times and step size are compatible (noise creation events)
    
    # raise value errors
    
    pass


def interpolate_knowledge_base_completeness(list, knowledge_level):
    index = knowledge_level - 1
    if index.is_integer():
        return list[int(index)]
    else:
        lower_index = int(index)
        upper_index = lower_index + 1
        fraction = upper_index - lower_index
        return list[lower_index] + fraction * (list[upper_index]- list[lower_index])
        