# only use step sizes that hit every full hour (for handeling end of workday)
step_size = 0.1 # h

# work days / hours
work_days_per_week = 5
work_hours_per_day = 8


# Upper limit of knowledge scale
upper_limit_knowledge_scale = 3
upper_interface_severity_limit = 9


######### Knowledge

# expertise, efficiency calculation
scaling_factor_excess_knowledge = 0.2
use_knowledge_weight = True

# Consultation on expertise
consultation_effectiveness = 0.4 # complexity per hour
knowledge_retention_expert_consultation = 0.1 # %
knowledge_retention_knowledge_base = 0.05 # %

consultation_effort_min = 1 # h
consultation_effort_max = 2 # h
importance_reduction_factor_for_external_expert = 0.2 # lowers importance for experts that are not in own team


# Knowledge Base Latency
knowledge_base_latency_max = 1.5 # h
knowledge_base_latency_min = 0.5 # h


# Assignment time (Managers)
assignment_time_min = 0.1 # h
assignment_time_max = 0.5 # h


# information handling times (receiveing and sharing) --> Unit: h/h (effort for info handeling depends on how much info is received which depends on the amount of effort that was used to create it)
info_handling_time_max = 1.0 ######################### reduce
info_handling_time_min = 0.7


######### Process Settings
allow_activity_overlap = False

component_effort_increase_factor = 4  # increased effort for testing and prototyping

# task prioritization and selection
prioritization_weights = {
    'rank_pos_weight': 0.4,
    'complexity': 0.4,
    'req_importance': 0.2
    }
urgency_factor = 0.005 # percentage importance increace / h (0.0021 --> ~5% per day)
noise_importance = 0

######### Task Network

# randomization of task network generation
random_task_network = True
random_task_times = False

### Task Time generation
# Static Times:
nominal_task_effort = 6 # h
# Random Times:
min_task_effort = 2 # h
max_task_effort = 6 # h


# Non-random Task Network Generation
fully_linear_tasks = True # if True ignores task_parallelization and sets everything to 0
task_parallelization = { # Task concurrency to reduce the number of paths on critical path
    'System_Design': 0.5,
    'LF_System_Simulation': 0,
    'Design': 0.3,
    'Component_Simulation': 0,
    'Virtual_Integration': 0.2,
    'HF_System_Simulation': 0,
    'Prototyping': 0,
    'Testing': 0,
}

# Random Task Network Generation
reconnect_probability = 0.1
max_in = 8
max_out = 5
overlap_prob = {# Concurrency of tasks of the same activity (Probability to generate a parallel task)
    'System_Design': 0.33,
    'LF_System_Simulation': 0.1,
    'Design': 0.2,
    'Component_Simulation': 0.05,
    'Virtual_Integration': 0.2,
    'HF_System_Simulation': 0.1,
    'Prototyping': 0,
    'Testing': 0.05,
}