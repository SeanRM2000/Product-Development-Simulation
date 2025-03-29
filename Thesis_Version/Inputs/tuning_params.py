# only use step sizes that hit every full hour (for handeling end of workday)
step_size = 0.1 # h



###### fast fine tuning
effort_factor = 2
physical_effort_factor = 0.7
######

# work days / hours
work_days_per_week = 5
work_hours_per_day = 8


# Upper limit of knowledge scale
upper_limit_knowledge_scale = 3
upper_interface_severity_limit = 9


### fine tuning overall quality
alpha_goodness = 50
beta_goodness = 10


######### Knowledge
penalty_for_outdated_info = 1

# expertise, efficiency calculation
scaling_factor_excess_knowledge = 0.2
use_knowledge_weight = True

use_uncertainty_for_validation = False

# Consultation on expertise
problem_rate_factor = 1

learning_efficiency_collaboration = 1.2 # complexity per hour
learning_efficiency_consultation = 0.5
learning_efficiency_knowledge_base = 0.3 # complexity per hour
familiarity_increase_rate = 0.01 # % of effort spent

consultation_effort_min = 0.5 # h
consultation_effort_max = 1 # h
consultation_effort_average = (consultation_effort_min + consultation_effort_max) / 2

collaboration_effort_min = 2 # h
collaboration_effort_max = 3 # h
collaboration_effort_average = (collaboration_effort_min + collaboration_effort_max) / 2

importance_reduction_factor_for_external_expert = 0.2 # lowers importance for experts that are not in own team

# Rework
feasibility_detection_rate_factor = 0.5 # reduces the detection rate of feasibility problems

# Knowledge Base Latency
knowledge_base_latency_min = 0.5 # h
knowledge_base_latency_max = 1 # h
knowledge_base_latency_average = (knowledge_base_latency_min + knowledge_base_latency_max) / 2

# Assignment time (Managers)
assignment_time_min = 0.1 # h
assignment_time_max = 0.5 # h

# verification effort
verification_effort_factor = 1.5 # percent of effort that is added for verification of information (systems engineers verifying interfaces)

# information handling times (receiveing and sharing) --> Unit: h/h (effort for info handeling depends on how much info is received which depends on the amount of effort that was used to create it)
info_handling_time_max = 0.3
info_handling_time_min = 0.2

info_need_rate_factor = 0.2
base_info_exchange_propability = 0.1
info_exchange_propability_factor = 0.5

######### Process Settings
allow_activity_overlap = False

supplier_effort_factor = 0.5 # decreased effort for supplier activities
testing_increase_factor_systems = 2

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
random_task_network = False
random_task_times = False

fully_linear_tasks = True # if True ignores task_parallelization and sets everything to 0

### Task Time generation
# Static Times:
nominal_task_effort = 2 # h
# Random Times:
min_task_effort = 2 # h
max_task_effort = 6 # h


# Non-random Task Network Generation
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