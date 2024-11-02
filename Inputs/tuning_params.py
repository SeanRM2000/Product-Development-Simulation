
# Upper limit of knowledge scale
upper_limit_knowledge_scale = 3



######### Knowledge

# expertise, efficiency calculation
scaling_factor_excess_knowledge = 0.2
use_knowledge_weight = True

# Consultation on expertise
consultation_effectiveness = 0.4 # complexity per hour
knowledge_retention_expert_consultation = 0.1 # %
knowledge_retention_knowledge_base = 0.05 # %



######### Task Network

# randomization of task network generation
random_task_network = True
random_task_times = False

# Non-random Task Network
nominal_task_effort = 4 # h
fully_linear_tasks = False
task_parallelization = { # Task concurrency to reduce the number of 
    'Definition': 0.5,
    'Design': 0.3,
    'Testing': 0.1,
    'Integration': 0.2
}

# Random Task Network Generation
min_task_effort = 2 # h
max_task_effort = 6 # h
reconnect_probability = 0.1
max_in = 8
max_out = 5
overlap_prob = {# Concurrency of tasks of the same activity (Probability to generate a parallel task)
    'Definition': 0.25,
    'Design': 0.15,
    'Testing': 0.05,
    'Integration': 0.1
}