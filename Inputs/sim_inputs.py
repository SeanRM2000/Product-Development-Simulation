
# work days / hours
work_days_per_week = 5
work_hours_per_day = 8


######### Knowledge
initial_req_knowledge = 0.2
inital_design_knowledge = 0.4


######### Process Settings

# triangle distributions for activity effort calculation (h per complexity)
tri_distribution = {
    'System_Design': (8, 15, 12),
    'LF_System_Simulation': (4, 10, 8),
    'Design': (70, 100, 80),
    'Component_Simulation': (20, 40, 35),
    'Virtual_Integration': (2, 5, 4),
    'HF_System_Simulation': (3, 5, 4),
    'Prototyping': (6, 10, 8),
    'Testing': (5, 9, 7),
}

# Learning Factors to reduce effort after repeating tasks (Wright model) --> doubling number of repetetions leads to 1-x reduction 
learning_factors = {
    'System_Design': 0.6,
    'LF_System_Simulation': 0.85,
    'Design': 0.8,
    'Component_Simulation': 0.85,
    'Virtual_Integration': 0.6,
    'HF_System_Simulation': 0.85,
    'Prototyping': 0.98,
    'Testing': 0.99,
    }

# Concurrency to predecessor activites (0: Sequential, 1: Parallel) --> all activities that are not here are sequential
allow_activity_overlap = False
activity_overlap = {
    'System_Design': ('creation', 0.5),
    'Design': ('quantification', 0.5), 
    'Virtual_Integration': ('quantification', 0.3),
    'Prototyping': ('quantification', 0)
} 


######## Consultation
consultation_effort_min = 0.5 # h
consultation_effort_max = 1.5 # h
delay_factor_for_external_expert = 0.2 # extra time for experts that are not in own team




######### Manager Settings

# Assignment time
assignment_time_min = 0.1 # h
assignment_time_max = 0.5 # h

# task prioritization and selection
prioritization_weights = {
    'rank_pos_weight': 0.5,
    'complexity': 0.4,
    'req_importance': 0.1
    }
urgency_factor = 0.005 # percentage importance increace / h (0.0021 --> ~5% per day)


######### Knowledge Base

# Knowledge Base Latency (worst case, most likely, best case)
knowledge_base_latency_wc = 1 # h
knowledge_base_latency_ml = 0.5 # h
knowledge_base_latency_bc = 0.2 # h