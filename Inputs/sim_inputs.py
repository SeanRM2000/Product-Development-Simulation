
# work days / hours
work_days_per_week = 5
work_hours_per_day = 8


######### Knowledge
initial_req_knowledge = 0.2
inital_design_knowledge = 0.4


######### Process Settings

# triangle distributions for activity effort calculation
tri_distribution = {
    'Definition': [20, 50, 30],
    'Design': [110, 150, 130],
    'Integration': [70, 100, 80],
    'Testing': [80, 100, 90]
}

# Learning Factors
learning_factors = {
    'Definition': 0.4,
    'Design':  0.2,
    'Integration': 0.1,
    'Testing': 0
    }

# Concurrency between activites (0: Sequential, 1: Parallel) 
activity_overlap = {
    'def_def': 0.5, 
    'def_des': 0.5, 
    'des_test': 0, ### overlapped testing is problamatic with rework
    'test_int': 0, 
    'int_test': 0
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