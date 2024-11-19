# work days / hours
work_days_per_week = 5
work_hours_per_day = 8


######### Process Settings
allow_activity_overlap = False

component_effort_increase_factor = 4

# task prioritization and selection
prioritization_weights = {
    'rank_pos_weight': 0.4,
    'complexity': 0.4,
    'req_importance': 0.2
    }
urgency_factor = 0.005 # percentage importance increace / h (0.0021 --> ~5% per day)


