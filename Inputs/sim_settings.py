
# test data
use_test_data = True


# create noise tasks based on the availability of agents (False: availability will not be considered)
simulate_noise = False
noise_importance = 0.1

# only use step sizes that hit every full hour (for handeling end of workday)
step_size = 0.1 # h


# Monte Carlo
max_sim_runs = 100
check_stability = True
stability_criteria = 0.01 # % 
stability_interval = 25


# For plotting and results
output_learning = False
show_plots = True
summarize_gantt_chart = True ####### not yet implemented
use_moving_average = True
moving_average_plots = 168 # h
include_noise_in_results = True
include_noise_in_effort_breakdown = False
include_idle_in_effort_breakdown = False
split_plots = 'teams' # overall, profession, teams


