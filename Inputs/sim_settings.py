
# create noise tasks based on the availability of agents (False: availability will not be considered)
simulate_noise = False
noise_importance = 0

# only use step sizes that hit every full hour (for handeling end of workday)
step_size = 0.1 # h


# For plotting and results
output_learning = False
summarize_gantt_chart = True ####### not yet implemented
use_moving_average = True
moving_average_plots = 336 # h
include_noise_in_results = True
include_noise_in_effort_breakdown = False
include_idle_in_effort_breakdown = False
split_plots = 'teams' # overall, profession, teams


