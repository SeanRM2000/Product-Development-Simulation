# create noise tasks based on the availability of agents
simulate_noise = False

# fixed assignments --> requires only one responsible person --> skips assignment tasks during simulation
fixed_assignments = True

# For plotting and results
use_only_first_rep_for_sync = False
output_learning = False
output_quality = False
output_utils = False
summarize_gantt_chart = True ####### not yet implemented
use_moving_average = True
moving_average_plots = 750 # h
include_noise_in_util = False
include_noise_in_effort_breakdown = False
include_idle_in_effort_breakdown = False
split_plots = 'teams' # overall, profession, teams
plot_applied_effort = True