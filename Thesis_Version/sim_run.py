import random
import time
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
import os
import warnings
import pdb
from scipy.ndimage import uniform_filter1d
import matplotlib as mpl

# Classes
from architecture_graph import ArchitectureGraph
from organization_graph import OrganizationalGraph
from activity_network import ActivityNetwork
from task_network import TaskNetwork
from tools import Tools

# Functions
import sim_helper_functions as help_fnc

# Parameters
from Inputs.sim_settings import *
from Inputs.tuning_params import *


class PDsim:
    def __init__(self, 
                 overall_quality_goal:float=1,
                 folder:str=None, 
                 file_name_extention:str=None,
                 debug=False, debug_interval:int=100, debug_stop:int=None,
                 timeout=30,
                 enable_timeout = True,
                 montecarlo=False, 
                 log_events=False, slow_logs=False,
                 print_status=True,
                 random_seed=None
                 ):
        
        self.log_events = log_events
        self.print_status = print_status
        self.slow_logs = slow_logs
        self.montecarlo = montecarlo
        
        self.timeout = timeout
        self.enable_timeout = enable_timeout
        
        if not self.montecarlo:
            
            self.file_name_extention = file_name_extention
            if folder and 'Architecture/Inputs/' in folder:
                self.save_folder = folder.replace('Inputs', 'Outputs')
            else:
                # timestamp and folder for results and output files
                timestamp = time.time()
                dt_object = datetime.datetime.fromtimestamp(timestamp)
                self.formatted_time = dt_object.strftime("%Y-%m-%d_%H-%M-%S")
                self.save_folder = 'sim_runs/single_run_at_' + self.formatted_time
            os.makedirs(self.save_folder, exist_ok=True)
            
            self.init_start_time = time.time()
            # debugging
            self.debug = debug
            self.debug_interval = debug_interval
            self.debug_stop = debug_stop
            
            # random seed
            if random_seed:
                random.seed(random_seed)
                np.random.seed(random_seed)
        
        else:
            self.log_events = False
            random_seed = None
                

                
        # Consitency checks of architecture and organization
        if not self.montecarlo:
            help_fnc.consitency_check(folder)
        
        
        # project and org definition
        self.overall_quality_goal = overall_quality_goal
        self.architecture_class = ArchitectureGraph(folder=folder)
        self.architecture = self.architecture_class.architecture
        
        tools = Tools(architecture=self.architecture, folder=folder)
        self.knowledge_base = tools.knowledge_base
        self.tools = tools.tool_list
        
        
        self.org_network = OrganizationalGraph(architecture=self.architecture_class, folder=folder)
        
        self.activity_network_class = ActivityNetwork(self.architecture_class, tools, self.org_network, folder=folder)
        self.activity_network = self.activity_network_class.activity_graph

        self.task_network_class = TaskNetwork(
            self.activity_network_class, 
            self.architecture_class,
            randomize_structure=random_task_network, 
            randomize_task_times=random_task_times
        )

        self.task_network = self.task_network_class.task_graph
        
        
        # log file
        if self.log_events and not self.montecarlo:
            self.start_sim_log()
        
        
        # intervall for noise creation based on nominal task time
        if simulate_noise:
            self.noise_creation_interval = {}
            if random_task_times:
                interval = (min_task_effort + max_task_effort) / 2
            else:
                interval = nominal_task_effort
            for agent in self.org_network.all_agents:
                self.noise_creation_interval[agent] = interval
        else:
            self.noise_creation_interval = None
        
        # product node to track overall completion
        self.overall_product_node = self.architecture_class.root_node
        
        # start task is ready
        first_tasks = [task for task in self.task_network.nodes if not any(self.task_network.predecessors(task))]
        self.tasks_ready = set(first_tasks)

        # trackers for simulation results with inital values (0)
        self.time_points = [0]     
        self.cost_tracker = [0]
        self.cost_tracker_with_idle = [0]
        self.effort_tracker = [0]
        
        self.total_rework_effort = 0
        self.total_work_effort = 0
        self.total_technical_work_effort = 0
        self.total_technical_rework_effort = 0
        self.total_work = 0
        self.total_work_development_work = 0
        self.total_development_work_effort = 0
        
        self.effort_breakdown = {}
        self.effort_backlog_agents = {}
        self.personnel_tracker = {}
        self.active_agents = {}
        for agent in self.org_network.all_agents:
            self.personnel_tracker[agent] = ['Idle']
            self.effort_backlog_agents[agent] = [0]
            self.effort_breakdown[agent] = {}
            self.active_agents[agent] = [0]
        
        self.activity_rework_counter = {}
        self.gantt_tracker = {}
        for activity in self.activity_network.nodes:
            self.activity_rework_counter[activity] = 0
            self.gantt_tracker[activity] = [('Not Started', 0)]
            
        self.last_consitency_check = None
        self.overall_consitency = None
            

        if not self.montecarlo:
            self.init_time = time.time() - self.init_start_time
            
            print('Initialization done!')



    #########################################################################################################################################################
    #### Simulation Run #####################################################################################################################################
    #########################################################################################################################################################
    
    def sim_run(self):
        
        start_time = time.time()
        self.global_clock = 0
        
        # execution until product is fully developed
        while self.architecture.nodes[self.overall_product_node]['completion'] != True:

            # check for end of workday or weekend
            if self.global_clock != 0:
                self.check_end_of_work_day_weekend()
            
            # random chance to create a noise event (depending on availability of agents)
            if self.noise_creation_interval:
                self.create_noise()

            # assign tasks that are ready
            self.create_assignment_tasks()
            
            # reassignment of tasks? --> currently no dynamic responsibilities, therefore not needed
            
            
            # check assigned tasks, select one, and work on it
            completed_tasks = self.work_on_tasks()
            
            # complete finised work (events)
            self.complete_work(completed_tasks)
            
            # time step
            self.global_clock += step_size
            
            # collect all data for trackers
            self.collect_data()
            
            # timer status update
            if not self.montecarlo and not self.log_events and self.print_status:
                print(f'\rSimulation running... ({(time.time() - start_time):.1f} s)', end="", flush=True)
            
            # timeout error
            if self.enable_timeout and time.time() - start_time > self.timeout:
                raise TimeoutError(f'Simulation timed out after {self.timeout}s. Endless loop likely.')
            
            # debugging stop
            if not self.montecarlo and self.debug:
                self.check_debugging()

                
        # Sim results
        if self.montecarlo is False:
            self.sim_time = time.time() - start_time
            self.results()
        else:
            return self.results()
            
    
    
    #########################################################################################################################################################
    #### Working on Tasks ###################################################################################################################################
    #########################################################################################################################################################
              
    
    def work_on_tasks(self):       
        tasks_to_work_on = self.select_tasks_to_work_on()
        
        completed_tasks = {}
        # work on tasks 
        for agent, agent_task in tasks_to_work_on.items():
            data = self.org_network.get_agent(agent)

            task_info = data['task_queue'][agent_task]
            
            if task_info['task_type'] == 'Receive_Information' and task_info['additional_info'].get('Compatibility_Check', False):
                data['state'] = 'Check_Interface_Compatibility'
                data['state_additional_info']['verification_to_total_effort'] = task_info['additional_info']['verification_effort'] / task_info['inital_effort']
            else:
                data['state'] = task_info['task_type']
                data['state_additional_info'] = {}
            
            data['task_in_work'] = agent_task
            
            # active associated technical task
            if task_info['task_type'] == 'Technical_Work':
                data['technical_task'] = agent_task
                data['tool_in_use'] = task_info['tool']
            elif task_info['additional_info']:
                data['technical_task'] = task_info['additional_info']['task']
                data['tool_in_use'] =  task_info['tool']
            else:
                data['technical_task'] = ''
                data['tool_in_use'] = ''
            
             
            # check for technical problem or feasibility problem
            if task_info['task_type'] == 'Technical_Work': 
                # technical problem
                if self.task_network.nodes[agent_task]['activity_type'] in {'System_Design', 'Design'}:
                    if task_info['additional_info']['problem_probability'] > random.random():
                        self.technical_problem_with_general_knowledge(agent, agent_task)
                        continue # dont reduce effort for problem (causes problems with iteration)
                
                    # check feasibility
                    if self.check_feasibility_problem(agent, agent_task):
                        continue
                    
                ####### tool problems could also be added here
            
            # update remaining effort
            task_info['remaining_effort'] -= step_size
            
            if task_info['remaining_effort'] <= 0:
                completed_tasks[agent] = agent_task
        
        return completed_tasks
           
    
    def check_feasibility_problem(self, agent, task):
        activity = self.task_network.nodes[task]['activity_name']
        architecture_element = self.task_network.nodes[task]['architecture_element']
        
        # skip overall product node (feasibility of 1)
        if architecture_element == self.overall_product_node:
            return
        
        definition_quality = self.architecture.nodes[architecture_element]['definition_quality']
        
        
        # calculate probability to detect problem
        tool = self.activity_network.nodes[activity]['tool']
        tool_acc = 0 if self.tools[tool]['accuracy'] is None else self.tools[tool]['accuracy']
        digital_literacy = self.calc_EngTool_effectivness(agent, tool, use_excess=False)
        competency = self.org_network.get_agent(agent)['task_queue'][task]['additional_info']['competency']
        
        accuracy = tool_acc * digital_literacy + competency
        
        nominal_effort = self.org_network.get_agent(agent)['task_queue'][task]['nominal_effort']
        real_effort = self.org_network.get_agent(agent)['task_queue'][task]['inital_effort']
        
        detection_rate = feasibility_detection_rate_factor * (1 - definition_quality) * accuracy
        detection_probability = 1 - np.exp(-detection_rate * nominal_effort / real_effort * step_size)
        
        if detection_probability < random.random():
            return False
        else:
            
            if self.log_events:
                self.event_logger(f'{agent} encountered a Feasibility Problem (Feas: {definition_quality}) with {architecture_element}. {activity} has been paused.')
            
            parent_element = self.architecture_class.get_parent(architecture_element)
            parent_element_agent = self.architecture.nodes[parent_element]['responsible_designer']
            
            ###### technically would have to add some rework to the system design activity
            
            # pause activity
            self.activity_network.nodes[activity]['activity_status'] = 'Interface or Feasibility Problem'
            
            perceived_feasibility = 1 # not used currently
            
            if self.check_if_idle(agent) and self.check_if_idle(parent_element_agent):
                self.start_collaboration(task, parent_element_agent, agent, architecture_element, perceived_feasibility, 'Feasibility')
            else:
                self.add_request(agent, parent_element_agent, type='Collaboration', info={'task': task, 'element': architecture_element, 'sub_type': 'Feasibility', 'perceived_feasibility': perceived_feasibility})
            
            return True
        
        
    
    def select_tasks_to_work_on(self):
        tasks_to_work_on = {}
        possible_tasks = {}
        
        all_agents = self.org_network.all_agents
        random.shuffle(all_agents) # shuffled to ensure no favoratism towards any agent
        for agent in all_agents:
            data = self.org_network.get_agent(agent)
            
            # select possible task to work on (excluding technical tasks that have technical problems)
            possible_tasks[agent] = [task for task in list(data['task_queue'].keys()) if ((data['task_queue'][task]['task_type'] == 'Technical_Work' and 
                      self.task_network.nodes[task]['task_status'] not in {'Information Needed', 'Technical Problem'} and
                      self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]['activity_status'] != 'Interface or Feasibility Problem') or
                      data['task_queue'][task]['task_type'] != 'Technical_Work')]
            
            # check collaboration tasks
            collaboration_task = []
            for task in possible_tasks[agent]:
                task_info = data['task_queue'][task]
                if task_info['task_type'] in {'Collaboration', 'Consultation', 'Provide_Consultation'}:
                    collaboration_task.append(task)
            
            if len(collaboration_task) > 1: # consitency check
                raise ValueError(f'Multiple Collaboration Tasks assigned to {agent}: {collaboration_task}')
            elif len(collaboration_task) == 1:
                tasks_to_work_on[agent] = collaboration_task[0]
                del possible_tasks[agent]

        # check agents that have requests and are not working --> check if requests have higher priority than tasks
        agents_with_new_collab = []
        for agent, tasks in list(possible_tasks.items()):
            if agent not in possible_tasks: # skip if was removed
                continue
            
            requests = self.org_network.get_agent(agent)['collab_requests']
            if requests and not any(self.org_network.get_agent(agent)['task_queue'][task]['remaining_effort'] 
                                    < self.org_network.get_agent(agent)['task_queue'][task]['inital_effort']
                                    for task in tasks):
                
                # requestors are able to collaborate
                filtered_requests = [r for r in requests if r['requestor'] in possible_tasks]
                        
                if filtered_requests:
                    random.shuffle(filtered_requests)
                    request_priorities = [(request, self.calc_priority(agent, request=request)) for request in filtered_requests]
                    sorted_requests = sorted(request_priorities, key=lambda r: r[1], reverse=True)
                    
                    # check for more important task
                    if tasks:
                        random.shuffle(tasks) # shuffle to ensure randomness in case of tie
                        task_priorities  = [(task, self.calc_priority(agent, task=task)) for task in tasks]
                        sorted_tasks = sorted(task_priorities, key=lambda t: t[1], reverse=True)
                        if sorted_tasks[0][1] > sorted_requests[0][1]:
                            continue # request not accepted right now
                    
                    selected_request = sorted_requests[0][0]
                    requestor =  selected_request['requestor']
                    
                    # Event log
                    if self.log_events:
                        self.event_logger(f'{selected_request['request_type']} request from {requestor} accepted by {agent}')

                    # delete element from list
                    req_id = selected_request['req_id']
                    for request in list(self.org_network.get_agent(requestor)['pending_collab_requests']):
                        if request['req_id'] == req_id:
                            self.org_network.get_agent(requestor)['pending_collab_requests'].remove(request)
                        
                    self.org_network.get_agent(agent)['collab_requests'].remove(selected_request)
                    
                    
                    match selected_request['request_type']:
                        case 'Consultation':
                            self.start_consultation(selected_request['task'], requestor, agent, selected_request['knowledge_item'], selected_request['knowledge_level'])
                  
                        case 'Collaboration':
                            if selected_request['sub_type'] == 'Feasibility':
                                self.start_collaboration(selected_request['task'], requestor, agent, selected_request['element'], selected_request['perceived_feasibility'], selected_request['sub_type'])
                            else:
                                self.start_collaboration(selected_request['task'], requestor, agent, selected_request['element'], selected_request['rework_percent'], selected_request['sub_type'])
                    
                    
                    agents_with_new_collab.extend([requestor, agent])
                    del possible_tasks[requestor]
                    del possible_tasks[agent]
       
        # start requested collaborations
        for agent in agents_with_new_collab:
            for task in self.org_network.get_agent(agent)['task_queue']:
                task_info = self.org_network.get_agent(agent)['task_queue'][task]
                if task_info['task_type'] in {'Collaboration', 'Consultation', 'Provide_Consultation'}:
                    tasks_to_work_on[agent] = task
                    break
        
        # continue working on started work
        for agent, tasks in list(possible_tasks.items()):
            for task in tasks:
                task_info = self.org_network.get_agent(agent)['task_queue'][task]
                if task_info['remaining_effort'] < task_info['inital_effort']:
                    tasks_to_work_on[agent] = task
                    del possible_tasks[agent]
                    break
        
        # check information requests
        for agent, tasks in list(possible_tasks.items()):
            agent_info = self.org_network.get_agent(agent)

            # check if there already is a infosharing task skip this agent
            info_task_exists = False
            for task, task_data in agent_info['task_queue'].items():
                if task_data['task_type'] == 'Share_Information':
                    info_task_exists = True
                    break
            if info_task_exists:
                continue
            
            element = self.org_network.get_agent(agent).get('responsible_element', None)
            if not element:
                continue
            
            if self.architecture_class.get_hierarchical_children(element):
                activity = self.activity_network_class.generate_activity_name(element, 'System_Design')
                activity_type = 'System_Design'
            else:
                activity = self.activity_network_class.generate_activity_name(element, 'Design')
                activity_type = 'Design'
            
            completion_level = self.org_network.get_agent(agent)['product_information_completeness'][element][activity_type]
            req_prio = 0
            for request in agent_info['info_requests']:
                # check if information for request is available
                if round(request['amount'], 6) <= round(completion_level, 6):
                    req_prio += self.calc_priority(agent, request=request)
                
            if req_prio > 0:
                # check for most important task
                if tasks:
                    max_task_prio = max([self.calc_priority(agent, task=task) for task in tasks])
                else:
                    max_task_prio = 0
                        
                if req_prio > max_task_prio:
                    knowledge_base_completeness_level = self.knowledge_base['product_information_completeness'][element]
                    new_info = completion_level - knowledge_base_completeness_level * self.knowledge_base['product_information_consitency'][element]
                        
                    info_amount = self.activity_network.nodes[activity]['effort'] * new_info
                    
                    if self.log_events:
                        self.event_logger(f'Request for information on {element} from {request['requestor']} was accepted by {agent}')
                    
                    task = self.assign_task(agent, task_type='Share_Information', 
                                    info={'task': self.activity_network.nodes[activity]['tasks'][0], 'info_amount': info_amount, 'start_condition_for': []})
                    tasks_to_work_on[agent] = task
                    del possible_tasks[agent]

            
        
        # prioritize task    
        for agent, tasks in possible_tasks.items():
            selected_task = None
            agent_data = self.org_network.get_agent(agent)
            if len(tasks) == 1:
                selected_task = tasks[0]
            elif len(tasks) == 0 and agent_data['task_queue']:
                agent_data['state'] = 'Waiting'
            elif len(tasks) == 0:
                agent_data['state'] = 'Idle'
            else:
                for task in tasks: # priotitize sending rework info back
                    if agent_data['task_queue'][task]['task_type'] == 'Share_Information' and agent_data['task_queue'][task]['additional_info'].get('rework_info', False):
                        selected_task = task
                        break
                
                if not selected_task:
                    random.shuffle(tasks) # shuffle to ensure randomness in case of tie
                    task_priorities  = [(task, self.calc_priority(agent, task=task)) for task in tasks]
                    sorted_tasks = sorted(task_priorities, key=lambda t: t[1], reverse=True)
                    selected_task = sorted_tasks[0][0]
                    
                    # select information receival task if it exist for that task
                    if round(sorted_tasks[0][1], 5) == round(sorted_tasks[1][1], 5):
                        for task in tasks:
                            if agent_data['task_queue'][task]['task_type'] in {'Receive_Information', 'Share_Information'}:
                                if agent_data['task_queue'][task]['additional_info']['task'] == selected_task:
                                    selected_task = task
            
            
            # check if task is technical and newly started: check if information is required
            if (selected_task and agent_data['task_queue'][selected_task]['task_type'] == 'Technical_Work' and 
                agent_data['task_queue'][selected_task]['inital_effort'] == agent_data['task_queue'][selected_task]['remaining_effort']): 
                
                information_task = self.check_if_information_required(selected_task, agent)
                if information_task: # replace selected task if information is required
                    selected_task = information_task
            
            # add selected task to tasks to work on dict
            if selected_task:
                tasks_to_work_on[agent] = selected_task
                
        return tasks_to_work_on
                

    def check_if_information_required(self, task, agent):
        
        def get_interface_complexity_contribution(child, children):
            total_complexity = self.architecture.nodes[child]['technical_complexity']
            interfaces = self.architecture.nodes[child]['interfaces']
            internal_interface_complexity = 0
            for dep_node, edges in interfaces.items():
                for edge in edges:
                    total_complexity += self.architecture.edges[edge]['complexity']
                    if dep_node in children:
                        internal_interface_complexity += self.architecture.edges[edge]['complexity']
                    
            info_percentage = internal_interface_complexity / total_complexity
            return info_percentage
        
        
        activity = self.task_network.nodes[task]['activity_name']
        activity_type = self.task_network.nodes[task]['activity_type']
        architecture_element = self.task_network.nodes[task]['architecture_element']

        receive_from = [] # collects all information
        
        # receive information from predecessor activities (required to start)
        if self.task_network.nodes[task]['first_task'] and not self.org_network.get_agent(agent)['task_queue'][task].get('checked', False):
            self.org_network.get_agent(agent)['task_queue'][task]['checked'] = True
            
            match activity_type:
                
                case 'System_Design' | 'Design': # receive information from parent system design
                    parent_element = self.architecture_class.get_parent(architecture_element)
                    if not parent_element: # overall system node requires no inputs
                        return

                    total_complexity = sum([self.architecture.nodes[child]['development_complexity'] for child in self.architecture_class.get_hierarchical_children(parent_element)])
                    complexity = self.architecture.nodes[architecture_element]['development_complexity']                  
                    
                    receive_from.append((parent_element, self.activity_network_class.generate_activity_name(parent_element, 'System_Design'), complexity / total_complexity))
                    
                    
                case 'LF_System_Simulation' | 'Component_Simulation' | 'HF_System_Simulation': # quantification activities have one predecessor (receive all info)
                    receive_from.append((architecture_element, next(self.activity_network.predecessors(activity)), 1))
                    
                    if activity_type == 'HF_System_Simulation':
                        technical_complexity = self.architecture.nodes[architecture_element]['technical_complexity']
                        development_complexity = self.architecture.nodes[architecture_element]['development_complexity']
                        receive_from.append((architecture_element, self.activity_network_class.generate_activity_name(architecture_element, 'System_Design'), technical_complexity / development_complexity))
                    
                    
                case 'Virtual_Integration': # does not work for partial virtual integration
                    receive_from.append((architecture_element, self.activity_network_class.generate_activity_name(architecture_element, 'System_Design'), 1))
    
                    children = self.architecture_class.get_hierarchical_children(architecture_element)
                    for child in children:
                        info_percentage = get_interface_complexity_contribution(child, children)
                        if info_percentage > 0:
                            if self.architecture_class.get_hierarchical_children(child):
                                receive_from.append((child, self.activity_network_class.generate_activity_name(child, 'Virtual_Integration'), info_percentage))
                            else:
                                receive_from.append((child, self.activity_network_class.generate_activity_name(child, 'Design'), info_percentage))

                
                case 'Prototyping': # receive information from (system-)design or virtual integration activities
                    if self.architecture_class.get_hierarchical_children(architecture_element): # assembly
                        
                        virtual_integration_activity = self.activity_network_class.generate_activity_name(architecture_element, 'Virtual_Integration')
                        if virtual_integration_activity in self.activity_network.nodes:
                            receive_from.append((architecture_element, virtual_integration_activity, 1))
                        else:
                            receive_from.append((architecture_element, self.activity_network_class.generate_activity_name(architecture_element, 'System_Design'), 1))
                            
                        components = self.architecture_class.get_all_components(architecture_element)
                        for component in components:
                            info_percentage = 0
                            components_of_same_subsystem = self.architecture_class.get_hierarchical_children(self.architecture_class.get_parent(component))
                            if len(components_of_same_subsystem) == len(components): # lowest level subsystem
                                info_percentage = get_interface_complexity_contribution(component, components)
                            else: # higher level subsystem that only need interfaces between subsystems 
                                components_of_other_subsystems = [comp for comp in components if comp not in components_of_same_subsystem]
                                info_percentage = get_interface_complexity_contribution(component, components_of_other_subsystems)
                            if info_percentage > 0:
                                receive_from.append((component, self.activity_network_class.generate_activity_name(component, 'Design'), info_percentage))
                                    
                    else: # manufacturing
                        receive_from.append((architecture_element, self.activity_network_class.generate_activity_name(architecture_element, 'Design'), 1))
                

                case 'Testing': # receive information from (system-)design activities
                    technical_complexity = self.architecture.nodes[architecture_element]['technical_complexity']
                    
                    if self.architecture_class.get_hierarchical_children(architecture_element):
                        development_complexity = self.architecture.nodes[architecture_element]['development_complexity']
                        receive_from.append((architecture_element, self.activity_network_class.generate_activity_name(architecture_element, 'System_Design'), technical_complexity / development_complexity))
                    else:
                        total_complexity = technical_complexity + self.architecture.nodes[architecture_element]['total_interface_complexity']
                        receive_from.append((architecture_element, self.activity_network_class.generate_activity_name(architecture_element, 'Design'), technical_complexity / total_complexity))
                        

        # check if information from dependent elements is required
        elif activity_type in  {'System_Design', 'Design'} and not self.task_network.nodes[task]['first_task'] and not self.org_network.get_agent(agent)['task_queue'][task].get('checked', False):
            self.org_network.get_agent(agent)['task_queue'][task]['checked'] = True
            
            info_needed = []
            for dep_element, interfaces in self.architecture.nodes[architecture_element]['interfaces'].items():
                
                if activity_type == 'System_Design': # system design interfaces not implemented
                    pass
                

                else: # Design
                    for edge in interfaces:
                        info_exchange = False
                        
                        #technical_complexity = self.architecture.nodes[architecture_element]['technical_complexity']
                        interface_complexity = self.architecture.edges[edge]['complexity']
                        relative_importance = self.architecture.nodes[edge[1]]['req_importance'] / self.architecture.nodes[architecture_element]['req_importance']
                        dependency_strength = interface_complexity / 7.5 * relative_importance    # 7.5 is maximum interface complexity (5 interfaces with highest severity and max knowledge difference)
                        
                        interface_feasibility = self.architecture.edges[edge]['definition_quality']
                        agent_info_completeness = self.org_network.get_agent(agent)['product_information_completeness'][dep_element][activity_type] # this is their perceived completness
                        
                        info_need = self.activity_network.nodes[activity]['n_completed_tasks'] / self.activity_network.nodes[activity]['num_tasks'] # information need based on progression
                        information_gap = info_need - agent_info_completeness
                        
                        # information need probability
                        if information_gap <= 0:
                            info_need_probability = 0
                        else:
                            info_need_rate = info_need_rate_factor * (1 - interface_feasibility) * dependency_strength * (1 - agent_info_completeness) * (1 + information_gap)
                            nominal_effort = self.org_network.get_agent(agent)['task_queue'][task]['nominal_effort']
                            info_need_probability = 1 - np.exp(-info_need_rate * nominal_effort)
                        
                        # check if information is needed
                        if info_need_probability > random.random():
                            info_needed.append(info_need)
                            info_exchange = True
        
                        else: # use info check probability if no information is needed
                            tool = self.org_network.get_agent(agent)['task_queue'][task]['tool']
                            interoperability = self.tools[tool].get('interoperability', 0)
                            productivity = (self.knowledge_base['productivity'] + self.tools[tool]['productivity']) / 2
                            digital_literacy = self.calc_supportTool_effectivness(agent)
                            
                            propability_info_check = base_info_exchange_propability + interoperability * productivity * digital_literacy * info_exchange_propability_factor
                            if propability_info_check > random.random():
                                info_needed.append(False)
                                info_exchange = True
                                
                        if info_exchange:  
                            technical_complexity_dep = self.architecture.nodes[edge[1]]['technical_complexity']
                            info_percentage = interface_complexity / (technical_complexity_dep + interface_complexity)
                            receive_from.append((edge[1], self.activity_network_class.generate_activity_name(edge[1], 'Design'), info_percentage))
                            
        else:
            return
            
        # information sharing amount
        info_amounts = []
        elements = []
        request_info = []
        for i, (element, originating_activity, fraction) in enumerate(receive_from):
            
            # relative amount of new information
            agent_info_completeness = self.org_network.get_agent(agent)['product_information_completeness'][element][activity_type]
            agent_info_consitency = self.org_network.get_agent(agent)['product_information_consitency'][element][activity_type]
            knowledge_base_completeness = self.knowledge_base['product_information_completeness'][element]
            knowledge_base_info_consitency = self.knowledge_base['product_information_consitency'][element]
            
            new_info = self.calculate_new_information(element, agent, agent_info_completeness, agent_info_consitency, knowledge_base_completeness, knowledge_base_info_consitency)

            # check if more information is needed
            if not self.task_network.nodes[task]['first_task'] and info_needed[i]:
                missing_info = info_needed[i] - knowledge_base_completeness
                if missing_info > 0.0000001:
                    info_amounts.append(missing_info * self.activity_network.nodes[originating_activity]['effort'] * fraction)
                    request_info.append((element, info_needed[i], fraction)) # information will be requested afer the existing information is received
                    
            if new_info > 0:
                info_amounts.append(new_info * self.activity_network.nodes[originating_activity]['effort'] * fraction)
                elements.append(element)
                
            elif new_info == 0 and self.task_network.nodes[task]['first_task']:    
                info_amounts.append(0) # needed for reduced work of virtual integration and prototyping

        
        # reduced work required when reworking depending on amount of new information
        if self.task_network.nodes[task]['first_task'] and activity_type in {'Virtual_Integration', 'Prototyping'}:
            total_information_amount = 0
            new_information_amount = 0
            
            if self.architecture_class.get_hierarchical_children(architecture_element): # skip component prototyping
                for i, (element, originating_activity, fraction) in enumerate(receive_from):
                    information_amount = self.activity_network.nodes[originating_activity]['effort'] * fraction
                    total_information_amount += information_amount
            
                    if activity_type == 'Virtual_Integration':
                        new_information_amount += info_amounts[i]
                    else: # for prototyping full element has to be redone
                        new_information_amount += information_amount if info_amounts[i] > 0 else 0
                
                rework_effort_red = new_information_amount / total_information_amount
                
                if rework_effort_red == 0: # had to bypass this because the rework amount is only dependent opun interconnections between subsystems
                    warnings.warn(f'Effort for {activity} set to zero due to no changed information. It was changed to minimum amount (0.1 for Virtual integration and 0.3 for Prototyping)')
                
                if activity_type == 'Virtual_Integration':
                    min_rework_fraction = 0.1
                else:
                    min_rework_fraction = 0.3
                rework_effort_red = max(min_rework_fraction, rework_effort_red)
                
                self.activity_network.nodes[activity]['second_order_rework_reduction'] = rework_effort_red

                # change effort of first task, because it was already calculated
                self.org_network.get_agent(agent)['task_queue'][task]['remaining_effort'] *= rework_effort_red
                self.org_network.get_agent(agent)['task_queue'][task]['inital_effort'] *= rework_effort_red

                if rework_effort_red == 0: 
                    #print(receive_from)
                    raise ValueError(f'Effort for {activity} set to zero due to no changed information.')
        
        # create information task
        if elements or request_info:
            information_task = self.assign_task(agent, task_type='Receive_Information', info={'task': task, 'info_amount': sum(info_amounts), 'elements': elements, 'request_info': request_info})
        else: 
            information_task = None
            
        return information_task


    def calculate_new_information(self, element, agent, agent_info_completeness, agent_info_consitency, knowledge_base_completeness, knowledge_base_info_consitency):
        agent_correct_information = agent_info_completeness * agent_info_consitency
        knowledge_correct_information = knowledge_base_completeness * knowledge_base_info_consitency
        
        if round(agent_correct_information, 5) == round(knowledge_correct_information, 5): # rounded for float error
            if knowledge_base_completeness < agent_info_completeness:
                new_info = knowledge_base_completeness - agent_correct_information
            else:
                new_info = knowledge_base_completeness - agent_info_completeness
                
        elif round(agent_correct_information, 5) < round(knowledge_correct_information, 5):
            new_info = knowledge_base_completeness - agent_correct_information
            
        else: # knowledge base correct information should always be higher or equal
            if self.log_events:
                self.event_logger(f'        {agent}: Cons: {agent_info_consitency:.3f}, Comp: {agent_info_completeness:.3f}')
                self.event_logger(f'        Knowledge Base: Cons: {knowledge_base_info_consitency:.3f}, Comp: {knowledge_base_completeness:.3f}')
            print(f'{agent}: Cons: {agent_info_consitency:.3f}, Comp: {agent_info_completeness:.3f}')
            print(f'        Knowledge Base: Cons: {knowledge_base_info_consitency:.3f}, Comp: {knowledge_base_completeness:.3f}')
            raise ValueError(f'{agent} has better information on {element} than knowledge base.') # information is always first pushed to KB
        
        
        if new_info < 0:
            if round(new_info, 6) == 0:
                new_info = 0
                warnings.warn(f'New info was a small negativ number ({new_info}). It was set to zero.')
            else:
                raise ValueError(f'New information on {element} for {agent} is negative ({new_info}).')
        
        return new_info
        
        
    def calc_priority(self, agent, task=None, request=None):
        if task:
            task_info = self.org_network.get_agent(agent)['task_queue'][task]
            time_at_assignment = task_info['time_at_assignment']
            importance = task_info['importance']
            if task_info['task_type'] == 'Technical_Work':
                activity = self.task_network.nodes[task]['activity_name']
            elif task_info['task_type'] != 'Noise':
                linked_technical_task = task_info['additional_info']['task']
                activity = self.task_network.nodes[linked_technical_task]['activity_name']
            else:
                activity = None
                
        elif request:
            time_at_assignment = request['time_of_request']
            importance = request['importance']   
            
            linked_technical_task = request['task']
            activity = self.task_network.nodes[linked_technical_task]['activity_name']
            
        if activity:
            activity_completion = self.activity_network.nodes[activity]['n_completed_tasks'] / self.activity_network.nodes[activity]['num_tasks']
        else:
            activity_completion = 0
            
        time_since_assignment = self.global_clock - time_at_assignment
        prio = importance * (1 + urgency_factor * time_since_assignment)
        return prio * (1 + 2*activity_completion)
    
    
    def technical_problem_with_general_knowledge(self, agent, task):
        # change task status
        self.task_network.nodes[task]['task_status'] = 'Technical Problem'
        
        # random sample to select knowledge item that created the problem
        expertise = self.org_network.get_agent(agent)['expertise']
        knowledge_req_vector = self.task_network.nodes[task]['knowledge_req']
        problem_knowledge_item = random.choice([i for i, expertise_level in enumerate(expertise) if expertise_level < knowledge_req_vector[i]])
        req_knowledge_level = knowledge_req_vector[problem_knowledge_item]
        agent_knowledge_level = expertise[problem_knowledge_item]
        
        problem_knowledge_level = random.uniform(agent_knowledge_level, req_knowledge_level)
        
        # Event log
        if self.log_events:
            self.event_logger(f'Technical problem ({self.org_network.knowledge_items[problem_knowledge_item]}; Req: {req_knowledge_level} ({problem_knowledge_level:.2f}) Has: {expertise[problem_knowledge_item]:.2f}) with "{task}" occured for {agent}.')
        
        expert_to_consult = self.find_expert(agent, problem_knowledge_item, problem_knowledge_level, search_criteria='team', only_idle=True)
        
        if expert_to_consult:
            self.start_consultation(task, agent, expert_to_consult, problem_knowledge_item, problem_knowledge_level)
        else: # create knowledge base search task
            
            success = help_fnc.interpolate_knowledge_base_completeness(self.knowledge_base['completeness'][problem_knowledge_item], problem_knowledge_level) > random.random()
            
            if not success: # if information is not available the familiarity is used to determine if the agent checks or not
                if (self.calc_supportTool_effectivness(agent, use_excess=False) * self.org_network.get_agent(agent)['knowledge_base_familiarity'][0] > random.random()):
                    self.check_any_expert_for_consultation(agent, task, problem_knowledge_item, problem_knowledge_level)
                    return
            
            self.assign_task(
                agent, 
                task_type='Search_Knowledge_Base',
                info={'task': task, 'knowledge_item': problem_knowledge_item, 'knowledge_level': problem_knowledge_level, 'success': success}, 
                original_assignment_time=self.org_network.get_agent(agent)['task_queue'][task]['time_at_assignment']
            )
        
        
    def start_collaboration(self, task, requesting_agent, responsible_agent, element, rework_percent, type):
    
        # Event log
        if self.log_events:
            self.event_logger(f'Collaboration because of {type} problem with {element} between {requesting_agent} and {responsible_agent} started.')
        
        # defined here to be equal for both
        effort = random.triangular(collaboration_effort_min, collaboration_effort_max)
        
        # create tasks
        self.assign_task(
            responsible_agent, 
            task_type='Collaboration', 
            info={'task': task, 'requestor': requesting_agent, 'element': element, 'sub_type': type},
            effort=effort
        )
        self.assign_task(
            requesting_agent, 
            task_type='Collaboration', 
            info={'task': task, 'resp_agent': responsible_agent, 'element': element, 'sub_type': type, 'rework_percent': rework_percent},
            effort=effort
        ) 
        
    
    def start_consultation(self, task, agent, expert, knowledge_item, problem_knowledge_level):
        if agent == expert:
            raise ValueError(f'Consultation not possible: agent and expert are same ({agent})')
        
        # Event log
        if self.log_events:
            self.event_logger(f'Consultation on {self.org_network.knowledge_items[knowledge_item]} with {expert} started.')
        
        # defined here to be equal for both
        effort = random.triangular(consultation_effort_min, consultation_effort_max)
        
        # create consultation task for expert and agent
        self.assign_task(
            expert, 
            task_type='Provide_Consultation', 
            info={'task': task, 'agent': agent},
            effort=effort
        )
        self.assign_task(
            agent, 
            task_type='Consultation', 
            info={'task': task, 'knowledge_item': knowledge_item, 'knowledge_level': problem_knowledge_level, 'expert': expert},
            effort=effort
        ) 
        
        

    #########################################################################################################################################################
    #### Tasks Completion Events ############################################################################################################################
    #########################################################################################################################################################  
                
    def complete_work(self, completed_tasks):
        for agent, task in completed_tasks.items():
            agent_data = self.org_network.get_agent(agent)
            task_data = agent_data['task_queue'].get(task, None)
            
            if not task_data: # task was reset and therefore deleted from queue
                continue
            
            # Event log
            if self.log_events:
                self.event_logger(f'"{task}" was completed by {agent}.')
            
            match task_data['task_type']:
                case 'Technical_Work':
                    self.complete_technical_work(agent, task)
                case 'Assign_Task':
                    self.complete_assign_task(agent, task)
                case 'Consultation' | 'Provide_Consultation':
                    self.complete_consultation(agent, task_data['additional_info'])
                case 'Collaboration':
                    self.complete_collaboration(agent,task_data['additional_info'])
                case 'Search_Knowledge_Base':
                    self.complete_search_knowledge_base(agent, task_data)
                case 'Share_Information':
                    self.complete_information_sharing(agent, task_data['additional_info'])
                case 'Receive_Information':
                    self.complete_receive_information(agent, task_data, task)
                    
            # delete from queue
            del self.org_network.get_agent(agent)['task_queue'][task]
    
    
    def complete_technical_work(self, agent, task):
        
        def update_design_quality(element, competency, task_repetitions):
            element_data = self.architecture.nodes[element]
            feasibility = element_data['definition_quality']
            
            if task_repetitions == 1: # use existing product knowledge for first iteration
                existing_product_knowledge = self.knowledge_base['previous_product_knowledge'][element]
                element_data['design_quality'] = feasibility * (existing_product_knowledge + (1 - existing_product_knowledge) * competency)
                element_data['previous_design_quality'] = element_data['design_quality']
            else:
                # use worst perceived quality, 1 if no perceived quality (no gain possible)
                previous_quality = element_data['previous_design_quality']
                perceived_quality = min([value for value in element_data['previous_perceived_quality'].values() if value > previous_quality], default=1) 
                element_data['design_quality'] = previous_quality +  max(0, feasibility - perceived_quality) * competency
            

        # activity information
        activity = self.task_network.nodes[task]['activity_name']
        activity_type = self.task_network.nodes[task]['activity_type']
        
        # architecture element
        architecture_element = self.task_network.nodes[task]['architecture_element']
        
        # agent information
        agent_data = self.org_network.get_agent(agent)
        expertise = agent_data['expertise']
        
        # update task completion
        self.task_network.nodes[task]['repetitions'] += 1
        self.task_network.nodes[task]['task_status'] = 'Completed'
        if not self.task_network.nodes[task]['final_task'] or activity_type == 'Prototyping': # completion is done by final information sharing and prototyping (no info share)
            self.task_network.nodes[task]['completed'] = True
        
        # update activity completion
        self.activity_network.nodes[activity]['n_completed_tasks'] += 1
        if self.activity_network.nodes[activity]['n_completed_tasks'] == self.activity_network.nodes[activity]['num_tasks']:
            self.activity_network.nodes[activity]['activity_status'] = 'Completed'
        
        
        # behavior for different activity types
        match activity_type:
            
            case 'System_Design':
                
                # design quality
                competency = agent_data['task_queue'][task]['additional_info']['competency']
                update_design_quality(architecture_element, competency, self.task_network.nodes[task]['repetitions'])
                
                # log event
                if self.log_events:
                    self.event_logger(f'     System Design Quality of {architecture_element}: {self.architecture.nodes[architecture_element]['design_quality']:.3f}')
                
                # definition quality of children
                children = self.architecture_class.get_hierarchical_children(architecture_element)
                for child in children:
                    # definition quality (feasibility) for children
                    knowledge_req_child = self.architecture.nodes[child]['knowledge_req']
                    competency = help_fnc.calc_efficiency_competency(knowledge_req_child, expertise)[1]
                    product_knowledge = agent_data['product_knowledge'][child]
                    self.architecture.nodes[child]['definition_quality'] = product_knowledge + (1 - product_knowledge) * competency
                    
                    # log event
                    if self.log_events:
                        self.event_logger(f'     Definition Quality of {child}: {self.architecture.nodes[architecture_element]['definition_quality']:.3f}')
    
                    # definition quality of interfaces
                    for dep_element, interfaces in self.architecture.nodes[child]['interfaces'].items():
                        for interface in interfaces:
                            
                            if dep_element in children: # interface is between components or subsystems of the system/subsystem being worked on
                                
                                comp1 = help_fnc.calc_efficiency_competency(self.architecture.nodes[interface[0]]['knowledge_req'], expertise)[1]
                                comp2 = help_fnc.calc_efficiency_competency(self.architecture.nodes[interface[1]]['knowledge_req'], expertise)[1]
                                pk1 = agent_data['product_knowledge'][interface[0]]
                                pk2 = agent_data['product_knowledge'][interface[1]]
                                # self.architecture.edges[interface]['definition_quality'] = ((pk1 + (1 - pk1) * comp1) + (pk2 + (1 - pk2) * comp2)) / 2
                                self.architecture.edges[interface]['definition_quality'] = (pk1 + pk2) / 2
                                
                                # log event
                                if self.log_events:
                                    self.event_logger(f'     Interface Feasibility from {interface[0]} to {interface[1]}: {self.architecture.edges[interface]['definition_quality']:.3f}')
                                
                            else: # system definition quality ? --> is the high level feasibility of the actual interfaces 
                                  # --> needed to defne the actual feasibility of an interface --> enable info exchange on system level
                                pass


            case 'Design':
                
                # design quality
                if self.architecture.nodes[architecture_element].get('procure'): # procured parts are not technically designed 
                    self.architecture.nodes[architecture_element]['design_quality'] = 1
                else:
                    competency = agent_data['task_queue'][task]['additional_info']['competency']
                    update_design_quality(architecture_element, competency, self.task_network.nodes[task]['repetitions'])
                
                # update overall quality
                self.architecture.nodes[architecture_element]['overall_quality'] = self.architecture.nodes[architecture_element]['design_quality']
                
                # log event
                if self.log_events:
                    self.event_logger(f'     Design Quality of {architecture_element}: {self.architecture.nodes[architecture_element]['design_quality']:.3f}')
                
                # interface quality
                for edges in self.architecture.nodes[architecture_element]['interfaces'].values():
                    for edge in edges:
                        product_knowledge_of_interface = agent_data['product_knowledge'][edge[1]]
                        info_completeness = self.org_network.get_agent(agent)['product_information_completeness'][edge[1]][activity_type]
                        info_consitency = self.org_network.get_agent(agent)['product_information_consitency'][edge[1]][activity_type]
                        self.architecture.edges[edge]['product_knowledge_used'] = product_knowledge_of_interface
                        self.architecture.edges[edge]['info_used'].append((info_completeness, info_consitency))
                        
                        # log event
                        if self.log_events:
                            self.event_logger(f'     Interface Quality from {edge[0]} to {edge[1]}: {self.architecture_class.calc_interface_quality(edge):.3f}')
            
            
            case 'Prototyping' | 'Virtual_Integration':
                # integration problems based on interface quality could have also been included
                
                # calculate overall quality
                if self.task_network.nodes[task]['final_task']: # only final task has to calculate quality since individual tasks here do not influence the quality
                    
                    # only integration needs calculation
                    children = self.architecture_class.get_hierarchical_children(architecture_element)
                    if children:
                        
                        total_weighted_quality = 0
                        total_importance = 0
                        
                        for child in children:
                            element_importance = self.architecture.nodes[child]['req_importance']
                            total_importance += element_importance
                            child_quality = self.architecture.nodes[child]['overall_quality']
                            
                            # get relevant interface edges (part of subsystem)
                            relevant_edges = []
                            for dep_element, edges in self.architecture.nodes[child]['interfaces'].items():
                                if dep_element in children:
                                    relevant_edges.extend(edges)
                            
                            sum_of_interface_complexities = 0
                            weighted_interface_quality = 0
                            for edge in relevant_edges:
                                interface_complexity = self.architecture.edges[edge]['complexity']   
                                interface_quality = self.architecture_class.calc_interface_quality(edge)
                                
                                sum_of_interface_complexities += interface_complexity
                                weighted_interface_quality += interface_quality * interface_complexity
                            
                            child_overall_interface_quality = weighted_interface_quality / sum_of_interface_complexities
                            
                            total_weighted_quality += child_quality * element_importance / (1 + alpha_goodness * math.exp(-beta_goodness * child_overall_interface_quality))
                        
                        sys_des_quality = self.architecture.nodes[architecture_element]['design_quality']    
                        overall_quality = sys_des_quality * total_weighted_quality / total_importance
                    
                        self.architecture.nodes[architecture_element]['overall_quality'] = overall_quality
                    
                    # log event
                    if self.log_events:
                        self.event_logger(f'{activity_type} of {architecture_element} done. Overall Quality: {self.architecture.nodes[architecture_element]['overall_quality']:.3f}')
            
            
            case 'LF_System_Simulation' | 'Component_Simulation' | 'HF_System_Simulation' | 'Testing':
                if not self.check_quantification_success(task, agent):
                    return # skip assignment of successor tasks if failed
                
                if self.task_network.nodes[task]['final_task'] and activity_type == 'Testing':
                    self.architecture.nodes[architecture_element]['completion'] = True
                    if not self.architecture_class.get_parent(architecture_element):
                        return # skip information sharing when testing of overall product was successfull --> simulation done
        
        
        # info generation
        if activity_type in {'System_Design', 'Design'}:
            if activity_type == 'Design': # only relevant for Design right now
                self.update_synchonization(agent, task, architecture_element)
            
            total_tasks = self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]['num_tasks']
            
            # update information completeness
            current_info_completeness = self.org_network.get_agent(agent)['product_information_completeness'][architecture_element][activity_type]
            added_info = (1 - current_info_completeness) / (total_tasks - (self.activity_network.nodes[activity]['n_completed_tasks'] - 1))
            self.org_network.get_agent(agent)['product_information_completeness'][architecture_element][activity_type] += added_info
            
            if added_info < 0:
                raise ValueError(f'Generated Information for {task} is negative ({added_info})')

            # set final info to 1 (correct for float error)
            if self.task_network.nodes[task]['final_task']:
                self.org_network.get_agent(agent)['product_information_completeness'][architecture_element][activity_type] = 1
        
        # Share final data
        if self.task_network.nodes[task]['final_task'] and activity_type != 'Prototyping':
            self.share_information(task, agent, final=True)
        else:
            self.check_for_new_tasks(task, agent)
            if activity_type != 'Prototyping':
                self.share_information(task, agent)
            
    
    def share_information(self, task, agent, final=False):
        activity = self.task_network.nodes[task]['activity_name']
        activity_type = self.activity_network.nodes[activity]['activity_type']
        architecture_element = self.activity_network.nodes[activity]['architecture_element']
        tool = self.org_network.get_agent(agent)['task_queue'][task]['tool']
        
        info_to_share = False
        if final:
            info_to_share = True
            
            if activity_type in {'System_Design', 'Design'}:
                new_info_percentage = 1 - self.knowledge_base['product_information_completeness'][architecture_element] * self.knowledge_base['product_information_consitency'][architecture_element] 
            else: 
                new_info_percentage = 1 # other activities only have final info share

            dependent_tasks = sorted([task for task in self.task_network.successors(task) if self.task_network.nodes[task]['task_status'] not in {'Completed', 'Assigned'}])

            info_amount = self.activity_network.nodes[activity]['effort'] * new_info_percentage
        
            
        elif activity_type in {'System_Design', 'Design'}:
            interoperability = self.tools[tool].get('interoperability', 0)
            productivity = (self.tools[tool]['productivity'] + self.knowledge_base['productivity']) / 2
            digital_literacy = self.calc_supportTool_effectivness(agent)
            
            propability_info_sharing = base_info_exchange_propability + interoperability * productivity * digital_literacy * info_exchange_propability_factor
            if propability_info_sharing > random.random():
                info_to_share = True
                
                completion_level = self.org_network.get_agent(agent)['product_information_completeness'][architecture_element][activity_type]
                knowledge_base_completeness_level = self.knowledge_base['product_information_completeness'][architecture_element]
                new_info = completion_level - knowledge_base_completeness_level * self.knowledge_base['product_information_consitency'][architecture_element] 
                
                info_amount = self.activity_network.nodes[activity]['effort'] * new_info
                dependent_tasks = []
        
        if info_to_share:
            self.assign_task(agent, task_type='Share_Information', 
                            info={'task': task, 'info_amount': info_amount, 'start_condition_for': dependent_tasks})

    
    
    def check_for_new_tasks(self, task, agent):
        activity_type = self.task_network.nodes[task]['activity_type']
        architecture_element = self.task_network.nodes[task]['architecture_element']
        
        successors = sorted(list(self.task_network.successors(task))) # networkx functions output random orders ---> this cost me 4 days of debugging FML
        for succ in list(successors):
            if self.task_network.nodes[succ]['task_status'] in {'Completed', 'Assigned'}:
                self.event_logger(f'{succ} was removed from ready tasks {self.task_network.nodes[succ]['task_status']}')
                successors.remove(succ)

        if successors:
            # if single successor of same type: self assign that task
            if (len(successors) == 1 and 
                self.task_network.nodes[successors[0]]['activity_type'] == activity_type and 
                self.task_network.nodes[successors[0]]['architecture_element'] == architecture_element
                ):
                predecessors = list(self.task_network.predecessors(successors[0]))
                if all(self.task_network.nodes[pred]['completed'] for pred in predecessors):
                    
                    relevant_predecessors = [pred for pred in predecessors 
                                            if self.task_network.nodes[pred]['activity_type'] == activity_type and
                                            self.task_network.nodes[pred]['architecture_element'] == architecture_element]
                    if len(relevant_predecessors) == 1:
                        self.assign_task(agent, task_id=successors[0])
                        return
            
            for succ in successors:
                if all(self.task_network.nodes[pred]['completed'] for pred in self.task_network.predecessors(succ)):
                    self.tasks_ready.add(succ)
    
    
    def check_quantification_success(self, task, agent):
        quant_activity = self.task_network.nodes[task]['activity_name']
        activity_type = self.activity_network.nodes[quant_activity]['activity_type']
        architecture_element = self.activity_network.nodes[quant_activity]['architecture_element']
        
        # quality with some error
        perceived_quality, uncertainty, accuracy = self.calc_quantification_result(task, agent)

        if random.random() * self.overall_quality_goal < perceived_quality:
            return True # quantification successfull
        else:
            completion = ( self.activity_network.nodes[quant_activity]['n_completed_tasks'] - 1 ) / self.activity_network.nodes[quant_activity]['num_tasks']
            
            # get activity that is impacted
            directly_impacted_activities = {}
            match activity_type:
                case 'Component_Simulation' | 'LF_System_Simulation':  # always have only one predecessor (design or system design)
                    
                    directly_impacted_activities[next(self.activity_network.predecessors(quant_activity))] = perceived_quality
                    
                    definition_quality = self.architecture.nodes[architecture_element]['definition_quality']
                    design_quality = self.architecture.nodes[architecture_element]['design_quality']
                    if design_quality >= definition_quality: # design quality issues caused by feasibility
                        self.architecture.nodes[architecture_element]['perceived_definition_quality'] = 0
                        
                        if self.log_events:
                            self.event_logger(f'Feasibility of {architecture_element} has to be reworked.')
                            
                    else: # decrease feasibility perceived quality if design and feas
                        missingQ = 1 - design_quality
                        missingD = 1 - design_quality / definition_quality
                        missingF = missingQ - missingD
                        self.architecture.nodes[architecture_element]['perceived_definition_quality'] *= (1 - missingF / missingQ)
                    
                        if self.log_events:
                            self.event_logger(f'Design of {architecture_element} has to be reworked.')
                    
                    
                case 'HF_System_Simulation' | 'Testing':            

                    # get all possible descendents that could cause rework
                    possible_rework_causes = set([architecture_element])
                    descendants = self.architecture_class.get_all_hierarchical_descendants(architecture_element)
                    for descendant in descendants:
                        possible_rework_causes.add(descendant)
                        possible_rework_causes.update([edge for interface, edges in self.architecture.nodes[descendant]['interfaces'].items()
                                                       if interface in descendants for edge in edges])
                    
                    # get actual causes and weights (for selection)
                    rework_causes = []
                    possible_causes = []
                    possible_rework_causes = sorted(list(possible_rework_causes), key=help_fnc.sort_key)
                    for cause in possible_rework_causes:
                        if isinstance(cause, tuple): # edge
                            quality = self.architecture.edges[cause]['perceived_interface_quality']
                            causing_element = cause[0]
                            type = 'interface'
                        else: # node
                            if cause == architecture_element:
                                quality = self.architecture.nodes[cause]['perceived_quality'][activity_type]
                            else:
                                quality = self.architecture.nodes[cause]['perceived_quality'][f'Higher_Level_From_{quant_activity}']
                                
                            causing_element = cause
                            if self.architecture.nodes[cause].get('procure', False):
                                type = 'procure'
                            else:
                                type = 'design'
                        
                        if quality < 1:
                            possible_causes.append((cause, causing_element, quality, type))
                            
                            if random.random() > quality: 
                                rework_causes.append((cause, causing_element, quality, type))
                           
                            
                    # in none where selected make weighted choice
                    if not rework_causes:
                        total_weight = sum([1 / quality for _, _, quality, _ in possible_causes])
                        normalized_weights = [(1 / quality) / total_weight for _, _, quality, _ in possible_causes]
                        selected_index = random.choices(range(len(possible_causes)), weights=normalized_weights, k=1)[0]
                        rework_causes.append(possible_causes[selected_index])

                    # delete unnecessary causes 
                    if len(rework_causes) > 1:
                        all_causing_elements = set()
                        for _, causing_element, _, _ in rework_causes:
                            all_causing_elements.add(causing_element)
                            
                        for cause in list(rework_causes):
                            for ancestor in self.architecture_class.get_all_ancestors(cause[1]):
                                if ancestor in all_causing_elements:
                                    index = rework_causes.index(cause)
                                    rework_causes.pop(index)
                                    break # no need to further check this cause

                    # get impacted activities
                    for cause, causing_element, perceived_quality, type in rework_causes:
                        
                        if self.architecture_class.get_hierarchical_children(causing_element):
                            impacted_activity_type = 'System_Design'
                        else:
                            impacted_activity_type = 'Design'
                        
                        impacted_activity = self.activity_network_class.generate_activity_name(causing_element, impacted_activity_type)
                        
                        # quality used to determine rework amount
                        if type == 'interface':
                            cause_quality = self.architecture_class.calc_interface_quality(cause)
                        else:
                            cause_quality = self.architecture.nodes[cause]['design_quality']
                        
                        if impacted_activity in directly_impacted_activities:
                            directly_impacted_activities[impacted_activity] *= cause_quality  
                        else:
                            directly_impacted_activities[impacted_activity] = cause_quality
                        
                        # perceived quality of interface or feasibility is set to 0 if it was the cause --> starts collaboration otherwise there only is a chance to start collab
                        if type == 'interface':
                            if self.log_events:
                                self.event_logger(f'Interface from {cause[0]} to {cause[1]} has to be reworked.')
                            
                            self.architecture.edges[cause]['perceived_interface_quality'] = 0
                            
                        elif type == 'procure': # feasibility problem
                            if self.log_events:
                                self.event_logger(f'Feasibility of {cause} has to be reworked.')
                                
                            self.architecture.nodes[cause]['perceived_definition_quality'] = 0
                            
                        else: # design checks
                            definition_quality = self.architecture.nodes[cause]['definition_quality']
                            design_quality = self.architecture.nodes[cause]['design_quality']
                            if design_quality >= definition_quality: # design quality issues caused by feasibility
                                self.architecture.nodes[cause]['perceived_definition_quality'] = 0
                                
                                if self.log_events:
                                    self.event_logger(f'Feasibility of {cause} has to be reworked.')
                                    
                            else: # decrease feasibility perceived quality if design and feas
                                missingQ = 1 - design_quality
                                missingD = 1 - design_quality / definition_quality
                                missingF = missingQ - missingD
                                self.architecture.nodes[cause]['perceived_definition_quality'] *= (1 - missingF / missingQ)
                            
                                if self.log_events:
                                    self.event_logger(f'Design of {cause} has to be reworked.')
                                


            # Event log
            if self.log_events:
                self.event_logger(f'"{task}" failed due to quality issues ({round(completion * 100)}% completion - overall quality: {round(perceived_quality, 3)}). {", ".join([f'"{activity}" ({quality})' for activity, quality in directly_impacted_activities.items()])} have to be reworked.')
            
            # trigger rework and reset testing activity
            self.reset_activities([quant_activity], 'Interrupted')
            activities_to_reset = set()
            tasks_ready = []
            all_impacted_elements = set()
            for impacted_activity, quality in directly_impacted_activities.items():
                tasks_ready.extend(self.activity_rework(impacted_activity, (1 - quality)))

                impacted_element = self.activity_network.nodes[impacted_activity]['architecture_element']
                
                # reset information used on interfaces
                if activity_type not in {'LF_System_Simulation', 'Component_Simulation'} and self.architecture_class.get_hierarchical_children(architecture_element):
                    if self.architecture_class.get_hierarchical_children(impacted_element):
                        for element_component in self.architecture_class.get_all_components(impacted_element):
                            for _, edge in self.architecture.nodes[element_component]['interfaces'].items():
                                edge = edge[0]
                                
                                sum_info_used = 0
                                n_info = len(self.architecture.edges[edge]['info_used'])
                                for comp, cons in self.architecture.edges[edge]['info_used']:
                                    sum_info_used += comp * cons

                                average_info_used = sum_info_used / n_info
                                                
                                self.architecture.edges[edge]['info_used'] = []
                                self.architecture.edges[edge]['old_info'] = average_info_used
                    else:
                        self.reset_interface_information(impacted_element, (1 - quality))
                
                # reset 
                if activity_type in {'LF_System_Simulation', 'Component_Simulation'}:
                    self.update_knowledge_quality(impacted_element, quality, second_order_rework=False)
                    
                else:
                    self.update_knowledge_quality(impacted_element, quality, second_order_rework=True)
                    all_impacted_elements.add(impacted_element)
                    all_impacted_elements.update(self.architecture_class.get_all_hierarchical_descendants(impacted_element))
                    all_impacted_elements.update(self.architecture_class.get_all_ancestors(impacted_element))

                
            # reset activities inbetween
            if activity_type not in {'LF_System_Simulation', 'Component_Simulation'}:
                for impacted_activity in directly_impacted_activities:
                    activity_paths = list(nx.all_simple_paths(self.activity_network, source=impacted_activity, target=quant_activity))
                    for path in activity_paths:
                        for activity in path:
                            if activity not in {impacted_activity, quant_activity} and self.activity_network.nodes[activity]['architecture_element'] in all_impacted_elements:
                                activities_to_reset.add(activity)

                self.reset_activities(list(activities_to_reset), 'Rework Required')
            
                # check other activities that have been completed or ongoing (for second order rework, that might trigger rework for ongoing work)
                self.check_successor_rework(list(directly_impacted_activities))
            
            # information sharing task
            info_amount = completion * self.activity_network.nodes[quant_activity]['effort']
            self.assign_task(agent, task_type='Share_Information', info={'task': task, 'info_amount': info_amount, 'start_condition_for': tasks_ready, 'rework_info': True})
            
            return False # quantification not successful

    
    def calc_quantification_result(self, task, agent):
        architecture_element = self.task_network.nodes[task]['architecture_element']
        activity = self.task_network.nodes[task]['activity_name']
        activity_type = self.activity_network.nodes[activity]['activity_type']
        n_tasks_completed = self.activity_network.nodes[activity]['n_completed_tasks']
        
        tool = self.org_network.get_agent(agent)['task_queue'][task]['tool']
        tool_info = self.tools[tool]

        
        tool_accuracy = tool_info['accuracy']
        agent_competency = self.org_network.get_agent(agent)['task_queue'][task]['additional_info']['competency']
        accuracy = tool_accuracy * agent_competency * self.calc_EngTool_effectivness(agent, tool, use_excess=False) # 1 for perfect quantification, 0 for bad quantification
        
        if activity_type == 'LF_System_Simulation':
            quality_type = 'design_quality'
        else:
            quality_type = 'overall_quality'
        perceived_quality, uncertainty = self.calc_perceived_quality(architecture_element, quality_type, activity_type, accuracy, n_tasks_completed)
        
        # all other activities have to use design quality to specify perceived qualities of elements --> needed for rework
        if activity_type != 'LF_System_Simulation':
            self.calc_perceived_quality(architecture_element, 'design_quality', activity_type, accuracy, n_tasks_completed, uncertainty)

        # perceived_feasibility
        self.calc_perceived_quality(architecture_element, 'definition_quality', activity_type, accuracy, n_tasks_completed, uncertainty)
        
        # perceived quality of lower level elements
        if self.architecture_class.get_hierarchical_children(architecture_element) and activity_type in {'HF_System_Simulation', 'Testing'}:
            for decendant in self.architecture_class.get_all_hierarchical_descendants(architecture_element):
                quality_to_reset = f'Higher_Level_From_{activity}'
                
                # design quality and feasibility
                self.calc_perceived_quality(decendant, 'design_quality', quality_to_reset, accuracy, n_tasks_completed, uncertainty)
                self.calc_perceived_quality(decendant, 'definition_quality', quality_to_reset, accuracy, n_tasks_completed, uncertainty)
                
                # interface quality
                if not self.architecture_class.get_hierarchical_children(decendant):
                    for _, edge in self.architecture.nodes[decendant]['interfaces'].items():
                        self.calc_perceived_quality(edge[0], 'interface_quality', quality_to_reset, accuracy, n_tasks_completed, uncertainty)
                
        
        return perceived_quality, uncertainty, accuracy
    
    
    # perceived quality is the comultative mean of all perceived quality values
    def calc_mean_quality_update(self, element, quality_to_reset, n_tasks_completed, new_quality):
        previous_mean = self.architecture.nodes[element]['perceived_quality'].get(quality_to_reset, None)
        if previous_mean is not None:
            return (previous_mean * (n_tasks_completed - 1) + new_quality) / n_tasks_completed
        else:
            return new_quality
    
    
    def calc_perceived_quality(self, element, quality_type, quality_to_reset, accuracy, tasks_completed, uncertainty=None, upper_limit_acc=1):
        
        if quality_type == 'interface_quality':
            actual_quality = self.architecture_class.calc_interface_quality(element)
        else:
            actual_quality = self.architecture.nodes[element][quality_type]
        
        if accuracy < 0 or accuracy > upper_limit_acc:
            raise ValueError(f'Accuracy has to be between 0 and {upper_limit_acc}')
        
        upper_limit = 1 - (1 - actual_quality) ** (upper_limit_acc / accuracy)  # complementary power law, alternativ: power: aq^acc; ratio: aq/(aq-acc*(1-aq))
        if not uncertainty:
            return_uncertainty = True
            if use_uncertainty_for_validation:
                uncertainty = random.triangular(0, 1, 1)
            else:
                uncertainty = 1
        else:
            return_uncertainty = False
        perceived_quality = actual_quality + (upper_limit - actual_quality) * uncertainty
        
        if quality_type == 'design_quality':
            if use_uncertainty_for_validation:
                self.architecture.nodes[element]['perceived_quality'][quality_to_reset] = self.calc_mean_quality_update(element, quality_to_reset, tasks_completed, perceived_quality)
            else:
                self.architecture.nodes[element]['perceived_quality'][quality_to_reset] = perceived_quality
        
        
        # this is a bit quick and dirty --> would have to be done similar to the design quality
        elif quality_type == 'interface_quality':
            self.architecture.edges[element]['perceived_interface_quality'] = perceived_quality
        elif quality_type == 'definition_quality':
            self.architecture.nodes[element]['perceived_definition_quality'] = perceived_quality
        
        if return_uncertainty:
            return perceived_quality, uncertainty
        else:
            return perceived_quality
    
    
    def check_successor_rework(self, activities):
        
        dependent_elements = {}
        for activity in activities:
            architecture_element = self.activity_network.nodes[activity]['architecture_element']
            architecture_elements = [architecture_element]
            architecture_elements.extend(self.architecture_class.get_all_hierarchical_descendants(architecture_element))
            architecture_elements.extend(self.architecture_class.get_all_ancestors(architecture_element))
            dependent_elements[activity] = architecture_elements
        
        activities_to_reset = set()
        
        for activity, act_data in self.activity_network.nodes(data=True):
            if activity not in activities:
                if act_data['activity_status'] not in {'Waiting', 'Rework Required', 'Interrupted'}: # in 'In Progress', 'Completed'
                        for orginating_activity in activities:
                            if (act_data['architecture_element'] in dependent_elements[orginating_activity] 
                                and nx.has_path(self.activity_network, orginating_activity, activity)): # if activity is downstream activity of one of the activities that has to be reset
                                activities_to_reset.add(activity)
                                break
        
        if activities_to_reset:
            self.reset_activities(activities_to_reset, reset_reason='Rework Required')
    
    
    def incomplete_quality_update(self, element, quality_to_update, completion_level): # reduce quality for incomplete quantification but enhance partially with other qualities
        
        if self.architecture.nodes[element]['previous_perceived_quality'].get(quality_to_update, False):
            current_perceived_quality = self.architecture.nodes[element]['previous_perceived_quality'][quality_to_update]
            
            min_other_qualities = 1
            for p_q_type, quality in self.architecture.nodes[element]['previous_perceived_quality'].items():
                if p_q_type != quality_to_update:
                    if quality >= current_perceived_quality and quality < min_other_qualities:
                        min_other_qualities = quality

            partially_reduced_quality = current_perceived_quality * completion_level + min_other_qualities * (1 - completion_level)
            self.architecture.nodes[element]['previous_perceived_quality'][quality_to_update] = partially_reduced_quality


    def reset_activities(self, activities, reset_reason:{'Interrupted', 'Rework Required'}):

        for activity in activities:
            # incomplete testing results are less effective 
            if (reset_reason == 'Rework Required' and self.activity_network.nodes[activity]['activity_status'] == 'In Progress' and 
                self.activity_network.nodes[activity]['activity_type'] in {'LF_System_Simulation', 'Component_Simulation', 'HF_System_Simulation', 'Testing'}
                ):
                element = self.activity_network.nodes[activity]['architecture_element']
                activity_type = self.activity_network.nodes[activity]['activity_type']
                
                completion_level = self.activity_network.nodes[activity]['n_completed_tasks'] / self.activity_network.nodes[activity]['num_tasks']

                self.incomplete_quality_update(element, activity_type, completion_level)

                if activity_type in {'HF_System_Simulation', 'Testing'}:
                    quality_to_reset = f'Higher_Level_From_{activity}'
                    for decendant in self.architecture_class.get_all_hierarchical_descendants(element):
                        self.incomplete_quality_update(decendant, quality_to_reset, completion_level)

            
            self.activity_network.nodes[activity]['activity_status'] = reset_reason
            self.activity_network.nodes[activity]['n_completed_tasks'] = 0
            self.activity_network.nodes[activity]['second_order_rework_reduction'] = 1
            
            for task in self.activity_network.nodes[activity]['tasks']: 
                if reset_reason == 'Interrupted' and self.task_network.nodes[task]['final_task']:
                    self.task_network.nodes[task]['task_status'] = 'Completed' # if final task starts rework it is never set to completed this is needed however to avoid double deletion
                # delete tasks related to activity in agent queues
                if self.task_network.nodes[task]['task_status'] not in {'Waiting', 'Completed', 'Rework Required'}:
                    self.reset_task(task)
                
                if reset_reason == 'Interrupted':
                    self.task_network.nodes[task]['task_status'] = 'Waiting'
                elif reset_reason == 'Rework Required':
                    self.task_network.nodes[task]['task_status'] = 'Rework Required'
                    
                self.task_network.nodes[task]['completed'] = False
   
    
    def reset_task(self, task):
        responsible_agent = self.task_network.nodes[task]['assigned_to'] 
        # reset related info requests
        for request in list(self.org_network.get_agent(responsible_agent)['pending_info_requests']): 
            if request['task'] == task:
                req_id = request['req_id']
                requested_from = request['sent_to']
                self.org_network.get_agent(responsible_agent)['pending_info_requests'].remove(request)
                for reqs in list(self.org_network.get_agent(requested_from)['info_requests']):
                    if reqs['req_id'] == req_id:
                        self.org_network.get_agent(requested_from)['info_requests'].remove(reqs)
                        break
        
        # reset related collaboration requests
        for request in list(self.org_network.get_agent(responsible_agent)['pending_collab_requests']):
            if request['task'] == task:
                if request['request_type'] != 'Collaboration': # exclude collaboration to not be reset when additional collaboriation for this activity is started
                    req_id = request['req_id']
                    requested_from = request['sent_to']
                    self.org_network.get_agent(responsible_agent)['pending_collab_requests'].remove(request)
                    for reqs in list(self.org_network.get_agent(requested_from)['collab_requests']):
                        if reqs['req_id'] == req_id:
                            self.org_network.get_agent(requested_from)['collab_requests'].remove(reqs)
                            break
        
        # delete all assigned related tasks
        for task_in_q, task_info in list(self.org_network.get_agent(responsible_agent)['task_queue'].items()):
            if task_info['task_type'] == 'Technical_Work':
                if task_in_q == task:
                    del self.org_network.get_agent(responsible_agent)['task_queue'][task_in_q]
            elif task_info['task_type'] != 'Noise':
                if task_info['task_type'] != 'Collaboration': # exclude collaboration to not be reset when additional collaboriation for this activity is started
                    linked_technical_task = task_info['additional_info']['task']
                    if linked_technical_task == task:
                        if task_info['task_type'] == 'Consultation':
                            expert = task_info['additional_info']['expert']
                            for t, t_i in list(self.org_network.get_agent(expert)['task_queue'].items()):
                                if t_i['task_type'] == 'Provide_Consultation':
                                    del self.org_network.get_agent(expert)['task_queue'][t]
                                    break
                        del self.org_network.get_agent(responsible_agent)['task_queue'][task_in_q]



    def reduce_information(self, element, reduction_percentage, first_order=False):
        # find dresponsible agent
        responsible_agent = self.architecture.nodes[element]['responsible_designer']
        
        # reset completness of responsible agent
        if self.architecture_class.get_hierarchical_children(element):
            design_activity = 'System_Design'
        else:
            design_activity = 'Design'
        self.org_network.get_agent(responsible_agent)['product_information_completeness'][element][design_activity] *= reduction_percentage
        new_completeness = self.org_network.get_agent(responsible_agent)['product_information_completeness'][element][design_activity]
        
        # log event
        if self.log_events:
            self.event_logger(f'Information Completeness of {element} reduced to {new_completeness:.3f}')
        
        # reduce agents 
        for agent in self.org_network.all_agents:
            for activity_type in self.org_network.get_agent(agent)['responsibilities']:
                if not (agent == responsible_agent and activity_type == design_activity):
                    if first_order: # only excess information is wrong (all information underneath the new_completeness level is correct)
                        agent_consitency = self.org_network.get_agent(agent)['product_information_consitency'][element][activity_type]
                        agent_completeness = self.org_network.get_agent(agent)['product_information_completeness'][element][activity_type]
                        if new_completeness < agent_consitency * agent_completeness:
                            self.org_network.get_agent(agent)['product_information_consitency'][element][activity_type] = new_completeness / agent_completeness
                    else:
                        self.org_network.get_agent(agent)['product_information_consitency'][element][activity_type] *= reduction_percentage
            
            # log event
            if self.log_events:
                for activity_type in self.org_network.get_agent(agent)['responsibilities']:
                    if self.org_network.get_agent(agent)['product_information_consitency'][element][activity_type] > 0:
                        self.event_logger(f'        {agent}: Cons ({activity_type}): {self.org_network.get_agent(agent)['product_information_consitency'][element][activity_type]:.3f}, Comp ({activity_type}): {self.org_network.get_agent(agent)['product_information_completeness'][element][activity_type]:.3f}')
                    
        # reduce knowledge base
        if first_order:
            kb_consitency = self.knowledge_base['product_information_consitency'][element]
            kb_completeness = self.knowledge_base['product_information_completeness'][element]
            if new_completeness < kb_consitency * kb_completeness:
                self.knowledge_base['product_information_consitency'][element] = new_completeness / kb_completeness
            
        else:
            self.knowledge_base['product_information_consitency'][element] *= reduction_percentage
        
            # log event
        if self.log_events:
            if self.knowledge_base['product_information_consitency'][element] > 0:
                self.event_logger(f'        Knowledge Base: Cons: {self.knowledge_base['product_information_consitency'][element]:.3f}, Comp: {self.knowledge_base['product_information_completeness'][element]:.3f}')
                
                    
    def reset_quality(self, element):
        # save element qualities
        for q_type in {'definition', 'design', 'overall'}:
            self.architecture.nodes[element][f'previous_{q_type}_quality'] = self.architecture.nodes[element][f'{q_type}_quality']
            
        # reset and save perceived quality
        #if not all(perceived_quality == 0 for perceived_quality in self.architecture.nodes[element]['perceived_quality'].values()):
        self.architecture.nodes[element]['previous_perceived_quality'] = self.architecture.nodes[element]['perceived_quality'].copy()
        self.architecture.nodes[element]['perceived_quality'] = {
            'LF_System_Simulation': 0,
            'Component_Simulation': 0,
            'HF_System_Simulation': 0,
            'Testing': 0
            }
    
    
    def update_knowledge_quality(self, architecture_element, reduction_percentage, second_order_rework=False, start_with_first_order=True, reset_knowledge=True):

        def reset_hierarchicaly_dependent_elements(element, reduction_percentage):
            if self.architecture_class.get_hierarchical_children(element):
                for child in self.architecture_class.get_hierarchical_children(element):
                    self.reduce_information(child, reduction_percentage)
                    if reset_knowledge:
                        self.reset_quality(child)
                    self.architecture.nodes[element]['completion'] = False
                    
                    reset_hierarchicaly_dependent_elements(child, reduction_percentage)

        # reset direct element
        self.reduce_information(architecture_element, reduction_percentage, first_order=start_with_first_order)
        if reset_knowledge:
            self.reset_quality(architecture_element)
        self.architecture.nodes[architecture_element]['completion'] = False
        
        # recursivly reset hierarchically dependent elements 
        if second_order_rework:
            reset_hierarchicaly_dependent_elements(architecture_element, reduction_percentage)

    
    def reset_interface_information(self, element, rework_percentage):
        for _, edge in self.architecture.nodes[element]['interfaces'].items():
            
            edge = edge[0]
            
            if self.architecture.edges[edge]['info_used']:

                sum_info_used = 0
                n_info = len(self.architecture.edges[edge]['info_used'])
                
                for comp, cons in self.architecture.edges[edge]['info_used']:
                    sum_info_used += comp * cons

                average_info_used = sum_info_used / n_info
                
                n_entries_to_reset = max(math.ceil(rework_percentage * n_info), 1)
                
                n_cut = n_info - n_entries_to_reset
                
                old_info_used = []
                for _ in range(n_cut):
                    old_info_used.append((average_info_used, 1))
                
                self.architecture.edges[edge]['info_used'] = old_info_used
                
                # fall back
                if n_cut == 0:
                    self.architecture.edges[edge]['old_info'] = average_info_used


    
    def activity_rework(self, activity, rework_percentage):
        activity_info = self.activity_network.nodes[activity]
        total_tasks = activity_info['num_tasks']
        
        if activity_info['activity_status'] == 'Completed':
            activity_info['activity_status'] = 'Rework Required'
            
            n_tasks_to_rework = max(math.ceil(rework_percentage * total_tasks), 1)
            n_completed = total_tasks
            
            # get tasks to be reworked
            tasks = activity_info['tasks'].copy()
            tasks_with_no_rework = []

            for _ in range(total_tasks - n_tasks_to_rework):
                task = tasks.pop(0)
                tasks_with_no_rework.append(task)
            tasks_to_be_reworked = tasks
                
        else:
            # get completed tasks
            completed_tasks = []
            for task in activity_info['tasks']:
                if self.task_network.nodes[task]['completed'] or self.task_network.nodes[task]['task_status'] == 'Completed':
                    completed_tasks.append(task)
                
                # reset task if it is active
                if self.task_network.nodes[task]['task_status'] not in {'Waiting', 'Completed', 'Rework Required'}:
                    self.task_network.nodes[task]['task_status'] = 'Rework Required'
                    self.reset_task(task)
                    
            n_completed = len(completed_tasks)
            
            if n_completed == 0:
                n_tasks_to_rework = 0
                tasks_to_be_reworked = []
                tasks_with_no_rework = []
            else:
                n_tasks_to_rework = max(math.ceil(rework_percentage * n_completed), 1)
                
                tasks_with_no_rework = []
                for _ in range(n_completed - n_tasks_to_rework):
                    task = completed_tasks.pop(0)
                    tasks_with_no_rework.append(task)
                tasks_to_be_reworked = completed_tasks
            
        
        # reset activity and tasks to be reworked
        self.activity_network.nodes[activity]['n_completed_tasks'] = n_completed - n_tasks_to_rework
        for task in tasks_to_be_reworked:
            self.task_network.nodes[task]['task_status'] = 'Rework Required'
            self.task_network.nodes[task]['completed'] = False
        
        # get tasks to be started next
        tasks_ready = set()
        if len(tasks_with_no_rework) == 0:
            first_task = activity_info['tasks'][0]
            if all(self.task_network.nodes[pred]['completed'] for pred in self.task_network.predecessors(first_task)):
                tasks_ready.add(first_task)
        else:
            for task in tasks_with_no_rework:
                successors = sorted(list(self.task_network.successors(task))) # networkx functions output random orders ---> this cost me 4 days of debugging FML
                for succ in successors:
                    if succ in tasks_to_be_reworked:
                        if all(self.task_network.nodes[pred]['completed'] for pred in self.task_network.predecessors(succ)):
                            tasks_ready.add(succ)
        
        return sorted(list(tasks_ready))
    
                    
    def complete_information_sharing(self, agent, task_info):
        task = task_info['task']
        architecture_element = self.task_network.nodes[task_info['task']]['architecture_element']
        activity = self.task_network.nodes[task_info['task']]['activity_name']
        activity_type = self.activity_network.nodes[activity]['activity_type']
        
        # complete final task
        if self.task_network.nodes[task]['final_task'] and not task_info.get('rework_info', False):
            self.task_network.nodes[task]['completed'] = True
        
        # update knowledge base
        if activity_type in {'System_Design', 'Design'}:
            completeness = self.org_network.get_agent(agent)['product_information_completeness'][architecture_element][activity_type]
            self.knowledge_base['product_information_completeness'][architecture_element] = completeness
            self.knowledge_base['product_information_consitency'][architecture_element] = 1 # consitency is 1 from agent providing it

        # check if there was a request
        for request in self.org_network.get_agent(agent)['info_requests']:
            if round(request['amount'], 6) <= round(completeness, 6):
                requestor = request['requestor']
                existing_info = self.org_network.get_agent(requestor)['product_information_completeness'][architecture_element][activity_type] * self.org_network.get_agent(requestor)['product_information_consitency'][architecture_element][activity_type]
                missing_info = completeness - existing_info
                activity_effort = self.activity_network.nodes[activity]['effort']
                info_amount = request['fraction'] * activity_effort * missing_info
                
                if missing_info < 0:
                    raise ValueError(f'Missing information was below 0')

                # event log
                if self.log_events:
                    self.event_logger(f'Information on {architecture_element} requested by {requestor} has been made available.')
                
                # start information receival
                self.assign_task(requestor, 
                                 task_type='Receive_Information', 
                                 info={'task': request['task'], 'info_amount': info_amount, 'elements': [architecture_element], 'request_response': True})
                
                # delete request
                req_id = request['req_id']
                for pend_req in list(self.org_network.get_agent(requestor)['pending_info_requests']):
                    if pend_req['req_id'] == req_id:
                        self.org_network.get_agent(requestor)['pending_info_requests'].remove(pend_req)
                        break
                self.org_network.get_agent(agent)['info_requests'].remove(request)
                

        # system engineer checks compatibility of new information
        if activity_type in {'Design'}:   # system design not yet considered for interface information exchange
            parent_agent = self.architecture.nodes[self.architecture_class.get_parent(architecture_element)]['responsible_designer']
            info_to_check = {}
            
            agents_not_checking_information = []
            agents_checking_information = []
            agents_with_ongoing_information_task = {}
            
            for dependent_element, interfaces in self.architecture.nodes[architecture_element]['interfaces'].items():
                interface = interfaces[0] # only one for component level
                
                parent_element = self.architecture_class.get_parent(dependent_element)
                dependent_element_agent = self.architecture.nodes[dependent_element]['responsible_designer']
                if dependent_element_agent != parent_agent: # check if interface in subsystem
                    parent_element = self.architecture_class.get_parent(parent_element)
                    dependent_element_agent = self.architecture.nodes[parent_element]['responsible_designer'] # this only works with the two hierarchies currently implemented --> recursion would be needed
                
                # check if agent has ongoing information task
                if dependent_element_agent not in agents_with_ongoing_information_task:
                    for task, task_data in self.org_network.get_agent(dependent_element_agent)['task_queue'].items():
                        if (task_data['task_type'] == 'Receive_Information' and 
                            architecture_element in task_data['additional_info']['elements'] and 
                            task_data['additional_info']['Compatibility_Check']):
                            agents_with_ongoing_information_task[dependent_element_agent] = task_data
                
                # if information is being checked all information has to be received
                if dependent_element_agent in agents_with_ongoing_information_task or dependent_element_agent in agents_checking_information:
                    propability_info_check = 1
                else:
                    # probability to check information    
                    tool = self.activity_network.nodes[self.activity_network_class.generate_activity_name(parent_element, 'System_Design')]['tool']
                    interoperability = self.tools[tool].get('interoperability', 0)
                    productivity = (self.knowledge_base['productivity'] + self.tools[tool]['productivity']) / 2
                    digital_literacy = self.calc_supportTool_effectivness(dependent_element_agent)
                    propability_info_check = base_info_exchange_propability + interoperability * productivity * digital_literacy * info_exchange_propability_factor

                if ((random.random() < propability_info_check or self.task_network.nodes[task_info['task']]['final_task'])
                    and dependent_element_agent not in agents_not_checking_information):  # always task final info + only check every agent once
                    
                    agents_checking_information.append(dependent_element_agent)
                    
                    # new information
                    agent_info_completeness = self.org_network.get_agent(dependent_element_agent)['product_information_completeness'][architecture_element]['System_Design']
                    agent_info_consitency = self.org_network.get_agent(dependent_element_agent)['product_information_consitency'][architecture_element]['System_Design']
                    knowledge_base_completeness = self.knowledge_base['product_information_completeness'][architecture_element]
                    knowledge_base_info_consitency = self.knowledge_base['product_information_consitency'][architecture_element]
                    new_info = self.calculate_new_information(architecture_element, dependent_element_agent, agent_info_completeness, agent_info_consitency, knowledge_base_completeness, knowledge_base_info_consitency)
                    
                    # system engineer does not have to check if no new information is available
                    if new_info > 0:

                        # relevant information based on interface complexity
                        interface_complexity = self.architecture.edges[interface]['complexity']
                        technical_complexity = self.architecture.nodes[architecture_element]['technical_complexity']
                        total_interface_complexity = self.architecture.nodes[architecture_element]['total_interface_complexity']
                        info_percentage = interface_complexity / (technical_complexity + total_interface_complexity)
                        
                        info_amount = self.activity_network.nodes[activity]['effort'] * info_percentage * new_info
                        
                        # sum up information needed
                        if dependent_element_agent not in info_to_check:
                            info_to_check[dependent_element_agent] = 0
                        info_to_check[dependent_element_agent] += info_amount
                    
                else:
                    agents_not_checking_information.append(dependent_element_agent)
                    
            # create information receival tasks
            for agt, info in info_to_check.items():
                # if the agent already has a receive information task: increase the amount
                if agt in agents_with_ongoing_information_task:
                    newly_added_information = info - agents_with_ongoing_information_task[agt]['additional_info']['info_amount']
                    if newly_added_information < 0: # rework must have occured --> ignored in that case
                        continue
                    else:
                        parent_element = self.org_network.get_agent(agt)['responsible_element']
                        tool = self.activity_network.nodes[self.activity_network_class.generate_activity_name(parent_element, 'System_Design')]['tool']
                        interoperability = self.tools[tool].get('interoperability', 0)
                        digital_literacy = self.calc_supportTool_effectivness(agt)
                        productivity = (self.knowledge_base['productivity'] + self.tools[tool]['productivity']) / 2
                        familiarity = self.org_network.get_agent(agent)['knowledge_base_familiarity'][0]
                        efficiency = digital_literacy * productivity * familiarity
                        
                        nominal_additional_effort = random.triangular(info_handling_time_min, info_handling_time_max) * newly_added_information * (1 - interoperability)
                        
                        verification_efficiency = self.tools[tool]['productivity'] * self.calc_EngTool_effectivness(agt, tool)
                        additional_verification_effort = verification_effort_factor * nominal_additional_effort / verification_efficiency
                        
                        new_added_effort = max((additional_verification_effort + nominal_additional_effort / efficiency), step_size)
                        
                        # increase the ongoings task effort
                        agents_with_ongoing_information_task[agt]['remaining_effort'] += new_added_effort
                        agents_with_ongoing_information_task[agt]['inital_effort'] += new_added_effort
                        agents_with_ongoing_information_task[agt]['nominal_effort'] += nominal_additional_effort
                        agents_with_ongoing_information_task[agt]['additional_info']['verification_effort'] += additional_verification_effort
                        agents_with_ongoing_information_task[agt]['additional_info']['info_amount'] += newly_added_information

                else:
                    self.assign_task(agt, 
                                     task_type='Receive_Information', 
                                     info={'task': task_info['task'], 'info_amount': info, 'elements': [architecture_element], 'Compatibility_Check': True})
        
        
        # start successor tasks
        info_amount = task_info['info_amount']
        n_tasks = len(task_info['start_condition_for'])
        for suc_task in task_info['start_condition_for']:
            
            activity = self.task_network.nodes[suc_task]['activity_name']
            
            # rework information has to be received
            if task_info.get('rework_info', False):
                self.assign_task(self.activity_network.nodes[activity]['assigned_to'], 
                                    task_type='Receive_Information', 
                                    info={'task': suc_task, 'info_amount': info_amount / n_tasks, 'rework_info': True})
            
            if self.task_network.nodes[suc_task]['task_status'] in {'Waiting', 'Rework Required'}:
                if all(self.task_network.nodes[pred]['completed'] for pred in self.task_network.predecessors(suc_task)):
                    self.tasks_ready.add(suc_task)
                        
            elif self.task_network.nodes[suc_task]['task_status'] in {'Completed', 'Assigned'}:
                if all(self.task_network.nodes[pred]['completed'] for pred in self.task_network.predecessors(suc_task)):
                    ## some bug where this is triggered exist (not sure why --> possibly problems with resetting task, not checking status correctly, starting too early, assignment of successor tasks when predecessors where paused)
                    warnings.warn(f'Second order rework has to be checked for {suc_task} (start condition from info shareing from {task_info['task']})')
                    
                    ########### second order rework for special cases where not all downstream tasks were reset (feasibility --> feasibility rework was not implemented)
            
            
    def complete_receive_information(self, agent, task_data, originating_task):
        task_info = task_data['additional_info']
        task = task_info['task']
        activity_node = self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]   
        activity_type = activity_node['activity_type']
        
        architecture_elements = task_info.get('elements', [])
        
        # increase familiarity only when it was from a search
        if architecture_elements:
            self.org_network.get_agent(agent)['knowledge_base_familiarity'][0] += ((1 - self.org_network.get_agent(agent)['knowledge_base_familiarity'][0]) * 
                                                     task_data['nominal_effort'] * self.calc_supportTool_effectivness(agent, use_excess=False) * familiarity_increase_rate)
        
        
        if task_info.get('Compatibility_Check', False):
            info_to_use = 'System_Design'
        else:
            info_to_use = activity_type
        
        # receive information on elements
        element_info_to_check = {}
        for element in architecture_elements:
            # relative amount of new information
            agent_info_completeness = self.org_network.get_agent(agent)['product_information_completeness'][element][info_to_use]
            agent_info_consitency = self.org_network.get_agent(agent)['product_information_consitency'][element][info_to_use]
            knowledge_base_completeness = self.knowledge_base['product_information_completeness'][element]
            knowledge_base_info_consitency = self.knowledge_base['product_information_consitency'][element]
            new_info = self.calculate_new_information(element, agent, agent_info_completeness, agent_info_consitency, knowledge_base_completeness, knowledge_base_info_consitency)
            if new_info > 0:
                element_info_to_check[element] = new_info
            
            # increase knowledge
            self.org_network.get_agent(agent)['product_information_completeness'][element][info_to_use] = knowledge_base_completeness
            self.org_network.get_agent(agent)['product_information_consitency'][element][info_to_use] = knowledge_base_info_consitency
            
            if self.log_events:
                self.event_logger(f'Information on {element} received by {agent} (New Info: {new_info}).')
        
        # check if some information level is required and has to be requested
        requests = task_info.get('request_info', [])
        for element, req_completeness, fraction in requests:
            if req_completeness > self.knowledge_base['product_information_completeness'][element]:
                # pause task
                self.task_network.nodes[task]['task_status'] = 'Information Needed' 
                
                # request information
                responsible_agent = self.architecture.nodes[element]['responsible_designer']
                self.add_request(responsible_agent, 
                                 requestor=agent, 
                                 type='Information', 
                                 info={'task': task, 'amount': req_completeness, 'fraction': fraction, 'element': element})
                
                if self.log_events:
                    self.event_logger(f'{agent} needs information ({req_completeness}) on {element}. {task} paused and information requested from {responsible_agent}.')
                
            elif element not in architecture_elements: # if information became available it is received
                agent_info_completeness = self.org_network.get_agent(agent)['product_information_completeness'][element][activity_type]
                agent_info_consitency = self.org_network.get_agent(agent)['product_information_consitency'][element][activity_type]
                knowledge_base_completeness = self.knowledge_base['product_information_completeness'][element]
                knowledge_base_info_consitency = self.knowledge_base['product_information_consitency'][element]
                new_info = self.calculate_new_information(element, agent, agent_info_completeness, agent_info_consitency, knowledge_base_completeness, knowledge_base_info_consitency)
                if new_info > 0:
                    element_info_to_check[element] = new_info
                
                # add inforamtion
                self.org_network.get_agent(agent)['product_information_completeness'][element][activity_type] = knowledge_base_completeness
                self.org_network.get_agent(agent)['product_information_consitency'][element][activity_type] = knowledge_base_info_consitency
            

        # reset state of paused tasks
        if not self.org_network.get_agent(agent)['pending_info_requests'] and task_info.get('request_response', False):
            self.task_network.nodes[task]['task_status'] = 'Assigned'
            
            # event log
            if self.log_events:
                self.event_logger(f'{task} can be continued by {agent} because all requested information has been received.')
        
        
        # check if collaboration for interface or feasbility is necessary
        if task_info.get('rework_info', False):
            rework_element = self.task_network.nodes[task]['architecture_element']
            activity_name = self.task_network.nodes[task]['activity_name']
            
            # feasibility
            perceived_definition_quality = self.architecture.nodes[rework_element]['perceived_definition_quality']
            self.architecture.nodes[rework_element]['perceived_definition_quality'] = None # set to none because it will not be needed anymore until it is recalculated
            if random.random() > perceived_definition_quality:
                parent_element_agent = self.architecture.nodes[self.architecture_class.get_parent(rework_element)]['responsible_designer']
                
                ###### technically would have to add some rework to the system design activity
                
                # pause activity
                self.activity_network.nodes[activity_name]['activity_status'] = 'Interface or Feasibility Problem'
                
                if self.check_if_idle(agent) and self.check_if_idle(parent_element_agent):
                    self.start_collaboration(task, parent_element_agent, agent, rework_element, 1, 'Feasibility')
                else:
                    self.add_request(agent, parent_element_agent, type='Collaboration', info={'task': task, 'element': rework_element, 'sub_type': 'Feasibility', 'perceived_feasibility': 1})
            
            # interfaces
            if not self.architecture_class.get_hierarchical_children(rework_element):
                for dep_element, edge in self.architecture.nodes[rework_element]['interfaces'].items():
                    edge = edge[0]
                    perceived_interface_quality = self.architecture.edges[edge]['perceived_interface_quality']
                    self.architecture.edges[edge]['perceived_interface_quality'] = None # set to none because it will not be needed anymore until it is recalculated
                    
                    if perceived_interface_quality and random.random() > perceived_interface_quality:
                        # pause activity
                        self.activity_network.nodes[activity_name]['activity_status'] = 'Interface or Feasibility Problem'
                        
                        dep_element_agent = self.architecture.nodes[dep_element]['responsible_designer']
                        
                        if self.check_if_idle(agent) and self.check_if_idle(dep_element_agent):
                            self.start_collaboration(task, agent, dep_element_agent, dep_element, 1, 'Interface') 
                        else:
                            self.add_request(dep_element_agent, agent, type='Collaboration', info={'task': task, 'element': dep_element, 'sub_type': 'Interface', 'rework_percent': 1})
            
        
        # check if design is compatible with received information
        if element_info_to_check and not self.task_network.nodes[task]['first_task'] and activity_type in {'Design'}: # system design not yet considered for interfaces info exchange
            
            interface_rework = {}
            
            # system engineer check
            if task_info.get('Compatibility_Check', False):
                element, new_info_amount = list(element_info_to_check.items())[0] # compatibility check has only one
                element_info_completeness = self.org_network.get_agent(agent)['product_information_completeness'][element]['System_Design']
                old_info_completness = element_info_completeness - new_info_amount
                
                rework_percentage = []
                causing_dep_elements = []
                
                interfaces = sorted(list(self.architecture.nodes[element]['interfaces'].items()))
                random.shuffle(interfaces)
                for dep_element, _ in interfaces:
                    quality = self.architecture_class.calc_interface_quality((element, dep_element))
                    feasibility = self.architecture.edges[element, dep_element]['definition_quality']
                    
                    # quality of available information checked against feasibility
                    if min(1, feasibility) * random.random() > quality:
                        rework_percent = self.calculate_rework_percent(agent, element, dep_element, new_info_amount, quality, 'System_Design') * feasibility
                        
                        rework_percentage.append(rework_percent)
                        causing_dep_elements.append(dep_element)
                    
                    # check if relevant new information to compare
                    dep_element_completeness = self.org_network.get_agent(agent)['product_information_completeness'][dep_element]['System_Design']
                    relevant_new_info = min(element_info_completeness, dep_element_completeness) - old_info_completness
                    if relevant_new_info > 0: # can only compare to the lower level of completion and if new information is included in this portion
                        
                        # could also include second order rework check (change propagation through interfaces) --> not considered currently
                        
                        tool = self.activity_network.nodes[self.activity_network_class.generate_activity_name(self.org_network.get_agent(agent)['responsible_element'] , 'System_Design')]['tool']
                        if random.random() < self.tools[tool]['accuracy']: # detection of problems that go beyond the defined feasibility quality
                            
                            # quality of information that is available on both sides 
                            if random.random() > quality: 
                                rework_percent = self.calculate_rework_percent(agent, element, dep_element, relevant_new_info, quality, 'System_Design') * (1 - feasibility)
                                                            
                                if causing_dep_elements and causing_dep_elements[-1] == dep_element:
                                    rework_percentage[-1] += rework_percent
                                else:
                                    rework_percentage.append(rework_percent)
                                    causing_dep_elements.append(dep_element)
                                
                            dep_quality = self.architecture_class.calc_interface_quality((dep_element, element))
                            if random.random() > dep_quality:
                                rework_percent = self.calculate_rework_percent(agent, dep_element, element, relevant_new_info, dep_quality, 'System_Design') * (1 - feasibility)

                                interface_rework[dep_element] = ([rework_percent], [element])
                        
                        
                interface_rework[element] = (rework_percentage, causing_dep_elements)
                         
            else: # dependent element check 
                own_element = activity_node['architecture_element']
                
                element_info_to_check = sorted(list(element_info_to_check.items()))
                random.shuffle(element_info_to_check)
                for element, new_info_amount in element_info_to_check:
                    quality = self.architecture_class.calc_interface_quality((element, own_element))
                    # check quality by comparing the available information
                    if random.random() > quality:
                        own_info_completeness = self.org_network.get_agent(agent)['product_information_completeness'][own_element][activity_type]
                        dep_info_completeness = self.org_network.get_agent(agent)['product_information_completeness'][element][activity_type]
                        old_dep_info_completeness = dep_info_completeness - new_info_amount
                        relevant_new_info = min(own_info_completeness, dep_info_completeness) - old_dep_info_completeness
                        
                        if relevant_new_info > 0:
                            rework_percent = self.calculate_rework_percent(agent, element, own_element, relevant_new_info, quality, activity_type)
                            
                            interface_rework[element] = ([rework_percent], [own_element])
            
            # delete small rework
            for element in list(interface_rework.keys()):
                percent_list, causing_list = interface_rework[element]

                new_percent_list = []
                new_causing_list = []
                for p, c in zip(percent_list, causing_list):
                    if p >= 0.01: # 1 % rework
                        new_percent_list.append(p)
                        new_causing_list.append(c)

                if new_percent_list:
                    interface_rework[element] = (new_percent_list, new_causing_list)
                else:
                    del interface_rework[element]
            
            # start rework and collaboration
            if task_info.get('Compatibility_Check', False):
                consitency = self.org_network.get_agent(agent)['product_information_consitency'][element]['System_Design']
            else:
                consitency = self.org_network.get_agent(agent)['product_information_consitency'][element][activity_type]
            self.interface_problem(interface_rework, originating_task, consitency)
    
    
    def calculate_rework_percent(self, agent, element, dependent_element, relvant_info, quality, activity_type):
        interface_complexity = self.architecture.edges[element, dependent_element]['complexity']
        technical_complexity = self.architecture.nodes[element]['technical_complexity']
        info_percentage =  min(interface_complexity / technical_complexity, 1)
        consitency = self.org_network.get_agent(agent)['product_information_consitency'][element][activity_type]
        
        return relvant_info * (1 - quality) * info_percentage * consitency

                        
    def interface_problem(self, interface_rework, originating_task, consitency):          
            # start or request consultation and reset activities    
            for element, (rework_percentages, dep_elements) in interface_rework.items():                
                
                if self.architecture_class.get_hierarchical_children(element):
                    activity = self.activity_network_class.generate_activity_name(element, 'System_Design')
                else:
                    activity = self.activity_network_class.generate_activity_name(element, 'Design')
                
                rework_element_agent = self.architecture.nodes[element]['responsible_designer']
                
                # skip reset if collaboration already exists
                if self.activity_network.nodes[activity]['activity_status'] == 'Interface or Feasibility Problem':
                    for task_in_q_data in self.org_network.get_agent(rework_element_agent)['task_queue'].values():
                        if task_in_q_data['task_type'] == 'Collaboration' and task_in_q_data['additional_info']['element'] in dep_elements:
                            index = dep_elements.index(task_in_q_data['additional_info']['element'])
                            dep_elements.pop(index)
                            rework_percentages.pop(index)
                            
                    for request in self.org_network.get_agent(rework_element_agent)['pending_collab_requests']:
                        if request['request_type'] == 'Collaboration' and request['element'] in dep_elements:
                            index = dep_elements.index(request['element'])
                            dep_elements.pop(index)
                            rework_percentages.pop(index)
                            
                if not dep_elements:
                    continue
                    
                # log event
                if self.log_events:
                    self.event_logger(f'{element} has to be reworked due to interface problem with {", ".join([f'"{dep}"' for dep in dep_elements])}. {activity} has been paused.')
                
                
                # only reset knowledge if activity was already completed
                if self.activity_network.nodes[activity]['activity_status'] == 'Completed': 
                    knowledge_reset = True
                else:
                    knowledge_reset = False
                
                # reset activity and information
                rework = 1
                for percent in rework_percentages:
                    rework *= percent
                rework_task = self.activity_rework(activity, rework)
                self.reset_interface_information(element, rework)
                rework_task = rework_task[0] if rework_task else []
                
                if knowledge_reset:
                    if any([self.activity_network.nodes[succ]['activity_status'] == 'Completed' for succ in self.activity_network.successors(activity)]):
                        self.check_successor_rework([activity])
                    else:
                        self.reset_activities(self.activity_network.successors(activity), reset_reason='Rework Required')
                
                self.update_knowledge_quality(element, 1-rework, second_order_rework=True, start_with_first_order=False, reset_knowledge=knowledge_reset)
                
                # reset all tasks in activity to be reworked
                self.reset_information_tasks(activity, rework_task, originating_task, element)
                
                # reassign the rework task
                if rework_task:
                    self.assign_task(rework_element_agent, task_id=rework_task)
                else:
                    rework_task = self.activity_network.nodes[activity]['tasks'][0] # alternativ if no other task exsits
                
                # set status
                self.activity_network.nodes[activity]['activity_status'] = 'Interface or Feasibility Problem'

                # create collaboration tasks or requests
                for dep_element in dep_elements:
                    dep_element_agent = self.architecture.nodes[dep_element]['responsible_designer']
                    
                    if self.check_if_idle(rework_element_agent) and self.check_if_idle(dep_element_agent):
                        self.start_collaboration(rework_task, rework_element_agent, dep_element_agent, dep_element, consitency, 'Interface')
                    else:
                        self.add_request(dep_element_agent, rework_element_agent, type='Collaboration', info={'task': rework_task, 'element': dep_element, 'sub_type': 'Interface', 'rework_percent': consitency})

    
    def reset_information_tasks(self, activity, rework_task, originating_task, element):
        # get tasks to be reset
        if rework_task: # only tasks including and after rework task are reset
            reached_rework_task = False 
            tasks_to_reset = []
            for task in self.activity_network.nodes[activity]['tasks']:
                if task == rework_task:
                    reached_rework_task = True
                if reached_rework_task:
                    tasks_to_reset.append(task)
                    
        else: # all are reset
            tasks_to_reset = self.activity_network.nodes[activity]['tasks']
        
        # parse through queue and reset information tasks related to tasks to be reset
        for agent in self.org_network.all_agents:
            for task, task_data in list(self.org_network.get_agent(agent)['task_queue'].items()):
                if (task_data['task_type'] == 'Share_Information'
                    and task_data['additional_info']['task'] in tasks_to_reset
                    and not task == originating_task
                    and not task_data['additional_info'].get('rework_info', None)):
                    del self.org_network.get_agent(agent)['task_queue'][task]
                    


    def complete_collaboration(self, agent, task_info):
        resp_agent = task_info.get('resp_agent', None)
        
        # do nothing if the responsible agent task finishes
        if not resp_agent:
            return
        
        type = task_info['sub_type']
        
        # event log
        if self.log_events:
            self.event_logger(f'Collaboration of {agent} on {task_info['element']} with {resp_agent} because of a {type} problem was completed.')
        
        if type == 'Feasibility':
            element = task_info['element']
            complexity = self.architecture.nodes[element]['technical_complexity']
        else:
            dep_element = task_info['element']
            element = self.org_network.get_agent(agent)['responsible_element']
            complexity = self.architecture.edges[element, dep_element]['complexity']
        
        # increase product knowledge
        element_for_knowledge_gain = task_info['element']
        inital_knowledge = self.org_network.get_agent(agent)['product_knowledge'][element_for_knowledge_gain]
        knowledge_gain = help_fnc.calc_knowledge_gain(inital_knowledge, complexity, additional_factor=task_info['rework_percent'])
        self.org_network.get_agent(agent)['product_knowledge'][element_for_knowledge_gain] = inital_knowledge + knowledge_gain
        
        
        if self.log_events:
            self.event_logger(f'Product Knowledge of {agent} on {element_for_knowledge_gain} was increased by {knowledge_gain:.3f} to {self.org_network.get_agent(agent)['product_knowledge'][element_for_knowledge_gain]:.3f}.')
        
        # increase definition quality
        if type == 'Feasibility': # for feasibility problems it would be better to start rework of the higher level system design task and then check for possible second order rework 
            knowledge_req_child = self.architecture.nodes[element]['knowledge_req']
            expertise = self.org_network.get_agent(agent)['expertise']
            competency = help_fnc.calc_efficiency_competency(knowledge_req_child, expertise)[1]
            product_knowledge = self.org_network.get_agent(agent)['product_knowledge'][element]
            self.architecture.nodes[element]['definition_quality'] = product_knowledge + (1 - product_knowledge) * competency
            
            #### perceived feasibility ???
        
        else: # create in info activity if agent is missing information
            if self.architecture_class.get_hierarchical_children(element):
                activity_type = 'System_Design'
            else:
                activity_type = 'Design'
            

            agent_info_completeness = self.org_network.get_agent(agent)['product_information_completeness'][element_for_knowledge_gain][activity_type]
            agent_info_consitency = self.org_network.get_agent(agent)['product_information_consitency'][element_for_knowledge_gain][activity_type]
            
            # sets all info used for interface to the current info being used
            for i in range(len(self.architecture.edges[element, element_for_knowledge_gain]['info_used'])):
                self.architecture.edges[element, element_for_knowledge_gain]['info_used'][i] = (agent_info_completeness, agent_info_consitency)
            
            # new information
            knowledge_base_completeness = self.knowledge_base['product_information_completeness'][element_for_knowledge_gain]
            knowledge_base_info_consitency = self.knowledge_base['product_information_consitency'][element_for_knowledge_gain]
            new_info = self.calculate_new_information(element_for_knowledge_gain, agent, agent_info_completeness, agent_info_consitency, knowledge_base_completeness, knowledge_base_info_consitency)
            
            if new_info > 0:
                technical_complexity_dep = self.architecture.nodes[element_for_knowledge_gain]['technical_complexity']
                interface_complexity = self.architecture.edges[element, element_for_knowledge_gain]['complexity']
                info_percentage = interface_complexity / (technical_complexity_dep + interface_complexity)
                
                activity = self.activity_network_class.generate_activity_name(element_for_knowledge_gain, activity_type)
                info_amount = new_info * info_percentage * self.activity_network.nodes[activity]['effort']
                
                self.assign_task(agent, task_type='Receive_Information', info={'task': task_info['task'], 'info_amount': info_amount, 'elements': [element_for_knowledge_gain]})
            
            ###### maybe add rework for resp agent depending on his consitency + info_percentage
        
        
        # check that no feasibility problem (in requests) and no other interface problems (in pending requests) exists
        start_rework = True
        if type == 'Interface':
            agent_to_check = agent
        else:
            agent_to_check = resp_agent
        for req in self.org_network.get_agent(agent_to_check)['pending_collab_requests']:
            if req['request_type'] == 'Collaboration' and req['sub_type'] == 'Interface':
                start_rework = False
        for req in self.org_network.get_agent(agent_to_check)['collab_requests']:
            if req['request_type'] == 'Collaboration' and req['sub_type'] == 'Feasibility':
                start_rework = False
                
        # start required rework if no other request exist
        if start_rework:
            if self.architecture_class.get_hierarchical_children(element):
                activity = self.activity_network_class.generate_activity_name(element, 'System_Design')
            else:
                activity = self.activity_network_class.generate_activity_name(element, 'Design')
                
            self.activity_network.nodes[activity]['activity_status'] = 'Problem Resolved'
            
            if self.log_events:
                self.event_logger(f'All problems for {activity} have been resolved and it can be resumed.')
        
        
        
    
    def complete_consultation(self, agent, task_info):
        expert = task_info.get('expert', None)
        
        # do nothing if the expert task finishes
        if not expert:
            return
        
        # calculation of new knowledge level
        knowledge_item = task_info['knowledge_item']
        task_with_problem = task_info['task']
        complexity = self.architecture.nodes[self.task_network.nodes[task_with_problem]['architecture_element']]['development_complexity']
        inital_knowledge = self.org_network.get_agent(agent)['expertise'][knowledge_item]
        req_knowledge_level = task_info['knowledge_level']
        
        # Event log
        if self.log_events:
            self.event_logger(f'Consultation of {agent} on {self.org_network.knowledge_items[knowledge_item]} with {expert} was completed.')
        
        knowledge_gain = help_fnc.calc_knowledge_gain(inital_knowledge, complexity, req_knowledge_level)
        self.update_expertise(agent, task_with_problem, knowledge_gain, knowledge_item)
        
    
    def complete_search_knowledge_base(self, agent, task_data):
        task_info = task_data['additional_info']
        
        task_with_problem = task_info['task']
        knowledge_item = task_info['knowledge_item']
        problem_knowledge_level = task_info['knowledge_level']
        inital_knowledge = self.org_network.get_agent(agent)['expertise'][knowledge_item]
        complexity = self.architecture.nodes[self.task_network.nodes[task_with_problem]['architecture_element']]['development_complexity']
        
        # increase familiarity
        self.org_network.get_agent(agent)['knowledge_base_familiarity'][0] += ((1 - self.org_network.get_agent(agent)['knowledge_base_familiarity'][0]) * 
                                                task_data['nominal_effort'] * self.calc_supportTool_effectivness(agent, use_excess=False) * familiarity_increase_rate)
        
        if task_info['success']:
            # Event log
            if self.log_events:
                self.event_logger(f'Search for {self.org_network.knowledge_items[knowledge_item]} on knowledge base by {agent} was successfull.')

            knowledge_gain = help_fnc.calc_knowledge_gain(inital_knowledge, complexity, problem_knowledge_level, knowledge_base_effectivness=self.calc_supportTool_effectivness(agent, use_excess=False))
            self.update_expertise(agent, task_with_problem, knowledge_gain, knowledge_item)
        
        else: # search failed
            # Event log
            if self.log_events:
                self.event_logger(f'Search for {self.org_network.knowledge_items[knowledge_item]} on knowledge base by {agent} was not successfull. Searching for expert.')
            
            # check for experts
            self.check_any_expert_for_consultation(agent, task_with_problem, knowledge_item, problem_knowledge_level)


    def check_any_expert_for_consultation(self, agent, task_with_problem, knowledge_item, knowledge_req):
        
            idle_expert, expert_to_request = self.find_expert(agent, knowledge_item, knowledge_req, search_criteria='team')
            if not expert_to_request:
                idle_expert, expert_to_request = self.find_expert(agent, knowledge_item, knowledge_req, search_criteria='organization')
            
            if idle_expert: # start consultation
                self.start_consultation(task_with_problem, agent, idle_expert, knowledge_item, knowledge_req)
                
            elif expert_to_request: # request consultation
                self.add_request(expert_to_request, agent, type='Consultation', info={'task': task_with_problem, 'knowledge_item': knowledge_item, 'knowledge_level': knowledge_req})
                
            else: # no one in project has expertise reset problem resolved to continue work with unchanged expertise
                self.task_network.nodes[task_with_problem]['task_status'] = 'Problem Resolved'
                # Event log
                if self.log_events:
                    self.event_logger(f'No expert with required knowledge ({self.org_network.knowledge_items[knowledge_item]} - {knowledge_req}) exists. {agent} continuing work on {task_with_problem}.')
    

    def update_expertise(self, agent, task_with_problem, knowledge_gain, knowledge_item):
        self.task_network.nodes[task_with_problem]['task_status'] = 'Problem Resolved'
        
        self.org_network.get_agent(agent)['expertise'][knowledge_item] += knowledge_gain
        
        # update tasks in queue (considering knowledge retention except for problem task)
        for task in self.org_network.get_agent(agent)['task_queue']:
            if self.org_network.get_agent(agent)['task_queue'][task]['task_type'] == 'Technical_Work':

                tool = self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]['tool']
                efficiency, competency, problem_probability = self.get_efficiency_competency(agent, task, tool)
                
                additional_info = {'efficiency': efficiency, 'competency': competency, 'problem_probability': problem_probability}
                
                # update task
                data = self.org_network.get_agent(agent)['task_queue'][task]
                old_efficiency = data['additional_info']['efficiency']
                effort_done = data['inital_effort'] - data['remaining_effort'] 
                
                data['remaining_effort'] = data['remaining_effort'] * old_efficiency / efficiency
                data['inital_effort'] = effort_done + data['remaining_effort']
                data['additional_info'] = additional_info
                
                if old_efficiency > efficiency:
                    raise ValueError('Old efficiency is higher after knowledge gain.')
        
        # Event log
        if self.log_events:
            self.event_logger(f'{self.org_network.knowledge_items[knowledge_item]} expertise for {agent} increased by {knowledge_gain:.2f} and problem with {task_with_problem} resolved.')
        
    
    def complete_assign_task(self, agent, task):
        teams = self.org_network.get_agent(agent)['task_queue'][task]['additional_info'].get('teams', None)
        task_to_assign = self.org_network.get_agent(agent)['task_queue'][task]['additional_info']['task']
        if teams:
            selected_team = self.find_best_team(teams)
            self.activity_network.nodes[self.task_network.nodes[task_to_assign]['activity_name']]['assigned_to_team'] = selected_team
            manager = self.org_network.get_manager(team=selected_team)
            self.assign_task(manager, 
                             task_type='Assign_Task', 
                             info={'task': task_to_assign}
                             )
        else:
            team = self.org_network.get_team(agent)
            agent_to_assign = self.find_best_agent(task_to_assign, team)
            self.assign_task(agent_to_assign, 
                             task_id=task_to_assign
                             )
  
        
    def find_best_team(self, teams):
        effort_per_team = []
        for team in teams:
            team_effort_backlog = 0
            for member in self.org_network.get_members(team):
                member_effort_backlog = 0
                for task_info in self.org_network.get_agent(member)['task_queue'].values():
                    member_effort_backlog += task_info['remaining_effort']
                team_effort_backlog += member_effort_backlog
                
            effort_per_team.append((team, team_effort_backlog / len(self.org_network.get_members(team))))

        random.shuffle(effort_per_team) # shuffle to ensure randomness if tied
        sorted_teams = sorted(effort_per_team, key=lambda x: x[1])
        
        return sorted_teams[0][0] # return team with least amount of effort


    #########################################################################################################################################################
    #### Task Prioritization and Assignment #################################################################################################################
    #########################################################################################################################################################
    
    
    def add_request(self, agent, requestor, type, info):
        importance = self.task_network.nodes[info['task']]['importance']
        
        # org effects currently excluded
        #common_manager = self.org_network.get_common_manager([self.org_network.get_team(agent=requestor), self.org_network.get_team(agent)])
        #if common_manager != self.org_network.get_manager(agent=agent):
        #    org_distance = nx.shortest_path_length(self.org_network.organization, agent, common_manager)       
        #    importance = importance / (1 + importance_reduction_factor_for_external_expert * org_distance)
        
        req_information = {
            'req_id': f'{type}_{round(self.global_clock, 1)}',
            'request_type': type,
            'requestor': requestor,
            'time_of_request': self.global_clock,
            'importance': importance,
            'task': info['task']
        }
        
        
        pending_req = {
            'task': info['task'],
            'req_id': req_information['req_id'],
            'request_type': type,
            'sent_to': agent
        }
        
        match type:
            case 'Consultation':
                req_information['knowledge_item'] = info['knowledge_item']
                req_information['knowledge_level'] = info['knowledge_level']
                
                pending_req['knowledge_item'] = info['knowledge_item']
                
            case 'Information':
                req_information['req_id'] = f'{req_information['req_id']}_{info['element']}'
                pending_req['req_id'] = req_information['req_id']
                
                req_information['amount'] = info['amount']
                req_information['fraction'] = info['fraction']
                req_information['element'] = info['element']
            
            case 'Collaboration':
                req_information['req_id'] = f'{req_information['req_id']}_{info['element']}'
                pending_req['req_id'] = req_information['req_id']
                
                req_information['sub_type'] = info['sub_type']
                pending_req['sub_type'] = info['sub_type']
                
                if info['sub_type'] == 'Interface':
                    req_information['rework_percent'] = info['rework_percent']
                    pending_req['rework_percent'] = info['rework_percent']
                else:
                    req_information['perceived_feasibility'] =info['perceived_feasibility']
                    pending_req['perceived_feasibility'] =info['perceived_feasibility']
                
                req_information['element'] = info['element']
                pending_req['element'] = info['element']
            
        # update requests
        if type in {'Consultation', 'Collaboration'}:
            self.org_network.get_agent(agent)['collab_requests'].append(req_information)
            self.org_network.get_agent(requestor)['pending_collab_requests'].append(pending_req)
        else:
            self.org_network.get_agent(agent)['info_requests'].append(req_information)
            self.org_network.get_agent(requestor)['pending_info_requests'].append(pending_req)
        
        # Event log
        if self.log_events:
            self.event_logger(f'{type} from {agent} requested by {requestor}.')
    
    
    
    def assign_task(self, agent, task_id=None, task_type=None, info=None, effort=0, original_assignment_time=None):
        # Technical work in the task network
        if task_id:
            self.task_network.nodes[task_id]['task_status'] = 'Assigned'
            self.task_network.nodes[task_id]['assigned_to'] = agent
            
            tool = self.activity_network.nodes[self.task_network.nodes[task_id]['activity_name']]['tool']
            efficiency, competency, problem_probability = self.get_efficiency_competency(agent, task_id, tool)
            
            effort, nominal_effort = self.calc_actual_task_effort(task_id, efficiency)
            
            importance = self.task_network.nodes[task_id]['importance']
            info = {'efficiency': efficiency, 'competency': competency, 'problem_probability': problem_probability}
            task_type = 'Technical_Work'
            
            activity_node = self.activity_network.nodes[self.task_network.nodes[task_id]['activity_name']]
            
            # reduce effort depending on rework and info availability
            match activity_node['activity_type']:
                case 'Design' | 'System_Design':
                    total_tasks = activity_node['num_tasks']
                    remaining_tasks = total_tasks - activity_node['n_completed_tasks']
                    info_completeness = self.org_network.get_agent(agent)['product_information_completeness'][activity_node['architecture_element']][activity_node['activity_type']]
                    
                    effort *= (1 - info_completeness) * total_tasks / remaining_tasks
                    
                    if effort <= 0:
                        raise ValueError(f'Effort for {task_id} was set to zero or negative.')
                
                # apply second_order reduction, first task is changed after info is exchanged (task is assigned before that)
                case 'Prototyping' | 'Virtual_Integration':
                    if not self.task_network.nodes[task_id]['first_task']: 
                        effort *= activity_node['second_order_rework_reduction']
            
        else:
            tool = None
            nominal_effort = None
            
            
        if task_type != 'Noise':
            # get importance of dependent technical task for support tasks
            if not task_id:
                technical_task = info['task']
                importance = self.task_network.nodes[technical_task]['importance']
                activity_node = self.activity_network.nodes[self.task_network.nodes[technical_task]['activity_name']]
            
            # once an activity linked to a technical task is assigned that activity is considered in progress
            if (not (task_type == 'Share_Information' and 
                    activity_node['activity_status'] == 'Interrupted' and 
                    activity_node['n_completed_tasks'] == 0 # in the case of information sharing after quant activities fail status is not changed
                    ) and
                not (task_type in {'Share_Information', 'Receive_Information'} and activity_node['activity_status'] == 'Completed') and
                not activity_node['activity_status'] == 'Interface or Feasibility Problem'
                ):
                activity_node['activity_status'] = 'In Progress'


        if original_assignment_time:
            assignment_time = original_assignment_time
        else:
            assignment_time = self.global_clock
        

        match task_type:
            
            case 'Consultation' | 'Provide_Consultation':
                partner = info.get('agent') or info.get('expert')
                task_id = f'Consultation_with_{partner}_for_{info['task']}'
                
            case 'Collaboration':
                partner = info.get('requestor') or info.get('resp_agent')
                task_id = f'Collaboration_with_{partner}_for_{info['element']}'
             
            case 'Search_Knowledge_Base':
                req_level = info['knowledge_level']
                existing_level = self.org_network.get_agent(agent)['expertise'][info['knowledge_item']]
                amount = req_level - existing_level
                task_id = f'Search_Knowledge_Base_{round(self.global_clock, 1)}'
                access_efficiency = (self.knowledge_base['productivity'] * 
                                     self.calc_supportTool_effectivness(agent) * 
                                     self.org_network.get_agent(agent)['knowledge_base_familiarity'][0])
                nominal_effort = amount * random.triangular(knowledge_base_latency_min, knowledge_base_latency_max)
                effort = nominal_effort / access_efficiency
            
            case 'Noise':
                task_id = f'Noise_{round(self.global_clock, 1)}'
                importance = noise_importance
                if random_task_times:
                    effort = random.triangular(min_task_effort, max_task_effort)
                else:
                    effort = random.triangular(nominal_task_effort - (nominal_task_effort / 2), nominal_task_effort + (nominal_task_effort / 2))
        
            case 'Assign_Task':
                task_id = f'Assign_{info['task']}'
                effort = random.triangular(assignment_time_min, assignment_time_max)
                if info.get('teams', None):
                    effort *= len(info['teams'])
                else:
                    effort *= len(self.org_network.get_subordinates(agent)) + 1
                    
            case 'Receive_Information' | 'Share_Information':
                task_id = f'{task_type}_{info['task']}'
                
                if info.get('rework_info', False): # rework info gets increased importance
                    if task_type == 'Share_Information':
                        importance = self.task_network.nodes[info['start_condition_for'][0]]['importance']
                    else:
                        importance = self.task_network.nodes[activity_node['tasks'][0]]['importance']
                
                
                if info.get('Compatibility_Check', False):
                    parent_element = self.org_network.get_agent(agent)['responsible_element']
                    tool = self.activity_network.nodes[self.activity_network_class.generate_activity_name(parent_element, 'System_Design')]['tool']
                else:
                    tool = self.activity_network.nodes[self.task_network.nodes[info['task']]['activity_name']]['tool']

                
                interoperability = self.tools[tool].get('interoperability', 0)
                digital_literacy = self.calc_supportTool_effectivness(agent)
                productivity = (self.knowledge_base['productivity'] + self.tools[tool]['productivity']) / 2
                familiarity = self.org_network.get_agent(agent)['knowledge_base_familiarity'][0] if task_type == 'Receive_Information' else 1
                efficiency = digital_literacy * productivity * familiarity
                nominal_effort = random.triangular(info_handling_time_min, info_handling_time_max) * info['info_amount'] * (1 - interoperability)
                
                if info.get('Compatibility_Check', False):
                    verification_efficiency = self.tools[tool]['productivity'] * self.calc_EngTool_effectivness(agent, tool)
                    verification_effort = verification_effort_factor * nominal_effort / verification_efficiency
                    info['verification_effort'] = verification_effort
                else:
                    verification_effort = 0
                    
                effort = max((verification_effort + nominal_effort / efficiency), step_size)


        task_information = {
            'task_type': task_type,
            'remaining_effort': effort,
            'inital_effort': effort,
            'nominal_effort': nominal_effort,
            'time_at_assignment': assignment_time,
            'importance': importance,
            'tool': tool,
            'additional_info': info
            }
        self.org_network.get_agent(agent)['task_queue'][task_id] = task_information
        
        # Event log
        if self.log_events:
            self.event_logger(f'Task "{task_id}" was assigned to {agent}.')
        
        return task_id


    def create_assignment_tasks(self):
        tasks_ready = sorted(list(self.tasks_ready))
        self.tasks_ready = set()

        if fixed_assignments: # this is currently used
            for task in tasks_ready:
                activity_name = self.task_network.nodes[task]['activity_name']
                self.assign_task(
                    self.activity_network.nodes[activity_name]['assigned_to'],
                    task
                )
            
        else:
            if len(tasks_ready) <= 1:
                prioritized_task_list = tasks_ready
            else:
                random.shuffle(tasks_ready)
                prioritized_task_list = sorted(tasks_ready, 
                                            key=lambda t: (self.task_network.nodes[t]['importance']), 
                                            reverse=True)
            
            for task in prioritized_task_list:
                assigned_agent = self.task_network.nodes[task]['assigned_to']
                self.task_network.nodes[task]['task_status'] = 'Being Assigned'
                assigned_team = self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]['assigned_to_team']
                
                if assigned_agent:
                    self.assign_task(
                        assigned_agent,
                        task
                    )
                elif assigned_team:
                    manager = self.org_network.get_manager(team=assigned_team)
                    self.assign_task(
                        manager, 
                        task_type='Assign_Task', 
                        info={'task': task}
                    )
                else:
                    responsible_teams = self.find_responsible_teams(task)
                    if len(responsible_teams) == 1:
                        self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]['assigned_to_team'] = responsible_teams[0]
                        manager = self.org_network.get_manager(team=responsible_teams[0])
                        self.assign_task(
                            manager, 
                            task_type='Assign_Task', 
                            info={'task': task}
                        )
                    else:
                        manager = self.org_network.get_common_manager(responsible_teams)
                        self.assign_task(
                            manager, 
                            task_type='Assign_Task', 
                            info={'task': task, 'teams': responsible_teams}
                        )

                
    def find_responsible_teams(self, task):
        activity_type = self.task_network.nodes[task]['activity_type']
        architecture_element = self.task_network.nodes[task]['architecture_element']
        
        possible_teams = []
        
        for team in self.org_network.all_teams:
            # check if team has members (excluding managers)
            if any(self.org_network.get_agent(member)['profession'] != 'Manager' for member in self.org_network.get_members(team)):
                # check responsibilities
                for func_resp, prod_resp in self.org_network.organization.nodes[team]['responsibilities'].items():
                    if func_resp == activity_type:
                        for element in prod_resp:
                            if element == architecture_element:
                                possible_teams.append(team)
                                break
                        break # stop search once found
        
        return possible_teams

    
    def find_best_agent(self, task, team):
        # filter agents with responsibilities
        activity_type = self.task_network.nodes[task]['activity_type']
        possible_agents = []
        for agent in self.org_network.get_members(team):
            if activity_type in self.org_network.get_agent(agent)['responsibilities'].keys():
                if self.task_network.nodes[task]['architecture_element'] in self.org_network.get_agent(agent)['responsibilities'][activity_type]:
                    possible_agents.append(agent)

        # collect idle agents
        idle_agents = []
        for agent in possible_agents:
            if self.org_network.get_agent(agent)['state'] in {'Idle', 'Waiting'}:
                idle_agents.append(agent)
        
        
        if len(idle_agents) == 1:
            return idle_agents[0]
        
        elif len(idle_agents) == 0:
            # check workload (effort of tasks in queue)
            agent_workloads = {}
            for agent in possible_agents: 
                agent_workloads[agent] = 0
                for task_info in self.org_network.get_agent(agent)['task_queue'].values():
                    agent_workloads[agent] += task_info['remaining_effort']
            # check if more than one agent has minimum workload
            min_workload = min(agent_workloads.values())
            agents_with_min_workload = [agent for agent, workload in agent_workloads.items() if workload == min_workload]
            if len(agents_with_min_workload) == 1:
                return agents_with_min_workload[0]
            else:
                possible_agents = agents_with_min_workload
        
        elif len(idle_agents) > 1:
           possible_agents = idle_agents

        # check competency          
        agent_competencies = {} 
        for agent in possible_agents:
            tool = self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]['tool']
            agent_competencies[agent] = self.get_efficiency_competency(agent, task, tool)[1]
            
        max_competency = max(agent_competencies.values())
        agent_with_max_competency = [agent for agent, competency in agent_competencies.items() if competency == max_competency]
        return random.choice(agent_with_max_competency) # random choice if multiple possible agents


    #########################################################################################################################################################
    #### Other Simulation Functionality #####################################################################################################################
    #########################################################################################################################################################
    
    def calc_supportTool_effectivness(self, agent, use_excess=True): 
        digital_literacy = self.org_network.get_agent(agent)['digital_literacy']['EngineeringSupportTools']
        tool_complexity = self.knowledge_base['use_complexity']
        if use_excess:
            return min(1, digital_literacy / tool_complexity) + max(0, scaling_factor_excess_knowledge * (digital_literacy - tool_complexity))
        else:
            return min(1, digital_literacy / tool_complexity)
    
    def calc_EngTool_effectivness(self, agent, tool, use_excess=True):
        tool_complexity = self.tools[tool].get('use_complexity', None)
        if not tool_complexity:
            return 1
        else:
            digital_literacy = self.org_network.get_agent(agent)['digital_literacy']['EngineeringTools']
            if use_excess:
                return min(1, digital_literacy / tool_complexity) + max(0, scaling_factor_excess_knowledge * (digital_literacy - tool_complexity))
            else:
                return min(1, digital_literacy / tool_complexity)
    
    
    def find_expert(self, agent, knowledge_item, required_knowledge_level, search_criteria, architecture_element=None, only_idle=False):
        
        match search_criteria:
            case 'team':
                agent_list = self.org_network.get_members(self.org_network.get_team(agent)).copy()
            case 'organization':
                agent_list = self.org_network.all_agents.copy()
                
            case 'profession':
                profession = self.org_network.get_agent(agent)['profession']
                agent_list = []
                for member in self.org_network.all_agents:
                    if profession == self.org_network.get_agent(agent)['profession']:
                        agent_list.append(member)
                        
            case 'architecture element':
                agent_list = []
                for member in self.org_network.all_agents:
                    member_architecture_elements = []
                    for architecture_responsibilities in self.org_network.get_agent(member)['responsibilities'].values():
                        member_architecture_elements.extend(architecture_responsibilities)
                    if architecture_element in member_architecture_elements:
                        agent_list.append(member)
          
        agent_list.remove(agent) # skip the agent seeking help
        
        # check agents team for expert that are idle
        possible_experts = []
        possible_idle_experts = []
        for member in agent_list:
            if self.org_network.get_agent(member)['expertise'][knowledge_item] > required_knowledge_level:
                possible_experts.append(member)
                if self.check_if_idle(member):
                    possible_idle_experts.append(member)
        

        # choose expert
        if len(possible_idle_experts) == 1:
            idle_expert = possible_idle_experts[0]
        elif len(possible_idle_experts) > 1:
            max_expertise = max(self.org_network.get_agent(expert)['expertise'][knowledge_item] for expert in possible_idle_experts)
            top_experts = [expert for expert in possible_idle_experts if self.org_network.get_agent(expert)['expertise'][knowledge_item] == max_expertise]
            if len(top_experts) == 1:
                idle_expert = top_experts[0]
            else:
                idle_expert = random.choice(top_experts)     
        else: # if no expert in team or available
            idle_expert = None

        if not only_idle:
            if len(possible_experts) == 1:
                expert_to_request = possible_experts[0]
            elif len(possible_experts) > 1:
                max_expertise = max(self.org_network.get_agent(expert)['expertise'][knowledge_item] for expert in possible_experts)
                top_experts = [expert for expert in possible_experts if self.org_network.get_agent(expert)['expertise'][knowledge_item] == max_expertise]
                if len(top_experts) == 1:
                    expert_to_request = top_experts[0]
                else:
                    expert_to_request = random.choice(top_experts)     
            else: # if no expert in team or available
                expert_to_request = None
            
            return idle_expert, expert_to_request
        else:
            return idle_expert
    
    
    def get_efficiency_competency(self, agent, task, tool):
            knowledge_req = self.task_network.nodes[task]['knowledge_req']
            expertise = self.org_network.get_agent(agent)['expertise'].copy()
            
            if self.task_network.nodes[task]['activity_type'] in {'Prototyping'}:
                if self.architecture.nodes[self.task_network.nodes[task]['architecture_element']].get('procure', False):
                    return self.tools[tool]['productivity'], 1, 1
            
            if self.tools[tool]['type'] == 'digital':
                digital_literacy = self.org_network.get_agent(agent)['digital_literacy']['EngineeringTools']
                tool_complexity = self.tools[tool]['use_complexity']
            else: # digital literacy and tool complexity have no impact
                digital_literacy, tool_complexity = None, None
            
            tool_productivity = self.tools[tool]['productivity']
            
            return help_fnc.calc_efficiency_competency(knowledge_req, expertise, digital_literacy, tool_complexity, tool_productivity)


    def calc_actual_task_effort(self, task, efficiency):
        repetitions = self.task_network.nodes[task]['repetitions']
        learning_rate = self.task_network.nodes[task]['learning_rate']
        
        if self.task_network.nodes[task]['activity_type'] == 'Prototyping' and self.architecture.nodes[self.task_network.nodes[task]['architecture_element']]['procure']:
            nominal_effort = self.task_network.nodes[task]['nominal_effort']
        else:
            nominal_effort = self.task_network.nodes[task]['nominal_effort'] * (repetitions+1)  ** math.log(learning_rate, 2)
        return (1 / efficiency) * nominal_effort, nominal_effort


    def create_noise(self):
        for agent in self.org_network.all_agents:
            if self.global_clock >= self.noise_creation_interval[agent]:
                availability = self.org_network.get_agent(agent)['availability']
                if random.random() > availability:
                    self.assign_task(agent, task_type='Noise')
                        
                # new random intervall 
                if random_task_times:
                    new_interval = random.triangular(min_task_effort, max_task_effort)
                else:
                    new_interval = random.triangular(nominal_task_effort - (nominal_task_effort / 2), 
                                                     nominal_task_effort + (nominal_task_effort / 2)
                                                     )
                new_check_time = self.global_clock + new_interval
                self.noise_creation_interval[agent] = new_check_time

      
    def check_end_of_work_day_weekend(self):
        if round(self.global_clock, 2) % work_hours_per_day == 0:
            # Event log
            if self.log_events:
                self.event_logger('End of work day.')
            self.global_clock += 24 - work_hours_per_day
        
        day_of_week = (round(self.global_clock, 2) // 24) % 7
        if day_of_week >= work_days_per_week:
            # Event log
            if self.log_events:
                self.event_logger('End of work week.')
            
            self.global_clock += (7 - work_days_per_week) * 24

    
    def check_if_idle(self, agent):
        if not self.org_network.get_agent(agent)['task_queue']:
            return True
        else:
            for task, data in self.org_network.get_agent(agent)['task_queue'].items():
                if ((data['task_type'] == 'Technical_Work' and 
                     self.task_network.nodes[task]['task_status'] not in {'Information Needed', 'Technical Problem'} and
                     self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]['activity_status'] != 'Interface or Feasibility Problem') or
                     data['task_type'] != 'Technical_Work'):
                    return False
            return True
    
    
    def check_debugging(self):
        if self.debug_stop:
            if round(self.global_clock, 2) > round(self.debug_stop, 2):
                print(f'Debug Stop ({self.global_clock}; {self.debug_stop})')
                
                for activity, data in self.activity_network.nodes(data=True):
                    if data['activity_status'] != 'Completed':
                        print(f'{activity}: {data['activity_status']}')
                
                for task, data in self.task_network.nodes(data=True):
                    if data['task_status'] != 'Completed':
                        print(f'{task}: {data['task_status']}')
                input()  
        elif round(self.global_clock, 2) % round(self.debug_interval, 2) == 0:
            input()

                
    
    #########################################################################################################################################################
    #### Data Collection and Results ########################################################################################################################
    #########################################################################################################################################################
    
    def start_sim_log(self):

        self.log_file_name = self.save_folder + '/simulation_log.txt'
        with open(self.log_file_name, 'w') as f:
            f.write('Simulation Log \n')
            f.write('=======================================================================\n')
            f.write('=======================================================================\n')
            f.write('Architecture Elements:\n')
            f.write('=======================================================================\n')
            for node, data in self.architecture.nodes(data=True):
                f.write(f'{node}:\n')
                for key, value in data.items():
                    f.write(f'      {key}: {value}\n')
            f.write('=======================================================================\n')
            f.write('=======================================================================\n')
            f.write('Activities:\n')
            f.write('=======================================================================\n')
            for node, data in self.activity_network.nodes(data=True):
                f.write(f'{node}:\n')
                for key, value in data.items():
                    f.write(f'      {key}: {value}\n')
            f.write('=======================================================================\n')
            f.write('=======================================================================\n')
            f.write('Tasks:\n')
            f.write('=======================================================================\n')
            for node, data in self.task_network.nodes(data=True):
                f.write(f'{node}:\n')
                for key, value in data.items():
                    f.write(f'      {key}: {value}\n')
                f.write(f'      Successors: {sorted(list(self.task_network.successors(node)))}\n')
            f.write('=======================================================================\n')
            f.write('=======================================================================\n')
            f.write('Organization:\n')
            f.write('=======================================================================\n')
            for team in self.org_network.all_teams:
                members = self.org_network.get_members(team)
                f.write(f'{team}:\n')
                for member in members:
                    f.write(f'      {member}:\n')
                    data = self.org_network.get_agent(member)
                    for key, value in data.items():
                        f.write(f'              {key}: {value}\n')
            f.write('=======================================================================\n')
            f.write('=======================================================================\n')
            f.write('Tools:\n')
            f.write('=======================================================================\n')
            for tool, data in self.tools.items():
                f.write(f'{tool}:\n')
                for key, value in data.items():
                    f.write(f'      {key}: {value}\n')
            f.write('=======================================================================\n')
            f.write('=======================================================================\n')
            f.write('Simulation Event Logs:\n')
            f.write('=======================================================================\n')
            
    
    def event_logger(self, text):
        if not self.montecarlo:
            string = f'[{round(self.global_clock, 1)} hrs / {round(self.global_clock / (24 * 7), 2)} wks]: {text}'
            print(string)
            if self.slow_logs:
                time.sleep(0.01)
            with open(self.log_file_name, 'a') as f:
                f.write(string + '\n')
       
                
    def log_results(self):
        if self.log_events and not self.montecarlo:
            with open(self.log_file_name, 'a') as f:
                f.write('=======================================================================\n')
                f.write('=======================================================================\n')
                f.write('Simulation Results:\n')
                f.write('=======================================================================\n')
    
    
    def update_synchonization(self, agent, task, element):
        activity_type = self.task_network.nodes[task]['activity_type']
        if not use_only_first_rep_for_sync or (use_only_first_rep_for_sync and self.task_network.nodes[task]['repetitions'] == 1):
            sum_of_synchronizations = 0
            n_elements = 0
            for dep_element, _ in self.architecture.nodes[element]['interfaces'].items():
                dep_agent = self.architecture.nodes[dep_element]['responsible_designer']
                available_info = self.org_network.get_agent(dep_agent)['product_information_completeness'][dep_element][activity_type]
                info_comp = self.org_network.get_agent(agent)['product_information_completeness'][dep_element][activity_type]
                info_cons = self.org_network.get_agent(agent)['product_information_consitency'][dep_element][activity_type]
                
                outdated_info = info_comp * (1 - info_cons)
                missing_info = available_info - info_comp * info_cons
                        
                sum_of_synchronizations += 1 - (penalty_for_outdated_info * outdated_info + missing_info)
                
                n_elements += 1
            self.org_network.get_agent(agent)['synchronization'].append(sum_of_synchronizations / n_elements)
    
    
    def collect_data(self):
        # store time
        self.time_points.append(self.global_clock)
        
        total_effort = self.effort_tracker[-1]
        total_cost = self.cost_tracker[-1]
        total_cost_with_idle =  self.cost_tracker_with_idle[-1]
        
        #check the activities of every agent
        active_technical_tasks = set()
        active_activities = set()
        for team in self.org_network.all_teams:
            for agent in self.org_network.get_members(team):
                data = self.org_network.get_agent(agent)
                
                self.active_agents[agent].append(0.0)
                
                # effort and cost tracker
                self.personnel_tracker[agent].append(data['state'])
                if data['state'] != 'Noise': # noise not relevant for the cost and effort of a project
                    cost = step_size * data['salary'] / (52 * work_days_per_week * work_hours_per_day)
                    total_cost_with_idle += cost
                    
                    if data['state'] != 'Idle':
                        self.total_work += step_size
                        
                    if data['state'] == 'Waiting':
                        self.total_work_development_work += step_size
                    
                    if data['state'] not in {'Idle', 'Waiting'}:
                        total_effort += step_size
                        
                        # get tool cost
                        tool = data['tool_in_use']
                        if tool and data['state'] not in {'Receive_Information', 'Share_Information', 'Search_Knowledge_Base'}:
                            if 'cost_per_hour' in self.tools[tool]:  # testing and prototyping have constant tool cost for single task
                                tool_cost = self.tools[tool]['cost_per_hour'] * step_size
                            elif 'cost_per_month' in self.tools[tool]:
                                tool_cost = step_size * self.tools[tool]['cost_per_month'] / (work_hours_per_day * work_days_per_week * 4.35) # 4.35 weeks per month
                            cost += tool_cost
                        
                        if data['state'] in {'Receive_Information', 'Share_Information', 'Search_Knowledge_Base'}:
                            cost += self.knowledge_base['cost_per_month'] * step_size / (work_hours_per_day * work_days_per_week * 4.35) # 4.35 weeks per month
                        
                        total_cost += cost

                        # active tasks tracker
                        tech_task = data['technical_task']
                        active_activity = self.task_network.nodes[tech_task]['activity_name']
                        if data['state'] not in {'Check_Interface_Compatibility'}:#, 'Collaboration'}:
                            active_technical_tasks.add(tech_task)
                            active_activities.add(active_activity)
                                
                        # work/rework effort tracker
                        if self.task_network.nodes[tech_task]['repetitions'] >= 1 and not (self.task_network.nodes[tech_task]['repetitions'] == 1 
                                                                                           and self.task_network.nodes[tech_task]['task_status'] == 'Completed'):
                            self.total_rework_effort += step_size
                            self.activity_network.nodes[active_activity]['total_rework_effort'] += step_size
                        else:
                            self.total_work_effort += step_size
                            self.activity_network.nodes[active_activity]['total_work_effort'] += step_size
                        
                        # technical work / rework counter
                        if self.task_network.nodes[tech_task]['activity_type'] not in {'Prototyping', 'Testing'}:
                            self.total_work_development_work += step_size
                            
                        if data['state'] == 'Technical_Work':
                            if self.task_network.nodes[tech_task]['activity_type'] not in {'Prototyping', 'Testing'}:
                                self.total_development_work_effort += step_size
                                
                            if self.task_network.nodes[tech_task]['repetitions'] >= 1 and not (self.task_network.nodes[tech_task]['repetitions'] == 1 
                                                                                           and self.task_network.nodes[tech_task]['task_status'] == 'Completed'):
                                self.total_technical_rework_effort += step_size
                            else:
                                self.total_technical_work_effort += step_size

                        self.active_agents[agent][-1] = 1.0
                        
                        # cost breakdown
                        self.activity_network.nodes[active_activity]['cost'] += cost
                
                # effort breakdown
                match data['state']:
                    case 'Receive_Information' | 'Share_Information' | 'Search_Knowledge_Base':
                        label = 'Information Handling'
                    case 'Collaboration' | 'Provide_Consultation' | 'Consultation':
                        label = 'Collaboration'
                    case 'Assign_Task':
                        label = 'Coordination/Management'
                    case 'Technical_Work':
                        label = 'Technical Work'
                    case _:
                        label = data['state']
                        
                if data['state'] == 'Check_Interface_Compatibility': # special case for this, split effort
                    verification_to_total_effort = data['state_additional_info']['verification_to_total_effort']
                    if label not in self.effort_breakdown[agent]:
                        self.effort_breakdown[agent]['Technical Work'] = 0
                        self.effort_breakdown[agent]['Information Handling'] = 0
                    self.effort_breakdown[agent]['Technical Work'] += step_size * verification_to_total_effort
                    self.effort_breakdown[agent]['Information Handling'] += step_size * (1 - verification_to_total_effort)
                    
                else:
                    if label not in self.effort_breakdown[agent]:
                        self.effort_breakdown[agent][label] = 0
                    self.effort_breakdown[agent][label] += step_size
                        
                # effort backlog
                if not self.montecarlo:
                    effort_backlog = 0
                    for task_info in data['task_queue'].values():
                        if task_info['task_type'] != 'Noise':
                            effort_backlog += task_info['remaining_effort']
                    self.effort_backlog_agents[agent].append(effort_backlog)
            
                
        self.effort_tracker.append(total_effort)
        self.cost_tracker.append(total_cost)
        self.cost_tracker_with_idle.append(total_cost_with_idle)

        # track active activities for gantt
        for activity, activity_info in self.activity_network.nodes(data=True):
            last_state = self.gantt_tracker[activity][-1][0]
            # completed
            if activity_info['activity_status'] in {'Completed', 'Interrupted'}:   ##### make exception for information sharing
                if last_state != 'Completed':
                    self.gantt_tracker[activity].append(('Completed', self.global_clock))
            # not in progress    
            elif activity not in list(active_activities):
                if last_state in {'In Progress', 'Reworking'}:
                    self.gantt_tracker[activity].append(('Paused', self.global_clock))
            # reworking       
            elif any([self.task_network.nodes[task]['repetitions'] >= 1 and not (self.task_network.nodes[tech_task]['repetitions'] == 1 
                                                                                and self.task_network.nodes[tech_task]['task_status'] == 'Completed')
                      for task in list(active_technical_tasks) 
                      if self.task_network.nodes[task]['activity_name'] == activity and
                      last_state != 'In Progress']):
                if last_state != 'Reworking':
                    self.gantt_tracker[activity].append(('Reworking', self.global_clock))
            # in progress
            elif last_state != 'In Progress':
                self.gantt_tracker[activity].append(('In Progress', self.global_clock))
                
    
    def print_result(self, string):
        print(string)
        if self.log_events and not self.montecarlo:
            with open(self.log_file_name, 'a') as f:
                f.write(string + '\n')
                    
    
    def results(self):
        self.log_results()  
        if not self.montecarlo:
            self.data_prep_start_time = time.time()

        total_cost = self.cost_tracker[-1]
        lead_time = self.global_clock
        final_quality = self.architecture.nodes[self.overall_product_node]['overall_quality']
        
        effectivness = self.total_technical_work_effort / (self.total_technical_work_effort + self.total_technical_rework_effort)
        work_efficiency = self.total_development_work_effort / self.total_work_development_work
        
        n_total_testing = 0
        n_first_pass = 0
        cost_from_physical = 0
        total_iterations = 0
        n_total_des = 0
        for activity, act_data in self.activity_network.nodes(data=True):
            if act_data['activity_type'] in {'Design', 'System_Design'}:
                last_task = act_data['tasks'][-1]
                n_total_des += 1
                total_iterations += self.task_network.nodes[last_task]['repetitions']
            
            if act_data['activity_type'] == 'Testing':
                for task in act_data['tasks']:
                    n_total_testing += 1
                    if self.task_network.nodes[task]['repetitions'] == 1:
                        n_first_pass += 1
            
            if act_data['activity_type'] in {'Testing', 'Prototyping'}:
                cost_from_physical += act_data['cost']
        
        average_iterations = total_iterations / n_total_des
        fp_yield = n_first_pass / n_total_testing
        rel_cost_physical = cost_from_physical / total_cost
        
        
        
        sum_agent_consitencies = 0
        relevant_agents = 0
        for agent in self.org_network.all_agents:
            if self.org_network.get_agent(agent).get('responsible_element', False):
                if not self.architecture_class.get_hierarchical_children(self.org_network.get_agent(agent)['responsible_element']): # only design relevant right now
                    sum_agent_consitencies += np.mean(self.org_network.get_agent(agent)['synchronization'])
                    relevant_agents += 1
            
        average_consitency = sum_agent_consitencies / relevant_agents
        
        # skip print statements and plots in case of a monte carlo
        if self.montecarlo:
            return total_cost, lead_time, final_quality, effectivness, average_iterations, fp_yield, rel_cost_physical, work_efficiency, average_consitency
        
        
        util_over_time, average_util, overall_average_utilization = self.calculate_utilization_over_time()
        effort_backlog = self.sort_effort_backlog()
        applied_effort = self.sort_applied_effort()
        effort_breakdown, total_effort = self.sort_effort_breakdown()
        dev_cost_breakdown = self.calc_cost_breakdown()
        
        
        lead_time = help_fnc.convert_hours_to_ymd(self.global_clock)
        
        if not self.file_name_extention:
            print('\n_____________________________________________________________')
            
            print('\nResults:\n')
            self.print_result(f'    Lead Time: {lead_time[0]} year(s), {lead_time[1]} month(s), {lead_time[2]} day(s)')
            self.print_result(f'    Total Cost: ${round(self.cost_tracker[-1] / 1000, 1)}k (including idle: ${round(self.cost_tracker_with_idle[-1] / 1000, 1)}k)')
            self.print_result(f'    Final Quality: {(final_quality * 100):.1f} %')
            self.print_result(f'    Total Effort: {round(self.effort_tracker[-1] / work_hours_per_day, 1)} person-days')
            self.print_result(f'    Effectiveness: {(effectivness * 100):.1f} %')
            self.print_result(f'    Average Number of Iterations per Element: {average_iterations:.1f}') 
            self.print_result(f'    First Pass Yield: {(fp_yield * 100):.1f} %')
            self.print_result(f'    Relative Cost of Physical Testing: {(rel_cost_physical * 100):.1f} %')
            self.print_result(f'    Work Efficency: {(work_efficiency * 100):.1f} %')
            self.print_result(f'    Average Consitency: {(average_consitency * 100):.1f} %')
            
            
            # Resource Utilization
            if output_utils:
                if include_noise_in_util and simulate_noise:
                    self.print_result('\nResource Utilizations (including noise):')
                else:
                    self.print_result('\nResource Utilizations:')
                for entry, utilization in average_util.items():
                    if split_plots != 'profession':
                        self.print_result(f'     {entry}: {(utilization * 100):.1f}%')
                    else:
                        self.print_result(f'     {entry}s: {(utilization * 100):.1f}%')
            
            # Qualities
            if output_quality:
                print('\nQuality:')
                for element, data in self.architecture.nodes(data=True):
                    print(f'{element}:')
                    print(f'        Definition Quality: {data['definition_quality']:.3f}')
                    print(f'        Design Quality: {data['design_quality']:.3f}')
                    print(f'        Overall Quality: {data['overall_quality']:.3f}')
                    if data['interfaces']:
                        print('     Interface Qualities:')
                    for interface, edges in data['interfaces'].items():
                        print(f'                   To {interface}:')
                        for edge in edges:
                            print(f'                            Definition Quality{f' ({edge[0]} to {edge[1]})' if edge[1]!=interface else ''}: {self.architecture.edges[edge]['definition_quality']:.3f}') 
                            print(f'                            Design Quality{f' ({edge[0]} to {edge[1]})' if edge[1]!=interface else ''}: {self.architecture_class.calc_interface_quality(edge):.3f}') 
                            
                        
            # Learning
            if output_learning:
                self.print_result('\n Learning:')
                for agent in self.org_network.all_agents:
                    info = self.org_network.get_agent(agent)
                    self.print_result(f'     {agent}: ')
                    
                    has_learning = False
                    for i, expertise in enumerate(info['expertise']):
                        initial_expertise =  info['initial_expertise'][i]
                        if initial_expertise < expertise:
                            has_learning = True
                            self.print_result(f'        {self.org_network.knowledge_items[i]}: + {((expertise - initial_expertise) / initial_expertise * 100):.1f}%')
                        
                    for type, value in info['digital_literacy'].items():
                        initial_literacy = info['initial_digital_literacy'][type]
                        if initial_literacy < value:
                            has_learning = True
                            self.print_result(f'        {self.org_network.digital_literacy_items[i]}: + {((value - initial_literacy) / initial_literacy * 100):.1f}%')
                            
                    for i, familiarity in enumerate(info['knowledge_base_familiarity']):
                        initial_familiarity = info['initial_knowledge_base_familiarity'][i]
                        if initial_familiarity < familiarity:
                            has_learning = True
                            self.print_result(f'        {self.org_network.knowledge_bases[i]}: + {((familiarity - initial_familiarity) / initial_familiarity * 100):.1f}%')
                            
                    if not has_learning:
                        self.print_result('        No learning')
                        
                print('_____________________________________________________________\n')


        def moving_average(data):
            if not use_moving_average:
                return data
            window_size = int(moving_average_plots / step_size)
            smoothed_data = uniform_filter1d(data, size=window_size, mode='nearest')
            return smoothed_data
        
        # Convert time to weeks
        time_in_weeks = np.array(self.time_points) / (7 * 24)

        ################################################################################################################################################################################
        def abbreviate_activity(activity):
            abbreviations = {
                'Drone': 'Drn',
                'Air Frame': 'Airfr',
                'Propulsion System': 'Prop',
                'Flight Control System': 'FCS',
                'Main Body': 'Body',
                'Landing Gear': 'LG',
                'Arms': 'Arms',
                'Battery': 'Batt',
                'Motor': 'Motor',
                'Propeller': 'Prop',
                'Control Software': 'CSW',
                'Sensor Suite': 'Sens',
                'Controller': 'Ctrl',
            }
            activity_abbreviations = {
                'System_Design': 'SysD',
                'Design': 'Dsgn',
                'LF_System_Simulation': 'LFSim',
                'Component_Simulation': 'CompSim',
                'Virtual_Integration': 'VirtInt',
                'HF_System_Simulation': 'HFSim',
                'Prototyping': 'Proto',
                'Testing': 'Test',
            }
             
            for system, sys_abbr in abbreviations.items():
                if system in activity:
                    for act_type, act_abbr in activity_abbreviations.items():
                        if act_type in activity:
                            return f"{sys_abbr} {act_abbr}"
            return activity

        
       
        
        activity_starts = [(activity, states[1]) for activity, states in self.gantt_tracker.items()]
        sorted_activities = sorted(activity_starts, key=lambda x: x[1])
        y_labels = [abbreviate_activity(activity) for activity, _ in sorted_activities]
        y_positions = list(range(len(sorted_activities)))

        # Create a new figure
        fig, ax = plt.subplots(figsize=(7.5, 4))

        for idx, (activity, _) in enumerate(sorted_activities):
            states = self.gantt_tracker[activity]

            for i, (state, timestamp) in enumerate(states):
                if state in {"In Progress", "Reworking", "Paused"}:
                    start_time = timestamp / (7 * 24)
                    try:
                        end_time = states[i+1][1] / (7 * 24)
                    except IndexError:
                        warnings.warn(f'Error with Gantt Tracker for {activity}')
                        end_time = time_in_weeks[-1]

                    if state == "In Progress":
                        color = 'blue'
                    elif state == "Reworking":
                        color = 'red'
                    elif state == "Paused":
                        color = 'lightgrey'

                    ax.barh(idx, end_time - start_time, left=start_time, height=0.4, color=color)

        # Set the labels, title, and grid
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time (weeks)', labelpad=0, fontsize=12)
        ax.set_ylabel('Activities', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', labelsize=10)
        ax.set_ylim(len(sorted_activities) - 0.5, -0.5)
        ax.set_xlim(0, max(time_in_weeks))
        
        for idx, (label, tick) in enumerate(zip(ax.get_yticklabels(), ax.yaxis.get_major_ticks())):
            if idx % 2 != 0:
                # Shift odd labels and extend tick length
                label.set_x(-0.18)      # Shift label to the left
                tick.tick1line.set_visible(True)
                tick.tick1line.set_markersize(72)  # Extend tick length for odd labels
                tick.tick2line.set_visible(False)

        
        # Add the legend
        in_progress_patch = mpatches.Patch(color='blue', label='Work')
        reworking_patch = mpatches.Patch(color='red', label='Rework')
        paused_patch = mpatches.Patch(color='lightgrey', label='Paused')
        ax.legend(handles=[in_progress_patch, reworking_patch, paused_patch], loc='upper right', fontsize=10, frameon=False)
        
        fig.tight_layout(pad=0)
        
        plt.savefig(self.save_folder +  '/Gantt.svg', format='svg')
        
        plt.clf()
        plt.close()
        
        fig, ax1 = plt.subplots(figsize=(7.5, 2.2))

        # Overall Backlog
        overall_backlog = moving_average(effort_backlog['Overall'])
        ax1.plot(time_in_weeks, overall_backlog, linestyle='--', color='darkorange', label='Overall Backlog', linewidth=1)

        # Calculate x-axis limits
        x_min = 0
        x_max = max(time_in_weeks)
        ax1.set_xlim(x_min, x_max)

        # Calculate y-axis limits for ax1 (Backlog)
        backlog_min = 0
        backlog_max = math.ceil(max(overall_backlog) / 10) * 10
        ax1.set_ylim(backlog_min, backlog_max)
        ax1.set_ylabel('Effort Backlog (h)', color='darkorange', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelsize=10)
        ax1.set_xlabel('Time (weeks)', labelpad=0, fontsize=12)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Define custom ticks for ax1
        backlog_ticks = np.arange(backlog_min, backlog_max + 5, backlog_max / 5)
        ax1.set_yticks(backlog_ticks)

        # Secondary axis (Resource Utilization)
        ax2 = ax1.twinx()

        overall_utilization = moving_average(util_over_time['Overall'] * 100)
        ax2.plot(time_in_weeks, overall_utilization, linestyle='--', color='royalblue', label='Overall Utilization', linewidth=1)

        # Calculate y-axis limits for ax2 (Utilization)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Resource Utilization (%)', color='royalblue', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelsize=10)

        # Align ticks on both axes
        aligned_util_ticks = np.arange(0, 100 + 10, 20)
        ax2.set_yticks(aligned_util_ticks)

        specific_entry = 'System Team'

        # Add specific entry plots
        ax1.plot(
            time_in_weeks,
            moving_average(effort_backlog[specific_entry]),
            color='darkorange',
            label=f'{specific_entry} Backlog', 
            linewidth=1
        )
        ax2.plot(
            time_in_weeks,
            moving_average(util_over_time[specific_entry] * 100),
            color='royalblue',
            label=f'{specific_entry} Utilization', 
            linewidth=1
        )
        fig.tight_layout(pad=0)
        # Save the plot
        plt.savefig(self.save_folder + '/Effort_Util.svg', format='svg')
        
        plt.clf()
        plt.close()

        
        ########################################################################################################################################################################

        # Create a figure with GridSpec for custom layout
        fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        plt.subplots_adjust(top=2, bottom=1.9, hspace=0.3)
        
        
        # Gantt Chart
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Prepare data for the Gantt chart
        activity_starts = [(activity, states[1]) for activity, states in self.gantt_tracker.items()]

        # Sort activities based on their start index
        sorted_activities = sorted(activity_starts, key=lambda x: x[1])
        y_labels = [activity for activity, _ in sorted_activities]
        y_positions = list(range(len(sorted_activities)))
        
        for idx, (activity, _) in enumerate(sorted_activities):
            states = self.gantt_tracker[activity]

            for i, (state, timestamp) in enumerate(states):
                if state in {"In Progress", "Reworking", "Paused"}:
                    start_time = timestamp / (7 * 24)
                    try:
                        end_time = states[i+1][1] / (7 * 24)
                    except:
                        warnings.warn(f'Error with Gantt Tracker for {activity}')
                        end_time = time_in_weeks[-1]

                    if state == "In Progress":
                        color = 'blue'
                    elif state == "Reworking":
                        color = 'red'
                    elif state == "Paused":
                        color = 'lightgrey'

                    ax1.barh(idx, end_time - start_time, left=start_time, height=0.4, color=color)
                    
        ax1.set_yticks(y_positions)
        ax1.set_yticklabels(y_labels, fontsize=8)
        ax1.invert_yaxis()
        ax1.grid(axis='x', linestyle='--', alpha=0.7)
        ax1.set_title('Gantt Chart of Activities')
        ax1.set_xlabel('Time (weeks)', labelpad=0)
        ax1.set_ylabel('Activities')
        # legend
        in_progress_patch = mpatches.Patch(color='blue', label='Work')
        reworking_patch = mpatches.Patch(color='red', label='Rework')
        paused_patch = mpatches.Patch(color='lightgrey', label='Paused')
        ax1.legend(handles=[in_progress_patch, reworking_patch, paused_patch], loc='upper right', frameon=False)



        
        # Effort Backlog
        if plot_applied_effort:
            ax2 = fig.add_subplot(gs[1, 0])
            for entry, effort_data in applied_effort.items():
                if split_plots == 'profession':
                    label = f'{entry}s'
                else:
                    label = f'{entry}'
                if label in {'Suppliers', 'Supplier'}:
                    continue
                    
                if entry == 'Overall':
                    ax2.plot(time_in_weeks, moving_average(effort_data), linestyle='--', color='dimgray', label=label)
                else:
                    ax2.plot(time_in_weeks, moving_average(effort_data), label=label)
                    
            ax2.set_ylabel('Applied Effort (person-weeks)')
            ax2.set_xlabel('Time (weeks)', labelpad=0)
            ax2.grid(True)
            ax2.set_xlim(left=0)
            ax2.set_ylim(bottom=0)
            moving_avrg_string = f'moving average: {round(moving_average_plots / 24, 1)} days'
            ax2.set_title(f'Effort Applied over Time ({moving_avrg_string})')
            
        else:
            ax2 = fig.add_subplot(gs[1, 0])
            for entry, effort_data in effort_backlog.items():
                if split_plots == 'profession':
                    label = f'{entry}s'
                else:
                    label = f'{entry}'
                if label in {'Suppliers', 'Supplier'}:
                    continue
                
                if False:
                    for i, value in enumerate(effort_data):
                        active_agents = applied_effort[entry][i]
                        if active_agents == 0:
                            active_agents = 1
                        
                        effort_data[i] = value / active_agents
                
                if entry == 'Overall':
                    ax2.plot(time_in_weeks, moving_average(effort_data), linestyle='--', color='dimgray', label=label)
                else:
                    ax2.plot(time_in_weeks, moving_average(effort_data), label=label)
                    
            ax2.set_ylabel('Effort Backlog (h)')
            ax2.set_xlabel('Time (weeks)', labelpad=0)
            ax2.grid(True)
            ax2.set_xlim(left=0)
            ax2.set_ylim(bottom=0)
            moving_avrg_string = f'moving average: {round(moving_average_plots / 24, 1)} days'
            ax2.set_title(f'Effort Backlog over Time ({moving_avrg_string})')



        # Resource Utilization
        ax3 = fig.add_subplot(gs[2, 0])
        for entry, util_data in util_over_time.items():
            if split_plots == 'profession':
                label = f'{entry}s'
            else:
                label = f'{entry}'
            if label in {'Suppliers', 'Supplier'}:
                continue
            
            if entry == 'Overall':
                ax3.plot(time_in_weeks, moving_average(util_data * 100), linestyle='--', color='dimgray', label=label)
            else:    
                ax3.plot(time_in_weeks, moving_average(util_data * 100), label=label)
            
        ax3.set_ylabel('Resource Utilization (%)')
        ax3.set_xlabel('Time (weeks)', labelpad=0)
        ax3.legend(loc='lower right', bbox_to_anchor=(-0.05, 1), fontsize=9, frameon=False)
        ax3.grid(True)
        ax3.set_xlim(left=0)
        ax3.set_ylim(bottom=0)
        if include_noise_in_util and simulate_noise:
            moving_avrg_string += '; including noise'
        ax3.set_title(f'Resource Utilization over Time ({moving_avrg_string})')



        # Effort Break Down
        ax4 = fig.add_subplot(gs[0, 1]) 
        
        # exclusion of certain states
        exclude = set()
        if not include_noise_in_effort_breakdown:
            exclude.add('Noise')
        if not include_idle_in_effort_breakdown:
            exclude.add('Idle')
        
        all_states = set(k for v in effort_breakdown.values() for k in v) - exclude
        categories = list(effort_breakdown.keys())
        values = {subcategory: [effort_breakdown.get(category, {}).get(subcategory, 0) for category in categories] for subcategory in all_states}
        
        x = np.arange(len(categories))  # the label locations
        width = 0.5 
        bottom = np.zeros(len(categories))
        
        for subcategory in all_states:
            ax4.bar(
                x, 
                values[subcategory], 
                width, 
                bottom=bottom, 
                label=subcategory.replace("_", " ")
            )
            bottom += np.array(values[subcategory])
        
        ax4.set_title('Effort Breakdown')
        ax4.set_xticks(x)
        if split_plots == 'profession':
            labels = [category + 's' for category in categories]
        else:
            labels = categories
        ax4.set_xticklabels(labels, rotation=10, ha='right')
        ax4.set_ylabel('Effort (person-days)')
        ax4.legend(prop={'size': 8}, frameon=False)


        # Component Cost Breakdown
        ax5 = fig.add_subplot(gs[1, 1])

        # Filter components
        component_cost_breakdown = {}
        for element, costs in dev_cost_breakdown.items():
            if not self.architecture_class.get_hierarchical_children(element):
                
                component_cost_breakdown[element] = costs

        elements = list(component_cost_breakdown.keys())
        activities = list(next(iter(component_cost_breakdown.values())).keys())
        x = np.arange(len(elements))
        width = 0.5

        # Plot stacked bars
        bottom = np.zeros(len(elements))
        for activity in activities:
            activity_costs = [round(component_cost_breakdown[element][activity] / 1000, 1) for element in elements]
            bars = ax5.bar(x, activity_costs, width, label=activity, bottom=bottom)

            # Place labels at the center of each bar segment
            for bar, cost in zip(bars, activity_costs):
                if cost > 0:
                    ax5.text(
                        bar.get_x() + bar.get_width() / 2, 
                        bar.get_y() + bar.get_height() / 2, 
                        f'{cost}', 
                        ha='center', 
                        va='center', 
                        fontsize=8
                    )
            bottom += np.array(activity_costs)

        ax5.axvline(
            x=2.5, 
            color='black', 
            linestyle='--',
            linewidth=1
        )
        ax5.axvline(
            x=5.5, 
            color='black', 
            linestyle='--',
            linewidth=1
        )
        
        ax5.set_title('Component Cost Breakdown')
        ax5.set_ylabel('Development Cost ($k)')
        ax5.set_xticks(x)
        ax5.set_xticklabels(elements, rotation=10, ha='right')
        ax5.legend(prop={'size': 8}, frameon=False)

        # System Cost Breakdown
        ax6 = fig.add_subplot(gs[2, 1])

        # Filter elements
        system_cost_breakdown = {}
        for element, costs in dev_cost_breakdown.items():
            if self.architecture_class.get_hierarchical_children(element):
                system_cost_breakdown[element] = costs
                
                for decendent in self.architecture_class._get_all_hierarchical_descendants(element):
                    for cost_type, cost in dev_cost_breakdown[decendent].items():
                        system_cost_breakdown[element][cost_type] += cost


        elements = list(system_cost_breakdown.keys())
        activities = list(next(iter(system_cost_breakdown.values())).keys())
        x = np.arange(len(elements))

        # Plot stacked bars
        bottom = np.zeros(len(elements))
        for activity in activities:
            activity_costs = [round(system_cost_breakdown[element][activity] / 1000, 1) for element in elements]
            bars = ax6.bar(x, activity_costs, width, label=activity, bottom=bottom)

            # Place labels at the center of each bar segment
            for bar, cost in zip(bars, activity_costs):
                if cost > 0:
                    ax6.text(
                        bar.get_x() + bar.get_width() / 2, 
                        bar.get_y() + bar.get_height() / 2, 
                        f'{cost}', 
                        ha='center', 
                        va='center', 
                        fontsize=8
                    )
            bottom += np.array(activity_costs)

        ax6.axvline(
            x=0.5, 
            color='black', 
            linestyle='--',
            linewidth=1
        )
        
        ax6.set_title('System Cost Breakdown')
        ax6.set_ylabel('Development Cost ($k)')
        ax6.set_xticks(x)
        ax6.set_xticklabels(elements, rotation=45)
        ax6.legend(prop={'size': 8}, frameon=False)
        plt.tight_layout()


        self.data_prep_time = time.time() - self.data_prep_start_time
        self.total_time = time.time() -self.init_start_time
        
        if not self.file_name_extention:
            print('_____________________________________________________________\n')
            
            print(f'\nInitialization Time: {self.init_time:.2f} s')
            print(f'Simulation Time:     {self.sim_time:.2f} s')
            print(f'Data Prep Time:      {self.data_prep_time:.2f} s')
            print('____________________________')
            print(f'Total Time:          {self.total_time:.2f} s\n')
        
        
        
        if self.file_name_extention:
            plt.savefig(self.save_folder +  '/single_run_results_' + self.file_name_extention + '.png')
            plt.savefig(self.save_folder +  '/single_run_results_' + self.file_name_extention + '.svg', format='svg')
        else:
            plt.savefig(self.save_folder +  '/single_run_results.png')
            plt.savefig(self.save_folder +  '/single_run_results.svg', format='svg')
            plt.show()
            
    
    def calc_cost_breakdown(self):
        system_dev_cost_breakdown = {}
        for activity_info in self.activity_network.nodes.values():
            cost = activity_info['cost']

            # add architecture element
            architecture_element = activity_info['architecture_element']
            if architecture_element not in system_dev_cost_breakdown:
                system_dev_cost_breakdown[architecture_element] = {
                    'Development': 0,
                    'Virtual Validation': 0,
                    'Physical Validation': 0
                }
                
            # add activity
            activity_type =  activity_info['activity_type']
            match activity_type:
                case 'System_Design' | 'Design' | 'Virtual_Integration':
                    cost_type = 'Development'
                case 'LF_System_Simulation' | 'Component_Simulation' | 'HF_System_Simulation':
                    cost_type = 'Virtual Validation'
                case 'Prototyping' | 'Testing':
                    cost_type = 'Physical Validation'
                
            
            system_dev_cost_breakdown[architecture_element][cost_type] += cost
        return system_dev_cost_breakdown
    
    
    def sort_effort_breakdown(self):
        effort_breakdown = {}
        total_effort = {}
        for agent, data in self.effort_breakdown.items():
            if split_plots == 'profession':
                key = self.org_network.get_agent(agent)['profession']
            elif split_plots == 'overall':
                key = 'Overall'
            elif split_plots == 'teams':
                key = self.org_network.get_team(agent)
            if key in {'Suppliers', 'Supplier'}:
                continue
            
            if key not in effort_breakdown:
                effort_breakdown[key] = data
                total_effort[key] = 0
            else:
                for state, effort in data.items():
                    if state not in {'Idle', 'Noise'}:
                        total_effort[key] += effort
                    if state not in effort_breakdown[key]:
                        effort_breakdown[key][state] = effort
                    else:
                        effort_breakdown[key][state] += effort
                
        # Remove keys with only zero values in total_effort
        if not self.montecarlo:
            keys_to_delete = [key for key, value in total_effort.items() if value == 0]
            for key in keys_to_delete:
                del effort_breakdown[key]
                del total_effort[key]
        
        return effort_breakdown, total_effort
        
        
    def sort_applied_effort(self):
        applied_effort = {}

        for agent, data in self.active_agents.items():
            if split_plots == 'profession':
                key = self.org_network.get_agent(agent)['profession']
            elif split_plots == 'teams':
                key = self.org_network.get_team(agent)
            else:
                key = None
                
            if 'Overall' not in applied_effort:
                applied_effort['Overall'] = data
            else:
                applied_effort['Overall'] = np.add(applied_effort['Overall'], data)
            
            if key and key not in applied_effort:
                applied_effort[key] = data
            else:
                if key:
                    applied_effort[key] = np.add(applied_effort[key], data)

        return applied_effort
        
        
    def sort_effort_backlog(self):
        backlog = {}

        for agent, data in self.effort_backlog_agents.items():
            if split_plots == 'profession':
                key = self.org_network.get_agent(agent)['profession']
            elif split_plots == 'teams':
                key = self.org_network.get_team(agent)
            else:
                key = None
                
            if 'Overall' not in backlog:
                backlog['Overall'] = data
            else:
                backlog['Overall'] = np.add(backlog['Overall'], data)
            
            if key and key not in backlog:
                backlog[key] = data
            else:
                if key:
                    backlog[key] = np.add(backlog[key], data)

        return backlog


    def calculate_utilization_over_time(self):
        utilization_over_time = {}        
        total_steps = len(self.time_points)
        
        # utilization over time
        for i in range(total_steps):
            idle_count = {}
            working_count = {}
            idle_count['Overall'] = 0
            working_count['Overall'] = 0
            
            for agent, states in self.personnel_tracker.items():
                if split_plots == 'profession':
                    key = self.org_network.get_agent(agent)['profession']
                elif split_plots == 'teams':
                    key = self.org_network.get_team(agent)
                    
                if key not in idle_count:
                    idle_count[key] = 0
                    working_count[key] = 0
                
                if states[i] in {'Idle', 'Waiting'}:
                    idle_count[key] += 1
                    idle_count['Overall'] += 1
                elif not include_noise_in_util and states[i] not in {'Idle', 'Noise', 'Waiting'}:
                    working_count[key] += 1
                    working_count['Overall'] += 1
                elif include_noise_in_util and states[i] not in {'Idle', 'Waiting'}:
                    working_count[key] += 1
                    working_count['Overall'] += 1
                    
            for key in idle_count:
                if key not in utilization_over_time:
                    utilization_over_time[key] = np.zeros(total_steps)
                
                if idle_count[key] + working_count[key] != 0:
                    utilization_over_time[key][i] = working_count[key] / (idle_count[key] + working_count[key])
                else:
                    utilization_over_time[key][i] = 0 # if all agents of a profession are working on Noise
            
        # average utilization
        average_utilization = {}
        for key in utilization_over_time:
            average_utilization[key] = np.mean(utilization_over_time[key])
        overall_average_utilization = average_utilization['Overall']
        del average_utilization['Overall']
    
        return utilization_over_time, average_utilization, overall_average_utilization


# Warning Handling for debugging
def warning_handler(message, category, filename, lineno, file=None, line=None):
    print(f"Warning captured:\nMessage: {message}\nCategory: {category}\nFile: {filename}\nLine: {lineno}")
    pdb.set_trace()  # Pause execution and start the debugger


if __name__ == "__main__":
    #warnings.showwarning = warning_handler
    
    mpl.rcParams['svg.fonttype'] = 'none'
    plt.rcParams["font.family"] = "Times New Roman"
    
    sim = PDsim(
        overall_quality_goal=0.90,
        
        # Input data location (None for test data)
        #folder='Architecture/Inputs/DOE3 - New Tool/DOE3-22',
        folder='Architecture/Inputs/Baseline',
        
        # debugging
        debug=False, 
        debug_interval=None, 
        debug_stop=15488, 
        
        # logging
        enable_timeout=False,
        log_events=False,
        slow_logs=False,
        print_status=True,
        
        random_seed=None
    )
    
    sim.sim_run()
        