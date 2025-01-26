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
                 montecarlo=False, 
                 log_events=False, slow_logs=False,
                 random_seed=None):
        
        self.log_events = log_events
        self.slow_logs = slow_logs
        self.montecarlo = montecarlo
        
        self.timeout = timeout
        
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
                os.makedirs(self.save_folder)
            
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
                

        
        # project and org definition
        self.overall_quality_goal = overall_quality_goal
        self.architecture_class = ArchitectureGraph(folder=folder)
        self.architecture = self.architecture_class.architecture
        
        tools = Tools(architecture=self.architecture, folder=folder)
        self.knowledge_base = tools.knowledge_base
        self.tools = tools.tool_list
        
        self.activity_network_class = ActivityNetwork(self.architecture_class, tools)
        self.activity_network = self.activity_network_class.activity_graph
        
        self.task_network_class = TaskNetwork(
            self.activity_network_class, 
            self.architecture_class,
            randomize_structure=random_task_network, 
            randomize_task_times=random_task_times
        )
        self.task_network = self.task_network_class.task_graph
        
        self.org_network = OrganizationalGraph(architecture=self.architecture_class, folder=folder)
        
        # Consitency checks of architecture and organization --> has to be based on the json files because of knowledge vector ordering
        #consitency_check(self.architecture, self.activity_network, self.task_network, self.org_network, self.org_capabilities)
        
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



        ###### track platform usage (how often, search success rate, )

        # trackers for simulation results with inital values (0)
        self.time_points = [0]     
        self.cost_tracker = [0]
        self.cost_tracker_with_idle = [0]
        self.effort_tracker = [0]
        
        self.total_rework_effort = 0
        self.total_work_effort = 0
        
        self.effort_breakdown = {}
        self.effort_backlog_agents = {}
        self.personnel_tracker = {}
        for agent in self.org_network.all_agents:
            self.personnel_tracker[agent] = ['Idle']
            self.effort_backlog_agents[agent] = [0]
            self.effort_breakdown[agent] = {}
        self.effort_backlog_teams = {}
        for team in self.org_network.all_teams:
            self.effort_backlog_teams[team] = [0]
        
        self.activity_rework_counter = {}
        self.gantt_tracker = {}
        for activity in self.activity_network.nodes:
            self.activity_rework_counter[activity] = 0
            self.gantt_tracker[activity] = [('Not Started', 0)]
            

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
            
            
            # check for communication (out and ingoing) ---have to be checked seperatly because two agent have to work together
            # handle communication/collaboration --> higher liklyhood/priority for collaboration inside a team
            # check if activities have been competed --> review, check for rework or decision gates, consultation, search for data
            # if activity has a successor that requires only the completed task as input information and is in the responsibilities of current agent and the current agent has a small workload or check this before creating communication etc
            
            
            # coordination meeting between team members: fast and only them; coordination between different teams: slower, members + project manager / or other manager/intermediate
            
            
            # random chance to create a noise event (depending on availability of agents)
            if self.noise_creation_interval:
                self.create_noise()

            # assign tasks that are ready
            self.create_assignment_tasks()
            
            # reassignment of tasks??? --> check if there are memebers on a team that are idle and other workers have tasks that can be switched (have to change responsibility definition to the team instead of people)
            # create handoff meeting if switching person 

            
            # check assigned tasks, select one, and work on it
            completed_tasks = self.work_on_tasks()
            
            # complete finised work
            self.complete_work(completed_tasks)
            
            # time step
            self.global_clock += step_size
            
            # collect all data for trackers
            self.collect_data()
            
            
            # timer status update
            if not self.montecarlo and not self.log_events:
                print(f'\rSimulation running... ({(time.time() - start_time):.1f} s)', end="", flush=True)
            
            # timeout error
            if time.time() - start_time > self.timeout:
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
            
            data['state'] = task_info['task_type']
            
            # active associated technical task
            if task_info['task_type'] == 'Technical_Work':
                data['technical_task'] = agent_task
                data['tool_in_use'] = task_info['tool']
            elif task_info['additional_info']:
                data['technical_task'] = task_info['additional_info']['task']
                data['tool_in_use'] = ''  ####################################### tool use for information activities
            else:
                data['technical_task'] = ''
                data['tool_in_use'] = ''
            
             
            # check for technical problem 
            if task_info['task_type'] == 'Technical_Work': 
                if not self.task_network.nodes[agent_task]['activity_type'] in {'Prototyping', 'Testing'}:
                    if task_info['additional_info']['problem_probability'] > random.random():
                        self.technical_problem_with_general_knowledge(agent, agent_task)
                        continue # dont reduce effort for problem (causes problems with iteration)
            
            
            # update remaining effort
            task_info['remaining_effort'] -= step_size
            
            if task_info['remaining_effort'] <= 0:
                completed_tasks[agent] = agent_task
        
        return completed_tasks
           
            
    def select_tasks_to_work_on(self):
        tasks_to_work_on = {}
        possible_tasks = {}
        
        all_agents = self.org_network.all_agents
        random.shuffle(all_agents) # shuffled to ensure no favoratism towards any agent
        for agent in all_agents:
            data = self.org_network.get_agent(agent)
            
            # select task to work on (excluding technical tasks that have technical problems)
            possible_tasks[agent] = [task for task in list(data['task_queue'].keys()) if ((data['task_queue'][task]['task_type'] == 'Technical_Work' and 
                                                                                           self.task_network.nodes[task]['task_status'] != 'Technical Problem') or 
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
        agents_with_new_consultation = []
        for agent, tasks in list(possible_tasks.items()):
            if agent not in possible_tasks:
                continue
            
            requests = self.org_network.get_agent(agent)['requests']
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
                            continue
                    
                        
                    selected_request = sorted_requests[0][0]
                    requestor =  selected_request['requestor']
                    
                    # Event log
                    self.event_logger(f'{selected_request['request_type']} request from {requestor} accepted by {agent}')

                    # delete element from list
                    self.org_network.get_agent(agent)['requests'].remove(selected_request)

                    match selected_request['request_type']:
                        case 'Consultation':
                            task = selected_request['task']
                            knowledge_item = selected_request['knowledge_item']
                            self.start_consultation(task, requestor, agent, knowledge_item)
                            
                            agents_with_new_consultation.extend([requestor, agent])
                            del possible_tasks[requestor]
                            del possible_tasks[agent]
                                                        
                        case 'Collaboration':
                            pass
       
        # start requested collaborations
        for agent in agents_with_new_consultation:
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
                random.shuffle(tasks) # shuffle to ensure randomness in case of tie
                task_priorities  = [(task, self.calc_priority(agent, task=task)) for task in tasks]
                sorted_tasks = sorted(task_priorities, key=lambda t: t[1], reverse=True)
                selected_task = sorted_tasks[0][0]
                
                # select information receival task if it exist for that task
                if round(sorted_tasks[0][1], 5) == round(sorted_tasks[1][1], 5):
                    for task in tasks:
                        if agent_data['task_queue'][task]['task_type'] == 'Receive_Information':
                            if agent_data['task_queue'][task]['additional_info']['task'] == selected_task:
                                selected_task = task
            
            
            # check if task is technical and newly started: check if information is required
            if (selected_task and agent_data['task_queue'][selected_task]['task_type'] == 'Technical_Work' and 
                agent_data['task_queue'][selected_task]['inital_effort'] == agent_data['task_queue'][selected_task]['remaining_effort']):
                
                information_task = self.check_if_information_required(selected_task, agent)
                
                if information_task: # replace selected task if information search is required
                    selected_task = information_task
            
            # add selected task to tasks to work on dict
            if selected_task:
                tasks_to_work_on[agent] = selected_task
                
        return tasks_to_work_on
                

    def check_if_information_required(self, task, agent):
        
        activity = self.task_network.nodes[task]['activity_name']
        activity_type = self.task_network.nodes[task]['activity_type']
        architecture_element = self.task_network.nodes[task]['architecture_element']
        
        ### in rare cases double information processing happens on rework (if the first task is reworked)
        
        ####################################### if input information is incomplete use product knowledge as asspumtion

        ###################### maybe check compatability of information
        #### also check knowledge base maybe
        
        
        ####### only works for no overlapping
        
        # receive information from predecessor activities (required to start)
        if self.task_network.nodes[task]['first_task']:
            
            if activity_type in {'System_Design', 'Design'}:
                knowledge_type = 'reqs'
            else:
                knowledge_type = 'design'
            
            if (self.org_network.get_agent(agent)['product_knowledge'][knowledge_type][architecture_element] 
                >= self.knowledge_base['product_knowledge_completeness'][knowledge_type][architecture_element]):
                return None
            else:
                new_knowledge = (self.knowledge_base['product_knowledge_completeness'][knowledge_type][architecture_element]
                                      - self.org_network.get_agent(agent)['product_knowledge'][knowledge_type][architecture_element])
            
            receive_from = []
            match activity_type:
                case 'System_Design' | 'Design': # Design Activities have one element they receive direct information from
                    parent = self.architecture_class.get_parent(architecture_element)
                    if not parent: # overall system node requires no inputs
                        return None
                    
                    total_importance = sum([self.architecture.nodes[child]['req_importance'] for child in self.architecture_class.get_hierarchical_children(parent)])
                    importance = self.architecture.nodes[architecture_element]['req_importance']
                    
                    receive_from.append((self.activity_network_class.generate_activity_name(parent, 'System_Design'), importance / total_importance))
                    
                case 'LF_System_Simulation' | 'Component_Simulation' | 'HF_System_Simulation': # quantification activities have one predecessor
                    receive_from.append((next(self.activity_network.predecessors(activity)), 1))
                    
                case 'Virtual_Integration':
                    def get_information_predecessor_for_virtual_integration(element, fraction=1):
                        total_importance = sum([self.architecture.nodes[child]['req_importance'] for child in self.architecture_class.get_hierarchical_children(element)])
                        for child in self.architecture_class.get_hierarchical_children(element):
                            importance = self.architecture.nodes[child]['req_importance']
                            if self.architecture_class.get_hierarchical_children(child):                  ###################### partial virtual integration not considered
                                virtual_integration_activity = self.activity_network_class.generate_activity_name(child, 'Virtual_Integration')
                                if virtual_integration_activity in self.activity_network.nodes:
                                    receive_from.append((virtual_integration_activity, fraction * importance / total_importance))
                                else:
                                    get_information_predecessor_for_virtual_integration(child, fraction * importance / total_importance)
                            else:
                                receive_from.append((self.activity_network_class.generate_activity_name(child, 'Design'), fraction * importance / total_importance))
                    
                    receive_from.append((self.activity_network_class.generate_activity_name(architecture_element, 'System_Design'), 1))
                    get_information_predecessor_for_virtual_integration(architecture_element)

                
                case 'Prototyping':
                    if self.architecture_class.get_hierarchical_children(architecture_element):
                        virtual_integration_activity = self.activity_network_class.generate_activity_name(architecture_element, 'Virtual_Integration')
                        if virtual_integration_activity in self.activity_network.nodes:
                            receive_from.append((virtual_integration_activity, 1))
                        else:
                            receive_from.append((self.activity_network_class.generate_activity_name(architecture_element, 'System_Design'), 1))
                    else:
                        receive_from.append((self.activity_network_class.generate_activity_name(architecture_element, 'Design'), 1))

                case 'Testing': # currently receives no information (test planning activities would if they are included go here)
                    pass
            
            # information sharing amount
            info_amount = 0
            for activity, fraction in receive_from:
                info_amount += self.activity_network.nodes[activity]['effort'] * fraction
            
            if info_amount > 0:
                information_task = self.assign_task(agent, task_type='Receive_Information', info={'task': task, 'info_amount': info_amount * new_knowledge})
                return information_task
            else:
                return None
        
        # check if information from dependent elements is required
        else:
            return None
            available_information = ...
            # check definition quality and knwoledge of dependent subsystems/components (include severity and importance?)
            # use as probability if information is needed:
            if random.random() > available_information:
                
                information_search_task = ...
                
                return information_search_task
            # read information from dependent task and check if it is compatible
            else:
                return None
        
        
    
        
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
        
        # Event log
        self.event_logger(f'Technical problem ({self.org_network.knowledge_items[problem_knowledge_item]} - Req: {req_knowledge_level} Has: {expertise[problem_knowledge_item]:.2f}) with "{task}" occured for {agent}.')
        
        expert_to_consult = self.find_expert(agent, problem_knowledge_item, req_knowledge_level, search_criteria='team', only_idle=True)
        
        if expert_to_consult:
            self.start_consultation(task, agent, expert_to_consult, problem_knowledge_item)
        else: # create knowledge base search task
            
            success = help_fnc.interpolate_knowledge_base_completeness(self.knowledge_base['completeness'][problem_knowledge_item], req_knowledge_level) > random.random()
            
            if not success: # if information is not available the familiarity is used to determine if the agent checks or not
                if (min(1, self.org_network.get_agent(agent)['digital_literacy']['EngineeringSupportTools'] / self.knowledge_base['use_complexity']) * 
                    self.org_network.get_agent(agent)['knowledge_base_familiarity'][0] > random.random()
                ):
                    self.check_any_expert_for_consultation(agent, task, problem_knowledge_item, req_knowledge_level)
                    return
            
            ### sampling of digital literacy will have to be changed later once it is more complex
            original_assignment_time = self.org_network.get_agent(agent)['task_queue'][task]['time_at_assignment']
            self.assign_task(
                agent, 
                task_type='Search_Knowledge_Base',
                info={'task': task, 'knowledge_item': problem_knowledge_item, 'success': success}, 
                original_assignment_time=original_assignment_time
            )
        
        
    
    def start_consultation(self, task, agent, expert, knowledge_item):
        if agent == expert:
            raise ValueError(f'Consultation not possible: agent and expert are same ({agent})')
        
        # Event log
        self.event_logger(f'Consultation on {self.org_network.knowledge_items[knowledge_item]} with {expert} started.')
            
        consultation_effort = random.triangular(consultation_effort_min, consultation_effort_max)

        # create consultation task for expert and agent
        self.assign_task(
            expert, 
            task_type='Provide_Consultation', 
            info={'task': task, 'agent': agent}, 
            effort=consultation_effort
        )
        self.assign_task(
            agent, 
            task_type='Consultation', 
            info={'task': task, 'knowledge_item': knowledge_item, 'expert': expert, 'consultation_effort': consultation_effort},
            effort=consultation_effort
        ) 
        
        

    #########################################################################################################################################################
    #### Tasks Completion Events ############################################################################################################################
    #########################################################################################################################################################  
                
    def complete_work(self, completed_tasks):
        for agent, task in completed_tasks.items():
            agent_data = self.org_network.get_agent(agent)
            task_data = agent_data['task_queue'][task]
            
            # Event log
            self.event_logger(f'"{task}" was completed by {agent}.')
            
            match task_data['task_type']:
                case 'Technical_Work':
                    self.complete_technical_work(agent, task)
                case 'Assign_Task':
                    self.complete_assign_task(agent, task)
                case 'Consultation' | 'Provide_Consultation':
                    self.complete_consultation(agent, task_data['additional_info'])
                case 'Search_Knowledge_Base':
                    self.complete_search_knowledge_base(agent, task_data['additional_info'])
                case 'Share_Information':
                    self.complete_information_sharing(agent, task_data['additional_info'])
                case 'Receive_Information':
                    self.complete_receive_information(agent, task_data['additional_info'])
                    

            # delete from queue
            del self.org_network.get_agent(agent)['task_queue'][task]
            break # there can only be one tasks finished at a time
    
    
    def complete_technical_work(self, agent, task):
        
        def update_quality(competency_or_product_knowledge, quality_type, node=None, edge=None, updated_knowledge_needed=False):
            if edge:
                element = self.architecture.edges[edge[0], edge[1]]
                name = f'Interface of {edge[0]} to {edge[1]}'

            if node:
                element = self.architecture.nodes[node]
                name = node
            
            if updated_knowledge_needed: # if there is a previous design version quality increments are based on that version
                old_quality = element[f'previous_{quality_type}_quality']

                quality_increment_from_task = (competency_or_product_knowledge - old_quality) / n_tasks
            else:
                #perceived_quality = element[f'previous_perceived_{quality_type}_quality']  ############################## has to take previous pk of own element into account
                                                                                ###### definition quality should be updated by feedback from downstream design, rework of definition can cause rework for other subsystems aswell
                                                                                ################### max has to be deleted
                                                                                
                #quality_increment_from_task = competency_or_product_knowledge * (1 - perceived_quality) / n_tasks
                
                perceived_quality = 0.5 * max(element[f'previous_perceived_{quality_type}_quality'], element[f'previous_{quality_type}_quality'])
                quality_increment_from_task = 0.8 * competency_or_product_knowledge * (1 - perceived_quality) / n_tasks #################################################
            
            element[f'{quality_type}_quality'] += quality_increment_from_task
            
            if element[f'{quality_type}_quality'] > 3:
                element[f'{quality_type}_quality'] = 3
            
            self.event_logger(f'{quality_type.capitalize()} Quality for {name} increased by {quality_increment_from_task:.3f} (Total: {element[f'{quality_type}_quality']:.3f})')
        
        # activity information
        activity = self.task_network.nodes[task]['activity_name']
        activity_type = self.task_network.nodes[task]['activity_type']
        n_tasks = self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]['num_tasks']
        
        # architecture element information
        architecture_element = self.task_network.nodes[task]['architecture_element']
        
        # agent information
        agent_data = self.org_network.get_agent(agent)
        expertise = agent_data['expertise']
        
        # update task completion
        self.task_network.nodes[task]['task_status'] = 'Completed'
        self.task_network.nodes[task]['completed'] = True
        
        # update activity completion
        self.activity_network.nodes[activity]['n_completed_tasks'] += 1
        if self.activity_network.nodes[activity]['n_completed_tasks'] == self.activity_network.nodes[activity]['num_tasks']:
            self.activity_network.nodes[activity]['activity_status'] = 'Completed'
        
        
        # behavior for different activity types
        match activity_type:
            
            case 'System_Design': # update definition quality of hierarchical children and interfaces
                
                ##### need for information in subsystems based on definition quality (only if different person)
                
                all_components = self.architecture_class.get_all_components(architecture_element)
                children = self.architecture_class.get_hierarchical_children(architecture_element)
                for child in children:
                    knowledge_req_child = self.architecture.nodes[child]['knowledge_req']
                    competency = 1 #help_fnc.calc_efficiency_competency(knowledge_req_child, expertise)[1]  #############################################################################
                    
                    ####### problem with definition qulity --> only rarly is updated since iterations goes back to design and not iteration
                    
                    update_quality(competency, 'definition', node=child, updated_knowledge_needed=False) ################## True

    
                    ################# change this to be higher level and not look at the expertice and knowledge of the component
                    for dep_element, interfaces in self.architecture.nodes[child]['interfaces'].items():
                        for interface in interfaces:
                            if dep_element in children:
                                if interface[0] in all_components and interface[1] in all_components:
                                    competency = help_fnc.calc_efficiency_competency(self.architecture.nodes[interface[0]]['knowledge_req'], expertise)[1]
                                    product_knowledge = agent_data['product_knowledge']['design'][interface[0]]  ####################################### check maybe has to relate to the other side or combination of both
                                    ############################################################ may have to be reduced based on the distance from component (will never be fully defined on system level)
                                    contribution = ((self.architecture_class.hierarchical_distance(architecture_element, 'root') + 1)
                                                    / (self.architecture_class.hierarchical_distance(interface[0], 'root') + 1)) # high level system design contributes less
                                    competency = 1 * contribution #competency# * product_knowledge     #######################################################################
                                    
                                    update_quality(competency, 'definition', edge=interface, updated_knowledge_needed=False)    ####### how will it increase with decomposition only with repetition new knowledge is needed

                if not self.architecture_class.get_parent(architecture_element): # highest level also defines own system
                    competency = 1#agent_data['task_queue'][task]['additional_info']['competency']
                    update_quality(competency, 'definition', node=architecture_element, updated_knowledge_needed=False)                #################################### could be replaced by an input of requirments uncertainty
                
                competency = agent_data['task_queue'][task]['additional_info']['competency']
                update_quality(competency, 'design', node=architecture_element)
                
                
                ##################################### should also depend upon product knowledge which is very low at beginning or zero for never before done products
                ##################################### here definition quality might converge to fast should not be able to go to 1 --> should include quality of system design and quality of requirements for design activities
                                                                                                                ################## maybe use design qulity for quality of system design
                                                                                                                ################## req quality could also depend on quality of system design

            case 'Design':
                
                # what level does product knowledge have for the own component (maybe depending on information evloution curve and completion)
                ################################# have to be added to system design aswell
                
                # check for definition problem requiring clarification
                activity_completion = self.activity_network.nodes[activity]['n_completed_tasks'] / self.activity_network.nodes[activity]['num_tasks']
                if activity_completion > self.org_network.get_agent(agent)['product_knowledge']['reqs'][architecture_element]:
                    if random.random() > self.architecture.nodes[architecture_element]['definition_quality']:
                        pass #############################################################################################################################
                    # definition quality smaller than progression or quality, maybe also use competency
                     
                # definition quality of interfaces defines (amoung other things --> complexity) the probabilitiy of information need and the overall goodness
                # check if information to dependent elements needed
                for element in self.architecture.nodes[architecture_element]['interfaces']:
                    if random.random() > self.org_network.get_agent(agent)['product_knowledge']['design'][element]:
                        pass    ############################################################################################################################
                                        
                competency = agent_data['task_queue'][task]['additional_info']['competency']
                update_quality(competency, 'design', node=architecture_element)
            
                
                for edges in self.architecture.nodes[architecture_element]['interfaces'].values():
                    for edge in edges:
                        product_knowledge_of_interface = agent_data['product_knowledge']['design'][edge[1]] ######################################################################
                        competency = help_fnc.calc_efficiency_competency(self.architecture.nodes[edge[1]]['knowledge_req'], expertise)[1] ################# product knowledge
                        update_quality(competency, 'design', edge=edge, updated_knowledge_needed=False)  ##################################### True
                        ############################################################### has to be changed using competency for now instead of prouct knowledge
                
                # update overall quality    
                if self.task_network.nodes[task]['final_task']:  ### has to be changed if overlaping is activated
                    def_quality = self.architecture.nodes[architecture_element]['definition_quality']
                    des_quality = self.architecture.nodes[architecture_element]['design_quality']
                    self.architecture.nodes[architecture_element]['overall_quality'] = def_quality * des_quality
            
            
            case 'Prototyping' | 'Virtual_Integration':
                
                #    elements_to_exclude = []
                #if activity_type == 'Virtual Integration':
                #    pass ################################### similar to prototyping but include exclusion of elements that are not integated
                
                ############################################################## add integration problems based on interfaces 
                
                
                ######################################################### somehow have to reduce the impact of Integration quality for low importance interfaces https://chatgpt.com/c/67516d5a-5e1c-8009-a110-ab6eae1b30cb
                
                
                if self.task_network.nodes[task]['final_task']: # only final task has to calculate quality since individual tasks here do not influence the quality
                    
                    children = self.architecture_class.get_hierarchical_children(architecture_element)
                    if children:
                        
                        def_quality = self.architecture.nodes[architecture_element]['definition_quality']
                        des_quality = self.architecture.nodes[architecture_element]['design_quality']
                        
                        total_weighted_quality = 0
                        total_importance = 0
                        
                        for child in children:
                            element_importance = self.architecture.nodes[child]['req_importance']
                            total_importance += element_importance
                            element_quality = self.architecture.nodes[child]['overall_quality']
                            
                            interface_edges = [edge for interface, edges in self.architecture.nodes[child]['interfaces'].items() if interface in children for edge in edges]
                            sum_of_interface_complexities = 0
                            weighted_interface_quality = 0
                            for edge in interface_edges:
                                interface_complexity = self.architecture.edges[edge]['complexity']
                                interface_quality = self.architecture.edges[edge]['design_quality'] * self.architecture.edges[edge]['definition_quality']
                                
                                sum_of_interface_complexities += interface_complexity
                                weighted_interface_quality += interface_quality * interface_complexity
                                
                            total_weighted_quality += element_quality * element_importance #* weighted_interface_quality / sum_of_interface_complexities
                                
                        overall_quality = des_quality * def_quality * total_weighted_quality / total_importance
                    
                        self.architecture.nodes[architecture_element]['overall_quality'] = overall_quality
                    
                    self.event_logger(f'{activity_type} of {architecture_element} done. Overall Quality: {self.architecture.nodes[architecture_element]['overall_quality']:.3f}')
            
            
            case 'LF_System_Simulation' | 'Component_Simulation' | 'HF_System_Simulation' | 'Testing':
                if not self.check_quantification_success(task, agent):
                    return # skip assignment of successor tasks if failed
                
                ###################################################################################### reuse of previous knowledge of testing/simulation to increase accuracy
                
                if self.task_network.nodes[task]['final_task'] and activity_type == 'Testing':
                    self.architecture.nodes[architecture_element]['completion'] = True   ###########################################################  has to be changed
                    if not self.architecture_class.get_parent(architecture_element): ################################# completion should be set after sharing happend by decicion maker
                        return 
        
        # Share final data
        if self.task_network.nodes[task]['final_task'] and activity_type != 'Prototyping':
            self.share_information(task, agent)
        else:
            self.check_for_new_tasks(task, agent)
    
    
    def share_information(self, task, agent, share_with=[]):  ############ maybe also share information when new activities branch off or wait for them to request information
        #information_on_knowledge_base = self.knowledge_base['product_knowledge_completeness'][]    ########### has to check how much information is on the knowledgebase and how much new information is coming
        
        activity = self.task_network.nodes[task]['activity_name']
        completion_level = self.activity_network.nodes[activity]['n_completed_tasks'] / self.activity_network.nodes[activity]['num_tasks']
        
        activity_type = self.activity_network.nodes[activity]['activity_type']
        architecture_element = self.activity_network.nodes[activity]['architecture_element']
        if activity_type in {'System_Design', 'Design'}:
            knowledge_base_completeness_level = self.knowledge_base['product_knowledge_completeness']['design'][architecture_element]
            new_info_percentage = completion_level - knowledge_base_completeness_level
        else: 
            new_info_percentage = 1
        
        if share_with:
            dependent_tasks = share_with
        else:
            dependent_tasks = sorted(list(self.task_network.successors(task)))   ### has to exclude tasks of same activity

        info_amount = self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]['effort'] * new_info_percentage
        self.assign_task(agent, task_type='Share_Information', info={'task': task, 'info_amount': info_amount, 'completion_level': completion_level, 'start_condition_for': dependent_tasks})
        
        ############################## send testing information back first??? and add analysis/review time?
        ####### could be to decision making agent
    
    
    def check_for_new_tasks(self, task, agent):
        activity_type = self.task_network.nodes[task]['activity_type']
        architecture_element = self.task_network.nodes[task]['architecture_element']
        
        successors = sorted(list(self.task_network.successors(task))) # networkx functions output random orders ---> this cost me 4 days of debugging FML
        for succ in list(successors):
            if self.task_network.nodes[succ]['task_status'] not in {'Waiting', 'Rework Required'}:
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
        
        # design or overall quality
        match activity_type:
            case 'LF_System_Simulation' | 'Component_Simulation':
                preceived_quality = self.calc_quantification_result(task, agent, 'design')
            case 'HF_System_Simulation' | 'Testing':
                preceived_quality = self.calc_quantification_result(task, agent, 'overall')
        
        ####################################################################### maybe check knwoledge base for previous knowledge on this if test was unsuccessfull
        
        ##################################################################################################
        #if architecture_element == self.architecture_class.root_node:
        quality_goal = self.overall_quality_goal
        #else:
        #    quality_goal = 0.75
        ##################################################################################################
        
        if random.random() * quality_goal < preceived_quality:
            return True # quantification successfull
        else:
            # amount of rework is dependent of testing progression
            rework_percentage = 1 - self.activity_network.nodes[quant_activity]['n_completed_tasks'] / self.activity_network.nodes[quant_activity]['num_tasks']
            
            # get activity that is impacted
            directly_impacted_activities = []
            match activity_type:
                case 'Component_Simulation' | 'LF_System_Simulation': ########################################## also need to add higher order rework (subsystem sim can lead to system rework)
                    directly_impacted_activities.append(next(self.activity_network.predecessors(quant_activity))) # always have only one predecessor (design or system design)
                    
                case 'HF_System_Simulation' | 'Testing':            
                    
                    elements_to_exclude = [] ############################ have to exclude elements that are not integrated virtually
                                        
                    # get all possible descendents that could cause rework
                    possible_rework_causes = set([architecture_element])
                    descendents = self.architecture_class.get_all_hierarchical_descendants(architecture_element)
                    for descendent in descendents:
                        possible_rework_causes.add(descendent)
                        possible_rework_causes.update([edge for interface, edges in self.architecture.nodes[descendent]['interfaces'].items()
                                                       if interface in descendents for edge in edges])
                    
                    # get actual causes and weights (for selection)
                    possible_causes = []
                    for cause in possible_rework_causes:
                        if isinstance(cause, tuple): # edge
                            quality = (self.architecture.edges[cause[0], cause[1]]['design_quality'] * 
                                       self.architecture.edges[cause[0], cause[1]]['definition_quality'])
                            cause = cause[0] # cause is the outgoing node
                        else: # node
                            quality = self.architecture.nodes[cause]['design_quality'] * self.architecture.nodes[cause]['definition_quality']
                        
                        if cause not in elements_to_exclude:
                            possible_causes.append((cause, quality)) ##################################### need to consider weights (element and average of elements for interface + severity)
                        
                    # check what descendent caused problem rework (can also include original element)
                    rework_causes = set()
                    possible_causes = sorted(list(possible_causes))
                    for cause, quality in possible_causes:
                        
                        ##################################################################################################
                        #if architecture_element == self.architecture_class.root_node:
                        quality_goal = self.overall_quality_goal
                        #else:
                        #    quality_goal = 0.75
                        ##################################################################################################
                        
                        if random.random() * quality_goal > quality:
                            rework_causes.add(cause)
                    
                    # in none where selected
                    if not rework_causes:
                        total_weight = sum([1 / quality for _, quality in possible_causes])
                        normalized_weights = [(1 / quality) / total_weight for _, quality in possible_causes]
                        possible_causes = [cause for cause, _ in possible_causes]
                        selected_cause = random.choices(possible_causes, weights=normalized_weights, k=1)[0]
                        rework_causes.add(selected_cause)

                    # delete unnecessary causes
                    if len(rework_causes) > 1:
                        for cause in list(rework_causes):
                            for ancestor in self.architecture_class.get_all_ancestors(cause):
                                if ancestor in rework_causes and cause in rework_causes:
                                    rework_causes.remove(cause)

                    # get impacted activities
                    for cause in sorted(list(rework_causes)):
                        if self.architecture_class.get_hierarchical_children(cause):
                            activity_type = 'System_Design'
                        else:
                            activity_type = 'Design'
                        directly_impacted_activities.append(self.activity_network_class.generate_activity_name(cause, activity_type))
            
            
            # trigger rework and reset testing activity
            self.reset_activities([quant_activity], 'Interrupted')
            activities_to_reset = set()
            tasks_ready = []
            for impacted_activity in directly_impacted_activities:
                tasks_ready.extend(self.activity_rework(impacted_activity, rework_percentage))

                # reset knowledge
                architecture_element = self.activity_network.nodes[impacted_activity]['architecture_element']
                self.update_knowledge_quality(architecture_element, (1 - rework_percentage))
                
                # reset activities inbetween
                activity_paths = list(nx.all_simple_paths(self.activity_network, source=impacted_activity, target=quant_activity))
                activities_to_reset.update({node for path in activity_paths for node in path if node not in {impacted_activity, quant_activity}})
        
            self.reset_activities(sorted(list(activities_to_reset)), 'Rework Required')
            
            # check other activities hat have been completed or ongoing (for second order rework, that might trigger rework for ongoing work)
            self.check_successor_rework(directly_impacted_activities)
            
            # Event log
            self.event_logger(f'"{task}" failed due to quality issues. {round(rework_percentage * 100)}% of completed work of {", ".join([f'"{activity}"' for activity in directly_impacted_activities])} has to be reworked.')
            
            # information sharing task
            info_amount = (1 - rework_percentage) * self.activity_network.nodes[quant_activity]['effort']
            self.assign_task(agent, task_type='Share_Information', info={'task': task, 'info_amount': info_amount, 'start_condition_for': tasks_ready}) ###### if shared with multiple sources using different tools might have to be changed later
            
            return False # quantification not successful


    
    def calc_quantification_result(self, task, agent, type_of_quality):
        architecture_element = self.task_network.nodes[task]['architecture_element']
        activity = self.task_network.nodes[task]['activity_name']
        tool = self.org_network.get_agent(agent)['task_queue'][task]['tool']
        tool_info = self.tools[tool]

        actual_quality = self.architecture.nodes[architecture_element][f'{type_of_quality}_quality']
        
        tool_accuracy= tool_info['accuracy']
        agent_competency = self.org_network.get_agent(agent)['task_queue'][task]['additional_info']['competency']
        
        if self.tools[tool]['type'] == 'digital':
            tool_use_complexity = tool_info['use_complexity']
            agent_tool_competency = self.org_network.get_agent(agent)['digital_literacy']['EngineeringTools']
        else:
            tool_use_complexity, agent_tool_competency = 1, 1
        
        bias = (1 - tool_accuracy * agent_competency * min(1, agent_tool_competency / tool_use_complexity))
        if bias <= 0:
            perceived_quality = actual_quality
        else:
            upper_bound = actual_quality * (1 + bias)
            perceived_quality = min(random.triangular(actual_quality, upper_bound, upper_bound), 1)

        # perceived quality is the comultative mean of all perceived quality values
        previous_mean = self.architecture.nodes[architecture_element][f'perceived_{type_of_quality}_quality']
        n_tasks_completed = self.activity_network.nodes[activity]['n_completed_tasks']
        self.architecture.nodes[architecture_element][f'perceived_{type_of_quality}_quality'] = previous_mean + (perceived_quality - previous_mean) / n_tasks_completed
        
        return perceived_quality
    
    
    def check_successor_rework(self, activities): ################################################################################# might have to be adapted later to not reset
        
        dependent_elements = {}
        for activity in activities:
            architecture_element = self.activity_network.nodes[activity]['architecture_element']
            architecture_elements = [architecture_element]
            architecture_elements.append(self.architecture_class.get_all_hierarchical_descendants(architecture_element))
            architecture_elements.append(self.architecture_class.get_all_ancestors(architecture_element))
            dependent_elements[activity] = architecture_elements
        
        activities_to_reset = set()################################# only tasks that are impact not all downstream tasks ---> check
        
        for activity, act_data in self.activity_network.nodes(data=True):
            if activity not in activities:
                for org_activity in activities:
                    if (act_data['architecture_element'] in dependent_elements[org_activity] and
                        act_data['activity_status'] not in {'Waiting', 'Rework Required', 'Interrupted'} and # in {'In Progress', 'Completed'}
                        nx.has_path(self.activity_network, org_activity, activity) # if activity is downstream activity of one of the activities that has to be reset
                        ): 
                        activities_to_reset.add(activity)
                        break

        self.reset_activities(activities_to_reset, reset_reason='Rework Required')
           
                        
    def reset_activities(self, activities, reset_reason:{'Interrupted', 'Rework Required'}):   ########################### only works if activity is fully done (no overlap)
        tasks_to_be_reset = []                                                                 ########################### virtual integration should only be impacted by amount of new information (only one component of 3 is new would only require 1/3 of effort scale with importance/integration complexity)
        for activity in activities:
            self.activity_network.nodes[activity]['activity_status'] = reset_reason
            self.activity_network.nodes[activity]['n_completed_tasks'] = 0
            
            for task in self.activity_network.nodes[activity]['tasks']: 
                if self.task_network.nodes[task]['task_status'] not in {'Waiting', 'Completed', 'Rework Required'}:
                    tasks_to_be_reset.append(task)
                
                if reset_reason == 'Interrupted':
                    self.task_network.nodes[task]['task_status'] = 'Waiting'
                elif reset_reason == 'Rework Required':
                    self.task_network.nodes[task]['task_status'] = 'Rework Required'
                    
                self.task_network.nodes[task]['completed'] = False
        
        if tasks_to_be_reset:
            for agent in self.org_network.all_agents:
                for task, task_info in list(self.org_network.get_agent(agent)['task_queue'].items()):
                    if task_info['task_type'] == 'Technical_Work':
                        if task in tasks_to_be_reset:
                            del self.org_network.get_agent(agent)['task_queue'][task]
                    elif task_info['task_type'] != 'Noise':
                        linked_technical_task = task_info['additional_info']['task']
                        if linked_technical_task in tasks_to_be_reset:
                            del self.org_network.get_agent(agent)['task_queue'][task]
    
        
    
    def update_knowledge_quality(self, architecture_element, reduction_percentage):
        
        ############################################### also has to reset peoples knowledge (never lower than inital knwoledge)
        
        
        ################################### testing data (complete and incomplete) has to be reduced by amount of change if predessessor triggers rework
        ########################### for incompleted testing and simulation maybe use a value that increased the quality (80% Completion with 0.5: 0.8*0.5+0.2*1) interrupped without failure
        
        if not self.architecture_class.get_parent(architecture_element): # special case of overall system node
            self.knowledge_base['product_knowledge_completeness']['reqs'][architecture_element] *= reduction_percentage
        
        all_dependent_elements = self.architecture_class.get_all_hierarchical_descendants(architecture_element)
        
        for element in all_dependent_elements + [architecture_element]:
            # reset knowledge base
            if element != architecture_element:
                self.knowledge_base['product_knowledge_completeness']['reqs'][element] *= reduction_percentage
            self.knowledge_base['product_knowledge_completeness']['design'][architecture_element] *= reduction_percentage
            self.architecture.nodes[element]['completion'] = False
            
            # set previous qualities and reset preceived qualities of interfaces
            if not self.architecture_class.get_hierarchical_children(element):
                interfaces = [edge for edges in self.architecture.nodes[element]['interfaces'].values() for edge in edges]
                for edge in interfaces:
                    for q_type in {'definition', 'design'}:
                        self.architecture.edges[edge][f'previous_{q_type}_quality'] = self.architecture.edges[edge][f'{q_type}_quality']
                        self.architecture.edges[edge][f'previous_perceived_{q_type}_quality'] = self.architecture.edges[edge][f'perceived_{q_type}_quality']
                        self.architecture.edges[edge][f'perceived_{q_type}_quality'] = 0
            
            # set previous qualities and reset preceived qualities of elements    
            for q_type in {'definition', 'design', 'overall'}:
                self.architecture.nodes[element][f'previous_{q_type}_quality'] = self.architecture.nodes[element][f'{q_type}_quality']
                self.architecture.nodes[element][f'previous_perceived_{q_type}_quality'] = self.architecture.nodes[element][f'perceived_{q_type}_quality']
                self.architecture.nodes[element][f'perceived_{q_type}_quality'] = 0

                
    def activity_rework(self, activity, rework_percentage):
        activity_info = self.activity_network.nodes[activity]
        total_tasks = activity_info['num_tasks']
        
        if activity_info['activity_status'] == 'Completed':
            activity_info['activity_status'] = 'Rework Required'
            
            n_tasks_to_rework = max(round(rework_percentage * total_tasks), 1)

            # get tasks to be reworked
            tasks = activity_info['tasks'].copy()
            tasks_with_no_rework = []

            for _ in range(total_tasks - n_tasks_to_rework):
                task = tasks.pop(0)
                tasks_with_no_rework.append(task)
            tasks_to_be_reworked = tasks
                
        else:
            # get completed tasks ################################################## has to reset ongoing work
            completed_tasks = []
            for task in activity_info['tasks']:
                if self.task_network.nodes[task]['completed']:
                    completed_tasks.append(task)
            n_completed = len(completed_tasks)
            
            n_tasks_to_rework = max(round(rework_percentage * n_completed), 1)
            
            tasks_with_no_rework = []
            for _ in range(n_completed - n_tasks_to_rework):
                task = completed_tasks.pop(0)
                tasks_with_no_rework.append(task)
            tasks_to_be_reworked = completed_tasks
        
        # reset activity and tasks to be reworked
        self.activity_network.nodes[activity]['n_completed_tasks'] -= n_tasks_to_rework
        for task in tasks_to_be_reworked:
            self.task_network.nodes[task]['task_status'] = 'Rework Required'
            self.task_network.nodes[task]['completed'] = False
        
        # get tasks to be started next
        tasks_ready = set()
        if len(tasks_with_no_rework) == 0:
            tasks_ready.add(activity_info['tasks'][0])
        else:
            for task in tasks_with_no_rework:
                successors = sorted(list(self.task_network.successors(task))) # networkx functions output random orders ---> this cost me 4 days of debugging FML
                for succ in successors:
                    if succ in tasks_to_be_reworked:
                        if all(self.task_network.nodes[pred]['completed'] for pred in self.task_network.predecessors(succ)):
                            tasks_ready.add(succ)
        
        return sorted(list(tasks_ready))
    
                    
    def complete_information_sharing(self, agent, task_info):
        architecture_element = self.task_network.nodes[task_info['task']]['architecture_element']
        activity = self.task_network.nodes[task_info['task']]['activity_name']
        activity_type = self.activity_network.nodes[activity]['activity_type']
        
        # update knowledge base    
        if activity_type in {'System_Design', 'Design'}:
            completion_level = task_info['completion_level']
            
            # requirements
            for child in self.architecture_class.get_hierarchical_children(architecture_element):
                self.knowledge_base['product_knowledge_completeness']['reqs'][child] = completion_level  ########################## has to be changed for versions
            if not self.architecture_class.get_parent(architecture_element): # highest level system design also defines requirements
                self.knowledge_base['product_knowledge_completeness']['reqs'][architecture_element] = completion_level
            
            # design
            self.knowledge_base['product_knowledge_completeness']['design'][architecture_element] = completion_level
        
        info_amount = task_info['info_amount']
        n_tasks = len(task_info['start_condition_for'])
        for task in task_info['start_condition_for']:
            if self.task_network.nodes[task]['task_status'] in {'Waiting', 'Rework Required'}:
                if all(self.task_network.nodes[pred]['completed'] for pred in self.task_network.predecessors(task)):
                    self.tasks_ready.add(task)
                    
                    # rework information for tasks that were already assigned
                    assigned_to_agent = self.task_network.nodes[task]['assigned_to']
                    if assigned_to_agent:
                        self.assign_task(assigned_to_agent, task_type='Receive_Information', info={'task': task, 'info_amount': (info_amount / n_tasks)})
            
            
    def complete_receive_information(self, agent, task_info):
        architecture_element = self.task_network.nodes[task_info['task']]['architecture_element']
        
        if self.task_network.nodes[task_info['task']]['activity_type'] in {'System_Design', 'Design'}:    ######## information sharing with dependent nodes has to be changed
            information_available = self.knowledge_base['product_knowledge_completeness']['reqs'][architecture_element] ##### has to be changed for versions
            self.org_network.get_agent(agent)['product_knowledge']['reqs'][architecture_element] = information_available
        else:
            information_available = self.knowledge_base['product_knowledge_completeness']['design'][architecture_element] ##### has to be changed for versions
            self.org_network.get_agent(agent)['product_knowledge']['design'][architecture_element] = information_available ### have to add versions to all of this
        
    
    def complete_consultation(self, agent, task_info):
        # do nothing if the expert task finishes
        if not task_info.get('expert', None):
            return
        
        # calculation of new knowledge level
        knowledge_item = task_info['knowledge_item']
        expert = task_info['expert']
        task_with_problem = task_info['task']
        consultation_effort = task_info['consultation_effort']
        complexity = self.architecture.nodes[self.task_network.nodes[task_with_problem]['architecture_element']]['development_complexity']
        inital_knowledge = self.org_network.get_agent(agent)['expertise'][knowledge_item]
        expert_knowledge = self.org_network.get_agent(expert)['expertise'][knowledge_item]
        
        # Event log
        self.event_logger(f'Consultation of {agent} on {self.org_network.knowledge_items[knowledge_item]} with {expert} was completed.')
        
        knowledge_gain = help_fnc.calc_knowledge_gain(inital_knowledge, complexity, expert_knowledge)
        self.update_expertise(agent, task_with_problem, knowledge_gain, knowledge_item, knowledge_retention_expert_consultation)
        
    
    
    def complete_search_knowledge_base(self, agent, task_info):
        task_with_problem = task_info['task']
        knowledge_item = task_info['knowledge_item']
        inital_knowledge = self.org_network.get_agent(agent)['expertise'][knowledge_item]
        knowledge_req = self.task_network.nodes[task_with_problem]['knowledge_req'][knowledge_item]
        complexity = self.architecture.nodes[self.task_network.nodes[task_with_problem]['architecture_element']]['development_complexity']
        
        # increase familiarity
        self.org_network.get_agent(agent)['knowledge_base_familiarity'][0] += ((1 - self.org_network.get_agent(agent)['knowledge_base_familiarity'][0]) * 
                                                                            min(1, self.org_network.get_agent(agent)['digital_literacy']['EngineeringSupportTools'] 
                                                                            / self.knowledge_base['use_complexity']) * knowledge_retention_knowledge_base)
        
        if task_info['success']:
            # Event log
            self.event_logger(f'Search for {self.org_network.knowledge_items[knowledge_item]} on knowledge base by {agent} was successfull.')
            
            effectivness = (min(1, self.org_network.get_agent(agent)['digital_literacy']['EngineeringSupportTools'] / self.knowledge_base['use_complexity']) * 
                            help_fnc.interpolate_knowledge_base_completeness(self.knowledge_base['completeness'][knowledge_item], knowledge_req))
            knowledge_gain = help_fnc.calc_knowledge_gain(inital_knowledge, complexity, knowledge_req, effectivness)
            self.update_expertise(agent, task_with_problem, knowledge_gain, knowledge_item, knowledge_retention_knowledge_base)
        
        else: # search failed
            # Event log
            self.event_logger(f'Search for {self.org_network.knowledge_items[knowledge_item]} on knowledge base by {agent} was not successfull. Searching for expert.')
            
            # check for experts
            self.check_any_expert_for_consultation(agent, task_with_problem, knowledge_item, knowledge_req)


    def check_any_expert_for_consultation(self, agent, task_with_problem, knowledge_item, knowledge_req):
        
            idle_expert, expert_to_request = self.find_expert(agent, knowledge_item, knowledge_req, search_criteria='team')
            if not expert_to_request:
                idle_expert, expert_to_request = self.find_expert(agent, knowledge_item, knowledge_req, search_criteria='organization')
            
            if idle_expert: # start consultation
                self.start_consultation(task_with_problem, agent, idle_expert, knowledge_item)
                
            elif expert_to_request: # request consultation
                self.add_request(expert_to_request, agent, type='Consultation', info={'task': task_with_problem, 'knowledge_item': knowledge_item})
                
            else: # no one in project has expertise reset problem resolved to continue work with unchanged expertise
                self.task_network.nodes[task_with_problem]['task_status'] = 'Problem Resolved'
                # Event log
                self.event_logger(f'No expert with required knowledge ({self.org_network.knowledge_items[knowledge_item]} - {knowledge_req}) exists. {agent} continuing work on {task_with_problem}.')
    


    def update_expertise(self, agent, task_with_problem, knowledge_gain, knowledge_item, knowledge_retention_factor):
        self.task_network.nodes[task_with_problem]['task_status'] = 'Problem Resolved'
        
        # update tasks in queue (considering knowledge retention except for problem task)
        for task in self.org_network.get_agent(agent)['task_queue']:
            if self.org_network.get_agent(agent)['task_queue'][task]['task_type'] == 'Technical_Work':
                if task == task_with_problem:
                    retention = 1
                else:
                    retention = knowledge_retention_factor

                expertise_increase = knowledge_gain * retention
                tool = self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]['tool']
                efficiency, competency, problem_probability = self.get_efficiency_competency(agent, task, tool, (knowledge_item, expertise_increase))
                
                new_effort = self.calc_actual_task_effort(task, efficiency)
                additional_info = {'efficiency': efficiency, 'competency': competency, 'problem_probability': problem_probability}
                
                # update task
                data = self.org_network.get_agent(agent)['task_queue'][task]
                
                effort_reduction = new_effort / data['inital_effort']
                effort_done = data['inital_effort'] - data['remaining_effort']
                data['remaining_effort'] = data['remaining_effort'] * effort_reduction
                data['inital_effort'] = effort_done + data['remaining_effort']
                data['additional_info'] = additional_info
        
        # Knowledge retention
        self.org_network.get_agent(agent)['expertise'][knowledge_item] += knowledge_gain * knowledge_retention_factor
        
        # Event log
        self.event_logger(f'{self.org_network.knowledge_items[knowledge_item]} expertise for {agent} increased by {knowledge_gain:.2f} (Retained: {(knowledge_retention_factor * knowledge_gain):.2f}) and problem with {task_with_problem} resolved.')
        
    
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
        
        common_manager = self.org_network.get_common_manager([self.org_network.get_team(agent=requestor), self.org_network.get_team(agent)])
        if common_manager != self.org_network.get_manager(agent=agent):
            org_distance = nx.shortest_path_length(self.org_network.organization, agent, common_manager)       
            importance = importance / (1 + importance_reduction_factor_for_external_expert * org_distance)
        
        req_information = {
            'request_type': type,
            'requestor': requestor,
            'time_of_request': self.global_clock,
            'importance': importance,
            'task': info['task']
        }
        
        match type:
            case 'Consultation':
                req_information['task'] = info['task']
                req_information['knowledge_item'] = info['knowledge_item']
                
            case 'Information':
                pass
            
            case 'Collaboration':
                pass
            
        # update requests
        self.org_network.get_agent(agent)['requests'].append(req_information)
        
        # Event log
        self.event_logger(f'{type} from {agent} requested by {requestor}.')
    
    
    
    def assign_task(self, agent, task_id=None, task_type=None, info=None, effort=0, original_assignment_time=None):
        # Technical work in the task network
        if task_id:
            self.task_network.nodes[task_id]['task_status'] = 'Assigned'
            self.task_network.nodes[task_id]['assigned_to'] = agent
            
            tool = self.activity_network.nodes[self.task_network.nodes[task_id]['activity_name']]['tool']
            efficiency, competency, problem_probability = self.get_efficiency_competency(agent, task_id, tool)
            
            effort = self.calc_actual_task_effort(task_id, efficiency)
            importance = self.task_network.nodes[task_id]['importance']
            info = {'efficiency': efficiency, 'competency': competency, 'problem_probability': problem_probability}
            task_type = 'Technical_Work'
            
            self.task_network.nodes[task_id]['repetitions'] += 1
            
        # get importance of dependent technical task for support tasks
        if task_type != 'Noise':
            if not task_id:
                technical_task = info['task']
                importance = self.task_network.nodes[technical_task]['importance']
            else:
                technical_task = task_id

            # once an activity linked to a technical task is assigned that activity is considered in progress
            activity = self.activity_network.nodes[self.task_network.nodes[technical_task]['activity_name']]
            if (not (task_type == 'Share_Information' and 
                    activity['activity_status'] == 'Interrupted' and 
                    activity['n_completed_tasks'] == 0 # in the case of information sharing after quant activities fail status is not changed
                    ) and
                not (task_type == 'Share_Information' and activity['activity_status'] == 'Completed')
                ):
                activity['activity_status'] = 'In Progress'

        if not task_id:
            tool = None

        if original_assignment_time:
            assignment_time = original_assignment_time
        else:
            assignment_time = self.global_clock
        

        match task_type:
            
            case 'Consultation' | 'Provide_Consultation':
                partner = info.get('agent') or info.get('expert')
                task_id = f'Consultation_with_{partner}_for_{info['task']}'
                effort = effort
                
            case 'Search_Knowledge_Base':  ################################################## adapt later to include multiple
                task_id = f'Search_Knowledge_Base_{round(self.global_clock, 1)}'
                access_efficiency = (self.knowledge_base['productivity'] * 
                                     min(1, self.org_network.get_agent(agent)['digital_literacy']['EngineeringSupportTools'] / self.knowledge_base['use_complexity']) * 
                                     self.org_network.get_agent(agent)['knowledge_base_familiarity'][0])
                effort = random.triangular(knowledge_base_latency_min, knowledge_base_latency_max) / access_efficiency
            
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
                
                ################################################ interoperability is 0 for physical tools since it it not defined
                interoperability = self.tools[self.activity_network.nodes[self.task_network.nodes[info['task']]['activity_name']]['tool']].get('interoperability', 0)
                digital_literacy = min(1, self.org_network.get_agent(agent)['digital_literacy']['EngineeringSupportTools'] / self.knowledge_base['use_complexity'])

                if interoperability != 1: # avoid division by 0
                    efficiency = digital_literacy * self.knowledge_base['productivity'] / (1 - interoperability)
                    effort = random.triangular(info_handling_time_min, info_handling_time_max) * info['info_amount'] / efficiency
                else: # interoperability of 1 would mean near instant handling of data
                    effort = step_size
                
                
        
        task_information = {
            'task_type': task_type,
            'remaining_effort': effort,
            'inital_effort': effort,
            'time_at_assignment': assignment_time,
            'importance': importance,
            'tool': tool,
            'additional_info': info
            }
        self.org_network.get_agent(agent)['task_queue'][task_id] = task_information
        
        # Event log
        self.event_logger(f'Task "{task_id}" was assigned to {agent}.')
        
        return task_id


    def create_assignment_tasks(self):
        tasks_ready = sorted(list(self.tasks_ready))
        self.tasks_ready = set()

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
    
    
    def get_efficiency_competency(self, agent, task, tool, knowledge_increase=None):
            knowledge_req = self.task_network.nodes[task]['knowledge_req']
            expertise = self.org_network.get_agent(agent)['expertise'].copy()
            
            if knowledge_increase:
                expertise[knowledge_increase[0]] += knowledge_increase[1]
            
            if self.tools[tool]['type'] == 'digital':
                digital_literacy = self.org_network.get_agent(agent)['digital_literacy']['EngineeringTools']
                tool_complexity = self.tools[tool]['use_complexity']
            else: # digital literacy and tool complexity have no impact
                digital_literacy = 1
                tool_complexity = 1
            
            tool_productivity = self.tools[tool]['productivity']
            
            return help_fnc.calc_efficiency_competency(knowledge_req, expertise, digital_literacy, tool_complexity, tool_productivity)


    def calc_actual_task_effort(self, task, efficiency):
        repetitions = self.task_network.nodes[task]['repetitions']
        learning_rate = self.task_network.nodes[task]['learning_rate']
        return (1 / efficiency) * self.task_network.nodes[task]['nominal_effort'] * (repetitions+1)  ** math.log(learning_rate, 2)


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
            self.event_logger('End of work day.')
            self.global_clock += 24 - work_hours_per_day
        
        day_of_week = (round(self.global_clock, 2) // 24) % 7
        if day_of_week >= work_days_per_week:
            # Event log
            self.event_logger('End of work week.')
            
            self.global_clock += (7 - work_days_per_week) * 24

    
    def check_if_idle(self, agent):
        if not self.org_network.get_agent(agent)['task_queue']:
            return True
        else:
            for task, data in self.org_network.get_agent(agent)['task_queue'].items():
                if ((data['task_type'] == 'Technical_Work' and 
                     self.task_network.nodes[task]['task_status'] != 'Technical Problem') or 
                     data['task_type'] != 'Technical_Work'):
                    return False
            return True
    
    
    def check_debugging(self):
        if self.debug_stop:
            if round(self.global_clock, 2) > round(self.debug_stop, 2):
                print(f'Debug Stop ({self.global_clock}; {self.debug_stop})')
                
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
        if self.log_events and not self.montecarlo:
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
            team_effort_backlog = 0
            for agent in self.org_network.get_members(team):
                data = self.org_network.get_agent(agent)
                
                # effort and cost tracker
                self.personnel_tracker[agent].append(data['state'])
                if data['state'] != 'Noise': # noise not relevant for the cost and effort of a project
                    cost = step_size * data['salary'] / (52 * work_days_per_week * work_hours_per_day)
                    total_cost_with_idle += cost
                    if data['state'] not in {'Idle', 'Waiting'}:
                        total_effort += step_size
                        
                        # get tool cost
                        if data['state'] == 'Technical_Work':
                            tool = data['tool_in_use']
                            if tool:
                                if 'cost_per_hour' in self.tools[tool]:
                                    tool_cost = self.tools[tool]['cost_per_hour'] * step_size
                                elif 'cost_per_month' in self.tools[tool]:
                                    tool_cost = step_size * self.tools[tool]['cost_per_month'] / (work_hours_per_day * work_days_per_week * 4.35) # weeks per month
                                cost += tool_cost
                        
                        total_cost += cost
                
                        # active tasks tracker
                        tech_task = data['technical_task']
                        active_technical_tasks.add(tech_task)
                        active_activity = self.task_network.nodes[tech_task]['activity_name']
                        active_activities.add(active_activity)
                        
                        # work/rework effort tracker
                        if self.task_network.nodes[tech_task]['repetitions'] > 1:
                            self.total_rework_effort += step_size
                            self.activity_network.nodes[active_activity]['total_rework_effort'] += step_size
                        else:
                            self.total_work_effort += step_size
                            self.activity_network.nodes[active_activity]['total_work_effort'] += step_size
                        
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
                    case _:
                        label = data['state']
                        
                if label not in self.effort_breakdown[agent]:
                    self.effort_breakdown[agent][label] = 0
                self.effort_breakdown[agent][label] += step_size
                        
                # effort backlog
                effort_backlog = 0
                for task_info in data['task_queue'].values():
                    if task_info['task_type'] != 'Noise':
                        effort_backlog += task_info['remaining_effort']
                self.effort_backlog_agents[agent].append(effort_backlog)
                team_effort_backlog += effort_backlog
            self.effort_backlog_teams[team].append(team_effort_backlog)
            
                
        self.effort_tracker.append(total_effort)
        self.cost_tracker.append(total_cost)
        self.cost_tracker_with_idle.append(total_cost_with_idle)

        # track active activities for gantt
        for activity, activity_info in self.activity_network.nodes(data=True):
            last_state = self.gantt_tracker[activity][-1][0]
            # completed
            if activity_info['activity_status'] in {'Completed', 'Interrupted'}:
                if last_state != 'Completed':
                    self.gantt_tracker[activity].append(('Completed', self.global_clock))
            # not in progress    
            elif activity not in list(active_activities):
                if last_state in {'In Progress', 'Reworking'}:
                    self.gantt_tracker[activity].append(('Paused', self.global_clock))
            # reworking       
            elif any([self.task_network.nodes[task]['repetitions'] > 1
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
        
        util_over_time, average_util, overall_average_utilization = self.calculate_utilization_over_time()
        effort_backlog = self.sort_effort_backlog()
        effort_breakdown, total_effort = self.sort_effort_breakdown()
        dev_cost_breakdown = self.calc_cost_breakdown()

        # skip print statements and plots in case of a monte carlo
        if self.montecarlo:
            lead_time = self.global_clock
            return lead_time, self.cost_tracker[-1], average_util, overall_average_utilization, total_effort
        
        
        lead_time = help_fnc.convert_hours_to_ymd(self.global_clock)
        
        if not self.file_name_extention:
            print('\n_____________________________________________________________')
            
            print('\nResults:\n')
            self.print_result(f'Lead Time: {lead_time[0]} year(s), {lead_time[1]} month(s), {lead_time[2]} day(s)')
            self.print_result(f'Total Cost: ${round(self.cost_tracker[-1] / 1000, 1)}k')
            self.print_result(f'Total Cost (including idle): ${round(self.cost_tracker_with_idle[-1] / 1000, 1)}k')
            self.print_result(f'Total Effort: {round(self.effort_tracker[-1] / work_hours_per_day, 1)} person-days')
            self.print_result(f'Rework Percentage: {(self.total_rework_effort / (self.total_rework_effort + self.total_work_effort) * 100):.1f} %')
            
            # Resource Utilization
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
                        print(f'                            Design Quality{f' ({edge[0]} to {edge[1]})' if edge[1]!=interface else ''}: {self.architecture.edges[edge]['design_quality']:.3f}') 
                            
                        
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
                        
                    for i, literacy in enumerate(info['digital_literacy']):
                        initial_literacy = info['initial_digital_literacy'][i]
                        if initial_literacy < literacy:
                            has_learning = True
                            self.print_result(f'        {self.org_network.digital_literacy_items[i]}: + {((literacy - initial_literacy) / initial_literacy * 100):.1f}%')
                            
                    for i, familiarity in enumerate(info['knowledge_base_familiarity']):
                        initial_familiarity = info['initial_knowledge_base_familiarity']
                        if initial_familiarity[i] < familiarity:
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
        ax.legend(handles=[in_progress_patch, reworking_patch, paused_patch], loc='upper right', fontsize=10)
        
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
        ax1.legend(handles=[in_progress_patch, reworking_patch, paused_patch], loc='upper right')



        
        # Effort Backlog
        ax2 = fig.add_subplot(gs[1, 0])
        for entry, effort_data in effort_backlog.items():
            if split_plots == 'profession':
                label = f'{entry}s'
            else:
                label = f'{entry}'
                
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
                
            if entry == 'Overall':
                ax3.plot(time_in_weeks, moving_average(util_data * 100), linestyle='--', color='dimgray', label=label)
            else:    
                ax3.plot(time_in_weeks, moving_average(util_data * 100), label=label)
            
        ax3.set_ylabel('Resource Utilization (%)')
        ax3.set_xlabel('Time (weeks)', labelpad=0)
        ax3.legend(loc='lower right', bbox_to_anchor=(-0.05, 1), fontsize=9)
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
        ax4.legend(ncol=3, prop={'size': 8})


        # Component Cost Breakdown
        ax5 = fig.add_subplot(gs[1, 1])

        # Filter components
        component_cost_breakdown = {}
        for element, costs in dev_cost_breakdown.items():
            if not self.architecture_class.get_hierarchical_children(element):
                del costs['System_Design']
                del costs['Virtual_Integration']
                del costs['LF_System_Simulation']
                del costs['HF_System_Simulation']
                
                component_cost_breakdown[element] = costs
                
                # Add cost to parent element
                parent_element = self.architecture_class.get_parent(element)
                if 'Subsys./Comp. Dev.' not in dev_cost_breakdown[parent_element]:
                    dev_cost_breakdown[parent_element]['Subsys./Comp. Dev.'] = 0
                total_cost = sum(costs.values())
                dev_cost_breakdown[parent_element]['Subsys./Comp. Dev.'] += total_cost

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

        ax5.set_title('Component Cost Breakdown')
        ax5.set_ylabel('Development Cost ($k)')
        ax5.set_xticks(x)
        ax5.set_xticklabels(elements, rotation=10, ha='right')
        ax5.legend(prop={'size': 8})

        # System Cost Breakdown
        ax6 = fig.add_subplot(gs[2, 1])

        # Filter elements
        system_cost_breakdown = {}
        for element, costs in dev_cost_breakdown.items():
            if self.architecture_class.get_hierarchical_children(element):
                del costs['Component_Simulation']
                del costs['Design']
                
                system_cost_breakdown[element] = costs

                # Add cost to parent element
                parent_element = self.architecture_class.get_parent(element)
                if parent_element:
                    if 'Subsys./Comp. Dev.' not in dev_cost_breakdown[parent_element]:
                        dev_cost_breakdown[parent_element]['Subsys./Comp. Dev.'] = 0
                    total_cost = sum(costs.values())
                    dev_cost_breakdown[parent_element]['Subsys./Comp. Dev.'] += total_cost

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

        ax6.set_title('System Cost Breakdown')
        ax6.set_ylabel('Development Cost ($k)')
        ax6.set_xticks(x)
        ax6.set_xticklabels(elements, rotation=45)
        ax6.legend(ncol=2, prop={'size': 8})
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

    def plot_and_save_results(self,
    gantt_tracker, effort_backlog, util_over_time, effort_breakdown, dev_cost_breakdown,
    architecture_class
    ):
        
        def moving_average(data):
            if not use_moving_average:
                return data
            window_size = int(moving_average_plots / step_size)
            smoothed_data = uniform_filter1d(data, size=window_size, mode='nearest')
            return smoothed_data
        
        def save_individual_plot(fig, subplot_title, save_folder, file_name_extension=None, format='png'):
            if file_name_extension:
                file_name = f"{save_folder}/{subplot_title.replace(' ', '_')}_{file_name_extension}.{format}"
            else:
                file_name = f"{save_folder}/{subplot_title.replace(' ', '_')}.{format}"
            fig.savefig(file_name, format=format)
            plt.close(fig)
            
        time_in_weeks = np.array(self.time_points) / (7 * 24)
        
        subplot_titles = [
        "Gantt Chart of Activities", "Effort Backlog over Time", "Resource Utilization over Time",
        "Effort Breakdown", "Component Cost Breakdown", "System Cost Breakdown"
        ]
        
        for subplot_title in subplot_titles:
            fig_subplot, ax_subplot = plt.subplots(figsize=(8, 6))

            if subplot_title == "Gantt Chart of Activities":
                pass
        
        
    
    def calc_cost_breakdown(self):
        system_dev_cost_breakdown = {}
        for activity_info in self.activity_network.nodes.values():
            cost = activity_info['cost']

            # add architecture element
            architecture_element = activity_info['architecture_element']
            if architecture_element not in system_dev_cost_breakdown:
                system_dev_cost_breakdown[architecture_element] = {
                    'System_Design': 0,
                    'LF_System_Simulation': 0,
                    'Design': 0,
                    'Component_Simulation': 0,
                    'Virtual_Integration': 0,
                    'HF_System_Simulation': 0,
                    'Prototyping': 0,
                    'Testing': 0,                 
                }
            
            # add activity
            activity_type =  activity_info['activity_type']
            system_dev_cost_breakdown[architecture_element][activity_type] += cost
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

###### Warning Handling
def warning_handler(message, category, filename, lineno, file=None, line=None):
    print(f"Warning captured:\nMessage: {message}\nCategory: {category}\nFile: {filename}\nLine: {lineno}")
    pdb.set_trace()  # Pause execution and start the debugger


if __name__ == "__main__":
    #warnings.showwarning = warning_handler
    
    mpl.rcParams['svg.fonttype'] = 'none'
    plt.rcParams["font.family"] = "Times New Roman"
    
    sim = PDsim(
        overall_quality_goal=0.95,
        
        folder='Architecture/Inputs/Baseline',
        
        debug=False, 
        debug_interval=100, 
        debug_stop=12000, 
        
        log_events=False,
        slow_logs=False, 
        
        random_seed=None
    )
    
    sim.sim_run()
        