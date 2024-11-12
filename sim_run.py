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

# Classes
from architecture_graph import ArchitectureGraph
from organization_graph import OrganizationalGraph
from activity_network import ActivityNetwork
from task_network import TaskNetwork
from tools import Tools

# Functions
from sim_helper_functions import convert_hours_to_ymd, calc_efficiency_competency, calc_knowledge_gain, interpolate_knowledge_base_completeness, consitency_check

# Parameters
from Inputs.sim_settings import *
from Inputs.tuning_params import *
from Inputs.sim_inputs import *

# Knowledge Base
from organizational_capabilities import knowledge_base

class PDsim:
    def __init__(self, debug=False, debug_interval=100, debug_stop=None, montecarlo=False, log_events=False, random_seed=None):
        
        self.log_events = log_events
        self.montecarlo = montecarlo
        
        if not self.montecarlo:
            # timestamp and folder for results and output files
            timestamp = time.time()
            dt_object = datetime.datetime.fromtimestamp(timestamp)
            self.formatted_time = dt_object.strftime("%Y-%m-%d_%H-%M-%S")
            self.folder_name = 'sim_runs/single_run_at_' + self.formatted_time
            os.makedirs(self.folder_name)
            
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
        self.architecture_graph = ArchitectureGraph(test_data=use_test_data)
        self.architecture = self.architecture_graph.architecture
        
        tools = Tools(test_data=use_test_data)
        self.tools = tools.tool_list
        self.knowledge_base = knowledge_base
        
        self.activity_network_class = ActivityNetwork(self.architecture_graph, tools)
        self.activity_network = self.activity_network_class.activity_graph
        
        self.task_network_class = TaskNetwork(
            self.activity_network_class, 
            self.architecture_graph,
            randomize_structure=random_task_network, 
            randomize_task_times=random_task_times
        )
        self.task_network = self.task_network_class.task_graph
        
        self.org_network = OrganizationalGraph(test_data=use_test_data)
        
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
        self.overall_product_node = self.architecture_graph.root_node
        
        # start task is ready
        first_tasks = [task for task in self.task_network.nodes if not any(self.task_network.predecessors(task))]
        self.tasks_ready = set(first_tasks)



        ###### track platform usage (how often, search success rate, )

        # trackers for simulation results with inital values (0)
        self.time_points = [0]     
        self.cost_tracker = [0]
        self.cost_tracker_with_idle = [0]
        self.effort_tracker = [0]
        
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



    #########################################################################################################################################################
    #### Simulation Run #####################################################################################################################################
    #########################################################################################################################################################
    
    def sim_run(self):
        
        if self.montecarlo is False:
            start_time = time.time()
        
        self.global_clock = 0
        
        # execution until product is fully developed
        while self.architecture.nodes[self.overall_product_node]['completion'] != 1:

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
            self.work_on_tasks()
            # work on tasks: define quality based on their expertise, knowledge
            # if input information is incomplete use product knowledge as asspumtion
            # what level does product knowledge have for the own component (maybe depending on information evloution curve and completion)
            
            
            # check for completed work
            self.check_completed_work()
            
            # time step
            self.global_clock += step_size
            
            # collect all data for trackers
            self.collect_data()
            
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
                data['tool_in_use'] = ''
            else:
                data['technical_task'] = ''
                data['tool_in_use'] = ''
            
            if data['technical_task']:
                activity = self.task_network.nodes[data['technical_task']]['activity_name']
                if self.activity_network.nodes[activity]['activity_status'] in {'Interrupted', 'Waiting'}:
                    self.activity_network.nodes[activity]['activity_status'] = 'In Progress'
                if self.activity_network.nodes[activity]['activity_status'] == 'Rework Needed':
                    self.activity_network.nodes[activity]['activity_status'] = 'Reworking'
                    
            
            if task_info['task_type'] == 'Technical_Work':  
                # check for technical problem 
                if task_info['additional_info']['problem_probability'] > random.random():
                    self.technical_problem(agent, agent_task)
                    continue # dont reduce effort for problem
                    
            # update remaining effort
            task_info['remaining_effort'] -= step_size
           
            
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
            
            requests = self.org_network.get_agent(agent)['requests'].copy()
            if requests and not any(self.org_network.get_agent(agent)['task_queue'][task]['remaining_effort'] 
                                    < self.org_network.get_agent(agent)['task_queue'][task]['inital_effort']
                                    for task in tasks):
                
                filtered_requests = [r for r in requests if r['requestor'] in possible_tasks]
                        
                if filtered_requests:
                    random.shuffle(filtered_requests)
                    request_priorities = [(request, self.calc_priority(agent, request=request)) for request in filtered_requests]
                    sorted_requests = sorted(request_priorities, key=lambda r: r[1], reverse=True)
                    
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
                            
                        case 'Collaboration':
                            pass
       
        # start requested collaborations
        for agent in agents_with_new_consultation:
            for task in self.org_network.get_agent(agent)['task_queue']:
                task_info = self.org_network.get_agent(agent)['task_queue'][task]
                if task_info['task_type'] in {'Collaboration', 'Consultation', 'Provid_ Consultation'}:
                    tasks_to_work_on[agent] = task
                    del possible_tasks[agent]
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
                selected_task = sorted(task_priorities, key=lambda t: t[1], reverse=True)[0][0]
            
            # check if task is technical --> newly started: check if information is required
            if selected_task and self.org_network.get_agent(agent)['task_queue'][selected_task]['task_type'] == 'Technical_Work':
                information_search_task = self.check_if_information_required(selected_task, agent)
                if information_search_task:
                    selected_task = information_search_task
            
            if selected_task:
                tasks_to_work_on[agent] = selected_task
                
        return tasks_to_work_on
                
    
    def check_if_information_required(self, task, agent): ## information required should be related to dependency strength and ratio (anteil an gesamt system)
        task_info = self.task_network.nodes[task]         #### also check knowledge base maybe
        activity_name = task_info['activity_name']
        activity_info = self.activity_network.nodes[activity_name]###################################################################################################
        
        return None
        available_information = ...
        # check knowledge of requirements and of dependent subsystems
        # use as probability if information is needed:
        if random.random() > available_information:
            
            information_search_task = ...
            
            return information_search_task
        
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
                
        elif request:                                                  ########################## could include a factor that reduces priority depending on requester - agent relationship
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
    
    
    def technical_problem(self, agent, task):
        
        ##### different behavior depending on the type of task: design --> redo; iterate back to different activity
        
        # change task status
        self.task_network.nodes[task]['task_status'] = 'Technical Problem'
        
        # random sample to select knowledge item that created the problem
        expertise = self.org_network.get_agent(agent)['expertise']
        knowledge_req = self.task_network.nodes[task]['knowledge_req']
        problem_knowledge_item = random.choice([i for i in range(len(expertise)) if expertise[i] < knowledge_req[i]])
        
        # Event log
        self.event_logger(f'Technical problem ({self.org_network.knowledge_items[problem_knowledge_item]} - Req: {knowledge_req[problem_knowledge_item]} Has: {expertise[problem_knowledge_item]:.2f}) with "{task}" occured for {agent}.')
        
        # check agents team for expert that are idle
        possible_experts = []
        for member in self.org_network.get_members(self.org_network.get_team(agent)):
            if member == agent:
                continue
            if self.org_network.get_agent(member)['expertise'][problem_knowledge_item] > knowledge_req[problem_knowledge_item] and self.check_if_idle(member):
                    possible_experts.append(member)
        
        # choose expert
        if len(possible_experts) == 1:
            expert_to_consult = possible_experts[0]
        elif len(possible_experts) > 1:
            max_expertise = max(self.org_network.get_agent(expert)['expertise'][problem_knowledge_item] for expert in possible_experts)
            top_experts = [expert for expert in possible_experts if self.org_network.get_agent(expert)['expertise'][problem_knowledge_item] == max_expertise]
            if len(top_experts) == 1:
                expert_to_consult = top_experts[0]
            else:
                expert_to_consult = random.choice(top_experts)     
        else: # if no expert in team or available
            expert_to_consult = None
        
        if expert_to_consult:
            self.start_consultation(task, agent, expert_to_consult, problem_knowledge_item)
        else: # create knowledge base search task
            original_assignment_time = self.org_network.get_agent(agent)['task_queue'][task]['time_at_assignment']
            self.assign_task(
                agent, 
                task_type='Search_Knowledge_Base',
                info={'task': task, 'knowledge_item': problem_knowledge_item}, 
                original_assignment_time=original_assignment_time
            )
        
        
    
    def start_consultation(self, task, agent, expert, knowledge_item):
        # Event log
        self.event_logger(f'Consultation on {self.org_network.knowledge_items[knowledge_item]} with {expert} started.')
            
        consultation_effort = random.triangular(consultation_effort_min, consultation_effort_max)
        
        # delay if expert and agent in different teams
        if self.org_network.get_team(agent) == self.org_network.get_team(expert):
            effort = consultation_effort
        else:
            common_manager = self.org_network.get_common_manager([self.org_network.get_team(agent=agent), self.org_network.get_team(expert)])
            org_distance = nx.shortest_path_length(self.org_network.organization, agent, common_manager)                                        ############## change
            effort = (delay_factor_for_external_expert) * consultation_effort # increased by organizational distance
            
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
            info={'task': task, 'knowledge_item': knowledge_item, 'expert': expert, 'consultation_effort': consultation_effort},
            effort=effort
        ) 
        
        

    #########################################################################################################################################################
    #### Tasks Completion Events ############################################################################################################################
    #########################################################################################################################################################  
                
    def check_completed_work(self):
        for agent in self.org_network.all_agents:
            agent_data = self.org_network.get_agent(agent)
            for task, task_data in list(agent_data['task_queue'].items()):
                if task_data['remaining_effort'] <= 0:

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

                    # delete from queue
                    del self.org_network.get_agent(agent)['task_queue'][task]
                    break # there can only be one tasks finished at a time
    
    
    def complete_technical_work(self, agent, task):
        
        def update_quality(competency_or_product_knowledge, quality_type, node=None, edge=None, updated_knowledge_needed=False):
            if edge:
                element = self.architecture.edges[edge[0], edge[1]]
            if node:
                element = self.architecture.nodes[node]
            
            if updated_knowledge_needed:
                if design_version != 0: # if there is a previous design version quality increments are based on that version
                    old_quality = element[f'{quality_type}_quality'][design_version-1]
                else:
                    old_quality = 0
                quality_increment_from_task = (competency_or_product_knowledge - old_quality) / n_tasks
            else:
                if design_version != 0:
                    perceived_quality = element[f'perceived_{quality_type}_quality'][design_version-1]
                else:
                    perceived_quality = 0
                quality_increment_from_task = competency_or_product_knowledge * (1 - perceived_quality) / n_tasks
            
            element[f'{quality_type}_quality'][design_version] += quality_increment_from_task
        
        # activity information
        activity = self.task_network.nodes[task]['activity_name']
        activity_type = self.task_network.nodes[task]['activity_type']
        n_tasks = self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]['num_tasks']
        
        # architecture element information
        architecture_element = self.task_network.nodes[task]['architecture_element']
        design_version = 0 ########################################################## 
        
        # agent information
        agent_data = self.org_network.get_agent(agent)
        expertise = agent_data['expertise']
        
        # update task completion
        self.task_network.nodes[task]['repetitions'] += 1
        self.task_network.nodes[task]['task_status'] = 'Completed'
        self.task_network.nodes[task]['completed'] = True
        
        # update activity completion
        self.activity_network.nodes[activity]['n_completed_tasks'] += 1
        if self.activity_network.nodes[activity]['n_completed_tasks'] == self.activity_network.nodes[activity]['num_tasks']:
            self.activity_network.nodes[activity]['activity_status'] = 'Completed'
        
        
        # behavior for different activity types
        match activity_type:
            
            case 'System_Design': # update definition quality of hierarchical children and interfaces
                all_components = self.architecture_graph.get_all_components(architecture_element)
                
                for child in self.architecture_graph.get_hierarchical_children(architecture_element):
                    knowledge_req_child = self.architecture.nodes[child]['knowledge_req']
                    competency = calc_efficiency_competency(knowledge_req_child, expertise)[1]
                    
                    update_quality(competency, 'definition', node=child)

                    for interface in self.architecture.nodes[child]['interfaces'].values():

                        if interface[0] in all_components and interface[1] in all_components:
                            competency_1 = calc_efficiency_competency(self.architecture.nodes[interface[0]]['knowledge_req'], expertise)[1]
                            product_knowledge_1 = agent_data['product_knowledge']['Design_Knowledge'][interface[0]]
                            competency_2 = calc_efficiency_competency(self.architecture.nodes[interface[1]]['knowledge_req'], expertise)[1]
                            product_knowledge_2 = agent_data['product_knowledge']['Design_Knowledge'][interface[1]]
                            competency = (competency_1 + competency_2 + product_knowledge_1 + product_knowledge_2) / 4
                            
                            update_quality(competency, 'definition', edge=interface, updated_knowledge_needed=True)

                if not self.architecture_graph.get_parent(architecture_element): # highest level also defines own system
                    competency = agent_data['task_queue'][task]['additional_info']['competency']
                    update_quality(competency, 'definition', node=architecture_element)
                    
                ##################################### should also depend upon product knowledge which is very low at beginning or zero for never before done products


            case 'LF_System_Simulation':
                perceived_definition_quality = self.calc_quantification_result(task, agent, 'definition_quality', design_version)
                
                if not self.check_quantification_success(task, perceived_definition_quality):
                    return # skip assignment of successor tasks


            case 'Design':
                competency = agent_data['task_queue'][task]['additional_info']['competency']
                update_quality(competency, 'design', node=architecture_element)
                
                for interface in self.architecture.nodes[architecture_element]['interfaces'].values():
                    product_knowledge_of_interface = agent_data['product_knowledge']['Design_Knowledge'][interface[1]]
                    update_quality(product_knowledge_of_interface, 'design', edge=interface, updated_knowledge_needed=True)
                
                # definition quality of interfaces defines (amoung other things --> complexity) the probabilitiy of information need and the overall goodness
            
            
            case 'Component_Simulation':
                perceived_design_quality = self.calc_quantification_result(task, agent, 'design_quality', design_version)
                
                if not self.check_quantification_success(task, perceived_design_quality):     ############################## send testing information back first??? and add analysis/review time?
                    return # skip assignment of successor tasks                                                                 ####### could be to decision making agent

            
            
            case 'Virtual Integration' | 'Prototyping':
                pass
            
            
            case 'HF_System_Simulation':
                pass # self.architecture.nodes[architecture_element]['overall_quality']
            
            
            case 'Testing':
                
                if self.task_network.nodes[task]['final_task'] and self.task_network.nodes[task]['activity_type'] == 'Testing':
                    self.architecture.nodes[architecture_element]['completion'] = 1   ###########################################################  has to be changed
        
        
        self.check_for_new_tasks(task, agent)
            
    
    def check_for_new_tasks(self, task, agent):
        activity_type = self.task_network.nodes[task]['activity_type']
        architecture_element = self.task_network.nodes[task]['architecture_element']
        
        successors = sorted(list(self.task_network.successors(task))) # networkx functions output random orders ---> this cost me 4 days of debugging FML
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
    
    
    def check_quantification_success(self, task, preceived_quality):
        activity = self.task_network.nodes[task]['activity_name']
        activity_type = self.task_network.nodes[task]['activity_type']
        
        # get activities that would be impacted by rework
        match activity_type:
            case 'Component_Simulation' | 'LF_System_Simulation': ########################################## also need to add higher order rework (subsystem sim can lead to system rework)
                directly_impacted_activity = list(self.activity_network.predecessors(activity))[0]
                
            case 'HF_System Simulation' | 'Testing': #########################################################
                pass
            
            
        ######## requirements importance to track higher level requirments (high importance will lead to a higher likelihood to reveal issues with the )
        
        # maybe check knwoledge base for previous knowledge on this if test was unsuccessfull
        
        if random.random() > preceived_quality:
            # amount of rework is dependent of testing progression
            rework_percentage = 1 - self.activity_network.nodes[activity]['n_completed_tasks'] / self.activity_network.nodes[activity]['num_tasks']
            
            # trigger rework and reset testing activity
            self.activity_rework(directly_impacted_activity, rework_percentage)
            self.reset_quantification_activity(activity)
                
            # Event log
            self.event_logger(f'"{task}" failed due to quality issues. {round(rework_percentage * 100)}% of completed work of "{directly_impacted_activity}" is being reworked.')

            return False
        else:
            return True


    
    def calc_quantification_result(self, task, agent, type_of_quality, version):
        architecture_element = self.task_network.nodes[task]['architecture_element']
        activity = self.task_network.nodes[task]['activity_name']
        tool = self.org_network.get_agent(agent)['task_queue'][task]['tool']
        tool_info = self.tools[tool]

        actual_quality = self.architecture.nodes[architecture_element][type_of_quality][version]
        
        tool_certainty = tool_info['certainty']
        agent_competency = self.org_network.get_agent(agent)['task_queue'][task]['additional_info']['competency']
        
        bias = (1 - tool_certainty * agent_competency)
        upper_bound = actual_quality * (1 + bias)
        perceived_quality = min(random.triangular(actual_quality, upper_bound, upper_bound), 1)

        self.task_network.nodes[task]['quality'] = perceived_quality 
        task_quality_values = [self.task_network.nodes[t]['quality'] for t in self.activity_network.nodes[activity]['tasks'] if self.task_network.nodes[t]['quality']]
        self.architecture.nodes[architecture_element][f'perceived_{type_of_quality}'][version] = np.mean(task_quality_values)
        
        return perceived_quality
    
       
    def activity_rework(self, activity, rework_percentage):
        activity_info = self.activity_network.nodes[activity]
        total_tasks = activity_info['num_tasks']
        
        if activity_info['activity_status'] == 'Completed':
            activity_info['activity_status'] = 'Rework Needed'
            
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
        if len(tasks_with_no_rework) == 0:
            self.tasks_ready.add(activity_info['tasks'][0])
        else:
            for task in tasks_with_no_rework:
                successors = sorted(list(self.task_network.successors(task))) # networkx functions output random orders ---> this cost me 4 days of debugging FML
                for succ in successors:
                    if succ in tasks_to_be_reworked:
                        if all(self.task_network.nodes[pred]['completed'] for pred in self.task_network.predecessors(succ)):
                            self.tasks_ready.add(succ)
    
    
    def reset_quantification_activity(self, activity):
        self.activity_network.nodes[activity]['activity_status'] = 'Interrupted'
        self.activity_network.nodes[activity]['n_completed_tasks'] = 0

        tasks_to_be_reset = []
        for task in self.activity_network.nodes[activity]['tasks']: 
            if self.task_network.nodes[task]['task_status'] not in {'Waiting', 'Completed'}:
                tasks_to_be_reset.append(task)
            
            self.task_network.nodes[task]['task_status'] = 'Waiting'
            self.task_network.nodes[task]['completed'] = False
            self.task_network.nodes[task]['quality'] = None
                
        # delete all ongoing tasks from task queues of agents
        for agent in self.org_network.all_agents:
            for task, task_info in list(self.org_network.get_agent(agent)['task_queue'].items()):
                if task_info['task_type'] == 'Technical_Work':
                    if task in tasks_to_be_reset:
                        del self.org_network.get_agent(agent)['task_queue'][task]
                elif task_info['task_type'] != 'Noise':
                    linked_technical_task = task_info['additional_info']['task']
                    if linked_technical_task in tasks_to_be_reset:
                        del self.org_network.get_agent(agent)['task_queue'][task]
                    
    
    
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
        
        knowledge_gain = calc_knowledge_gain(inital_knowledge, consultation_effort, complexity, expert_knowledge)
        self.update_expertise(agent, task_with_problem, knowledge_gain, knowledge_item, knowledge_retention_expert_consultation)
        
    
    
    def complete_search_knowledge_base(self, agent, task_info):
        task_with_problem = task_info['task']
        knowledge_item = task_info['knowledge_item']
        inital_knowledge = self.org_network.get_agent(agent)['expertise'][knowledge_item]
        knowledge_req = self.task_network.nodes[task_with_problem]['knowledge_req'][knowledge_item]
        complexity = self.architecture.nodes[self.task_network.nodes[task_with_problem]['architecture_element']]['development_complexity']
        
        ### sampling of digital literacy will have to be changed later once it is more complex
        
        success_probability = (interpolate_knowledge_base_completeness(knowledge_base['completeness'][knowledge_item], knowledge_req) 
                               * self.org_network.get_agent(agent)['digital_literacy']['EngineeringSupportTools'] / upper_limit_knowledge_scale)
        if success_probability > random.random():
            # Event log
            self.event_logger(f'Search for {self.org_network.knowledge_items[knowledge_item]} on knowledge base by {agent} was successfull.')
            
            knowledge_gain = calc_knowledge_gain(inital_knowledge, 1, complexity, knowledge_req)
            self.update_expertise(agent, task_with_problem, knowledge_gain, knowledge_item, knowledge_retention_knowledge_base)
        
        else: # search failed
            # Event log
            self.event_logger(f'Search for {self.org_network.knowledge_items[knowledge_item]} on knowledge base by {agent} was not successfull.')
            
            # check team for experts
            possible_experts = []
            possible_idle_experts = []
            for member in self.org_network.get_members(self.org_network.get_team(agent)):
                if member == agent:
                    continue
                if self.org_network.get_agent(member)['expertise'][knowledge_item] > knowledge_req:
                    possible_experts.append(member)
                    if self.check_if_idle(member):
                        possible_idle_experts.append(member)
            
            # check whole project for experts
            if not possible_experts: 
                for member in self.org_network.all_agents:
                    if member == agent:
                        continue
                    if self.org_network.get_agent(member)['expertise'][knowledge_item] > knowledge_req:
                        possible_experts.append(member)
                        if self.check_if_idle(member):
                            possible_idle_experts.append(member)

            # start consultation
            if possible_idle_experts:
                selected_expert = random.choice(possible_idle_experts)
                self.start_consultation(task_with_problem, agent, selected_expert, knowledge_item)
            
            # request consultation
            elif possible_experts:
                selected_expert = random.choice(possible_experts)
                self.add_request(selected_expert, agent, type='Consultation', info={'task': task_with_problem, 'knowledge_item': knowledge_item}) 

            # no one in project has expertise reset problem resolved to continue work with unchanged expertise
            else:
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
            self.activity_network.nodes[self.task_network.nodes[task]['activity_name']]['assigned_to_team'] = selected_team
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
        effort_per_person_per_team = [
            (team, self.effort_backlog_teams[team] / len(self.org_network.get_members(team)))           ############################################# have to get effort directly
            for team in teams]
        random.shuffle(effort_per_person_per_team) # shuffle to ensure randomness if tied
        sorted_teams = sorted(effort_per_person_per_team, key=lambda x: x[1])
        
        return sorted_teams[0][0] # return team with least amount of effort

    #########################################################################################################################################################
    #### Task Prioritization and Assignment #################################################################################################################
    #########################################################################################################################################################
    
    
    def add_request(self, agent, requestor, type, info):
        importance = self.task_network.nodes[info['task']]['importance']
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
            efficiency, competency, problem_probability = self.get_efficiency_competency( agent, task_id, tool)
            
            effort = self.calc_actual_task_effort(task_id, efficiency)
            importance = self.task_network.nodes[task_id]['importance']
            info = {'efficiency': efficiency, 'competency': competency, 'problem_probability': problem_probability}
            task_type = 'Technical_Work'
            
            
        # get importance of dependent technical task for support tasks
        elif task_type != 'Noise':
            depdendent_technical_task = info['task']
            importance = self.task_network.nodes[depdendent_technical_task]['importance']

        if not task_id:
            tool = None

        if original_assignment_time:
            assignment_time = original_assignment_time
        else:
            assignment_time = self.global_clock
        

        match task_type:
            
            case 'Consultation' | 'Provide_Consultation':
                partner = info.get('agent') or info.get('expert')
                task_id = f'Consultation_with_{partner}'
                effort = effort
                
            case 'Search_Knowledge_Base':  ################################################## adapt later to include multiple
                task_id = f'Search_Knowledge_Base_{round(self.global_clock, 1)}'
                access_efficiency = 1 / (self.org_network.get_agent(agent)['digital_literacy']['EngineeringSupportTools'] / upper_limit_knowledge_scale * self.org_network.get_agent(agent)['knowledge_base_familiarity'][0])
                effort = random.triangular(knowledge_base_latency_bc, knowledge_base_latency_wc, knowledge_base_latency_ml) * access_efficiency
            
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


    def create_assignment_tasks(self):
        tasks_ready = sorted(list(self.tasks_ready))
        self.tasks_ready = set()

        if len(tasks_ready) <= 1:
            prioritized_task_list =  tasks_ready
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
            
            return calc_efficiency_competency(knowledge_req, expertise, digital_literacy, tool_complexity, tool_productivity)


    def calc_actual_task_effort(self, task, efficiency):
        repetitions = self.task_network.nodes[task]['repetitions']
        learning_factor = self.task_network.nodes[task]['learning_factor']
        return (1 / efficiency) * self.task_network.nodes[task]['nominal_effort'] * (repetitions+1)  ** math.log(learning_factor, 2)


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
            if round(self.global_clock, 2) == round(self.debug_stop, 2):
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

        self.log_file_name = self.folder_name + '/simulation_log_' + self.formatted_time + '.txt'
        with open(self.log_file_name, 'w') as f:
            f.write('Simulation Log ' + self.formatted_time + '\n')
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
                                tool_cost = self.tools[tool].get('cost_per_hour', 0) * step_size
                                cost += tool_cost
                        
                        total_cost += cost
                
                        # active tasks tracker
                        tech_task = data['technical_task']
                        active_technical_tasks.add(tech_task)
                        active_activities.add(self.task_network.nodes[tech_task]['activity_name'])
                        
                        # cost breakdown
                        self.task_network.nodes[tech_task]['cost'] += cost
                
                # effort breakdown
                if data['state'] not in self.effort_breakdown[agent]:
                    self.effort_breakdown[agent][data['state']] = 0
                self.effort_breakdown[agent][data['state']] += step_size
                        
                # effort backlog
                effort_backlog = 0
                for task_info in data['task_queue'].values():
                    if not include_noise_in_results and task_info['task_type'] != 'Noise':    ########### team assignment is problematic now becasue it uses self.effort_backlog_teams
                        effort_backlog += task_info['remaining_effort']
                    elif include_noise_in_results:
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
            elif any([self.task_network.nodes[task]['repetitions'] >= 1
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
        
        util_over_time, average_util = self.calculate_utilization_over_time()
        effort_backlog = self.sort_effort_backlog()
        effort_breakdown, total_effort  = self.sort_effort_breakdown()
        dev_cost_breakdown = self.calc_cost_breakdown()

        # skip print statements and plots in case of a monte carlo
        if self.montecarlo:
            lead_time = self.global_clock
            return lead_time, self.cost_tracker[-1], average_util, total_effort
        
        
        lead_time = convert_hours_to_ymd(self.global_clock)
        
        print('\n_____________________________________________________________')
        
        print('\nResults:\n')
        self.print_result(f'Lead Time: {lead_time[0]} year(s), {lead_time[1]} month(s), {lead_time[2]} day(s)')
        self.print_result(f'Total Cost: ${round(self.cost_tracker[-1] / 1000, 1)}k')
        self.print_result(f'Total Cost (including idle): ${round(self.cost_tracker_with_idle[-1] / 1000, 1)}k')
        self.print_result(f'Total Effort: {round(self.effort_tracker[-1] / work_hours_per_day, 1)} person-days')

        # Resource Utilization
        if include_noise_in_results and simulate_noise:
            self.print_result('\nResource Utilizations (including noise):')
        else:
            self.print_result('\nResource Utilizations:')
        for entry, utilization in average_util.items():
            if split_plots != 'profession':
                self.print_result(f'     {entry}: {(utilization * 100):.1f}%')
            else:
                self.print_result(f'     {entry}s: {(utilization * 100):.1f}%')
            

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
            if use_moving_average is False:
                return data
            window_size = int(moving_average_plots / step_size)
            return np.convolve(data, np.ones(window_size) / window_size, mode='same')

        
        if show_plots:
            # Convert time to weeks
            time_in_weeks = np.array(self.time_points) / (7 * 24)

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
                #if not np.all(effort_data == 0):
                    if split_plots != 'profession':
                        ax2.plot(time_in_weeks, moving_average(effort_data), label=f'{entry}')
                    else:
                        ax2.plot(time_in_weeks, moving_average(effort_data), label=f'{entry}s')
            ax2.set_ylabel('Effort Backlog (h)')
            ax2.set_xlabel('Time (weeks)', labelpad=0)
            ax2.grid(True)
            ax2.set_xlim(left=0)
            ax2.set_ylim(bottom=0)
            moving_avrg_string = f'moving average: {round(moving_average_plots / 24, 1)} days'
            if include_noise_in_results and simulate_noise:
                moving_avrg_string += '; including noise'
            ax2.set_title(f'Effort Backlog over Time ({moving_avrg_string})')



            # Resource Utilization
            ax3 = fig.add_subplot(gs[2, 0])
            for entry, util_data in util_over_time.items():
                #if not np.all(util_data == 0):
                    if split_plots != 'profession':
                        ax3.plot(time_in_weeks, moving_average(util_data * 100), label=f'{entry}')
                    else:
                        ax3.plot(time_in_weeks, moving_average(util_data * 100), label=f'{entry}s')
            ax3.set_ylabel('Resource Utilization (%)')
            ax3.set_xlabel('Time (weeks)', labelpad=0)
            ax3.legend(loc='lower right', bbox_to_anchor=(-0.05, 1), fontsize=9)
            ax3.grid(True)
            ax3.set_xlim(left=0)
            ax3.set_ylim(bottom=0)
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
                    label=subcategory  # Legend entry for each subcategory
                )
                bottom += np.array(values[subcategory])
            
            ax4.set_title('Effort Breakdown')
            ax4.set_xticks(x)
            if split_plots == 'profession':
                labels = [category + 's' for category in categories]
            else:
                labels = categories
            ax4.set_xticklabels(labels)
            ax4.set_ylabel('Effort (person-days)')
            ax4.legend(ncol=3, prop={'size': 8})


            # Component Cost Breakdown
            ax5 = fig.add_subplot(gs[1, 1])

            # Filter components
            component_cost_breakdown = {}
            for element, costs in dev_cost_breakdown.items():
                if not self.architecture_graph.get_hierarchical_children(element):
                    del costs['System_Design']
                    del costs['Virtual_Integration']
                    del costs['LF_System_Simulation']
                    del costs['HF_System_Simulation']
                    
                    component_cost_breakdown[element] = costs
                    
                    # Add cost to parent element
                    parent_element = self.architecture_graph.get_parent(element)
                    if 'Subsys./Comp. Dev.' not in dev_cost_breakdown[parent_element]:
                        dev_cost_breakdown[parent_element]['Subsys./Comp. Dev.'] = 0
                    total_cost = sum(costs.values())
                    dev_cost_breakdown[parent_element]['Subsys./Comp. Dev.'] += total_cost

            elements = list(component_cost_breakdown.keys())
            activities = list(next(iter(component_cost_breakdown.values())).keys())
            x = np.arange(len(elements))
            width = 0.5  # Adjust width for stacked bars

            # Plot stacked bars
            bottom = np.zeros(len(elements))
            for activity in activities:
                activity_costs = [round(component_cost_breakdown[element][activity] / 1000, 1) for element in elements]
                bars = ax5.bar(x, activity_costs, width, label=activity, bottom=bottom)

                # Place labels at the center of each bar segment
                for bar, cost in zip(bars, activity_costs):
                    ax5.text(
                        bar.get_x() + bar.get_width() / 2, 
                        bar.get_y() + bar.get_height() / 2, 
                        f'{cost}', 
                        ha='center', 
                        va='center', 
                        fontsize=8  # Smaller font size
                    )
                bottom += np.array(activity_costs)

            ax5.set_title('Component Cost Breakdown')
            ax5.set_ylabel('Development Cost ($k)')
            ax5.set_xticks(x)
            ax5.set_xticklabels(elements)
            ax5.legend(prop={'size': 8})

            # System Cost Breakdown
            ax6 = fig.add_subplot(gs[2, 1])

            # Filter elements
            system_cost_breakdown = {}
            for element, costs in dev_cost_breakdown.items():
                if self.architecture_graph.get_hierarchical_children(element):
                    del costs['Component_Simulation']
                    del costs['Design']
                    
                    system_cost_breakdown[element] = costs

                    # Add cost to parent element
                    parent_element = self.architecture_graph.get_parent(element)
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
                    ax6.text(
                        bar.get_x() + bar.get_width() / 2, 
                        bar.get_y() + bar.get_height() / 2, 
                        f'{cost}', 
                        ha='center', 
                        va='center', 
                        fontsize=8  # Smaller font size
                    )
                bottom += np.array(activity_costs)

            ax6.set_title('System Cost Breakdown')
            ax6.set_ylabel('Development Cost ($k)')
            ax6.set_xticks(x)
            ax6.set_xticklabels(elements)
            ax6.legend(ncol=2, prop={'size': 8})


            self.print_sim_times()
            
            
            plt.tight_layout()
            plt.savefig(self.folder_name +  '/single_run_results_' + self.formatted_time + '.png')
            plt.show()
            
        else:
            self.print_sim_times()
    
    
    def print_sim_times(self):
        self.data_prep_time = time.time() - self.data_prep_start_time
        self.total_time = time.time() -self.init_start_time
        
        print('_____________________________________________________________\n')
        
        print(f'\nInitialization Time: {self.init_time:.2f} s')
        print(f'Simulation Time:     {self.sim_time:.2f} s')
        print(f'Data Prep Time:      {self.data_prep_time:.2f} s')
        print('____________________________')
        print(f'Total Time:          {self.total_time:.2f} s\n')

    
    def calc_cost_breakdown(self):
        system_dev_cost_breakdown = {}
        for task_info in self.task_network.nodes.values():
            cost = task_info['cost']

            # add architecture element
            architecture_element = task_info['architecture_element']
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
            activity_type =  task_info['activity_type']
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
            elif split_plots == 'overall':
                key = 'Overall'
            elif split_plots == 'teams':
                key = self.org_network.get_team(agent)
            if key not in backlog:
                backlog[key] = data
            else:
                backlog[key] = np.add(backlog[key], data)
        
        return backlog


    def calculate_utilization_over_time(self):
        utilization_over_time = {}        
        total_steps = len(self.time_points)
        
        # utilization over time
        for i in range(total_steps):
            idle_count = {}
            working_count = {}
            for agent, states in self.personnel_tracker.items():
                if split_plots == 'profession':
                    key = self.org_network.get_agent(agent)['profession']
                elif split_plots == 'overall':
                    key = 'Overall'
                elif split_plots == 'teams':
                    key = self.org_network.get_team(agent)
                if key not in idle_count:
                    idle_count[key] = 0
                    working_count[key] = 0
                
                if states[i] in {'Idle', 'Waiting'}:
                    idle_count[key] += 1
                elif not include_noise_in_results and states[i] not in {'Idle', 'Noise', 'Waiting'}:
                    working_count[key] += 1
                elif include_noise_in_results and states[i] not in {'Idle', 'Waiting'}:
                    working_count[key] += 1
                    
            for key in idle_count:
                if key not in utilization_over_time:
                    utilization_over_time[key] = np.zeros(total_steps)
                
                if idle_count[key] + working_count[key] != 0:
                    utilization_over_time[key][i] = working_count[key] / (idle_count[key] + working_count[key])
                else:
                    utilization_over_time[key][i] = 0 # if all agents of a profession are working on Noise
            
        # average utilization
        average_utilization = {}
        for entry in utilization_over_time:
            average_utilization[entry] = np.mean(utilization_over_time[entry])
    
        return utilization_over_time, average_utilization



if __name__ == "__main__":
    sim = PDsim(
        debug=False, 
        debug_interval=100, 
        debug_stop=9080, 
        log_events=True, 
        random_seed=42
    )
    
    sim.sim_run()
        