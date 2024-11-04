import networkx as nx
import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Classes
from activity_network import ActivityNetwork
from architecture_graph import ArchitectureGraph
from tools import Tools

# Parameters
from Inputs.sim_settings import *
from Inputs.tuning_params import *
from Inputs.sim_inputs import *

class TaskNetwork():
    def __init__(self, activity_network, architecture_graph, randomize_structure=False, randomize_task_times=False, random_seed=None):
        self.activity_network = activity_network
        self.architecture_graph = architecture_graph
        
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.random_structure = randomize_structure
        self.random_times = randomize_task_times
        
        # overlapping parameters
        self.activity_overlap = activity_overlap
        if self.random_structure:
            self.task_overlap_probabilities = overlap_prob
        else:
            self.task_parallelization = task_parallelization
    
        # generate graph
        self.task_graph = nx.DiGraph()
        self.final_overlap_edges = []
        first_activity = [node for node in list(self.activity_network.nodes) if not any(self.activity_network.predecessors(node))][0]
        self.recursive_task_network_generation(first_activity)

        # remove unnecessary edges
        task_graph_red = nx.transitive_reduction(self.task_graph)
        task_graph_red.add_nodes_from(self.task_graph.nodes(data=True))
        self.task_graph = task_graph_red
        
        # edges of final tasks during overlapping
        self.task_graph.add_edges_from(self.final_overlap_edges)
                
        # calculate rank positional weight and importance
        self.calc_rank_pos_weight()
        self.calc_importance()
            
        #self.task_graph.remove_edges_from(self.final_overlap_edges)     ############## maybe add data to these nodes to skip them in checks

    
    def add_task(self, task_number, activity_name, effort, dependencies=None, final_task=False):
        task_name = self.generate_task_name(task_number, activity_name)
        
        architecture_element = self.activity_network.nodes[activity_name]['architecture_element']
        activity_type = self.activity_network.nodes[activity_name]['activity_type']
        knowledge_req = self.architecture_graph.nodes[architecture_element]['knowledge_req']
        
        self.task_graph.add_node(task_name,
                                 task_number=task_number,
                                 final_task=final_task,
                                 activity_name=activity_name,
                                 architecture_element=architecture_element,
                                 activity_type=activity_type,
                                 knowledge_req=knowledge_req,
                                 nominal_effort=effort,
                                 learning_factor=learning_factors[activity_type],
                                 importance=0,
                                 task_status='Waiting',
                                 assigned_to=None,
                                 completed=False,
                                 repetitions=0,
                                 quality=0,
                                 cost=0
                                 )
        
        # add incoming dependencies
        if dependencies:
            if isinstance(dependencies, (list, tuple)):
                for dependency in dependencies:
                    self.task_graph.add_edge(dependency, task_name)
            else:
                self.task_graph.add_edge(dependencies, task_name)
        
        
        return task_name
    
    def calc_rank_pos_weight(self):
        total_effort = sum(self.task_graph.nodes[task]['nominal_effort'] for task in self.task_graph.nodes)

        def calculate_dependent_effort(task):
            dependent_tasks = nx.descendants(self.task_graph, task) | {task}
            return sum(self.task_graph.nodes[dependent]['nominal_effort'] for dependent in dependent_tasks)

        for task in self.task_graph.nodes:
            total_dependent_effort = calculate_dependent_effort(task)
            rank_pos_weight = total_dependent_effort / total_effort
            self.task_graph.nodes[task]['rank_pos_weight'] = rank_pos_weight
            
    def calc_importance(self):
        for task in self.task_graph.nodes:
            architecture_element = self.task_graph.nodes[task]['architecture_element']
            complexity = self.architecture_graph.nodes[architecture_element]['technical_complexity'] / upper_limit_knowledge_scale
            req_importance = self.architecture_graph.nodes[architecture_element]['req_importance']
            rank_pos_weight = self.task_graph.nodes[task]['rank_pos_weight']
            
            self.task_graph.nodes[task]['importance'] = (prioritization_weights['rank_pos_weight'] * rank_pos_weight +
                                                         prioritization_weights['complexity'] * complexity +
                                                         prioritization_weights['req_importance'] * req_importance) 


    def recursive_task_network_generation(self, activity_name, incoming_dependencies=None, previous_final_task=None, num_overlapped_tasks=0):
        # check if tasks for activity were already created
        existing_tasks = [task for task in self.task_graph.nodes if self.task_graph.nodes[task]['activity_name'] == activity_name]
        if existing_tasks:
            # only add dependencies when tasks are already created
            for dependency in incoming_dependencies:
                self.task_graph.add_edge(dependency, self.generate_task_name(0, activity_name))
            self.collect_overlap_connections(activity_name, num_overlapped_tasks, previous_final_task)
        else:
            activity_effort = self.activity_network.nodes[activity_name]['effort']
            task_times = self.generate_task_times(activity_effort)
            num_tasks = len(task_times)
            self.activity_network.nodes[activity_name]['num_tasks'] = num_tasks
            
            if self.random_structure:
                final_task = self.generate_random_task_network_for_activity(activity_name, task_times, incoming_dependencies)
            else:
                final_task = self.generate_non_random_task_network_for_activity(activity_name, task_times, incoming_dependencies)
            self.collect_activity_tasks(activity_name)

            if previous_final_task:
                self.collect_overlap_connections(activity_name, num_overlapped_tasks, previous_final_task)
            
            # Generate Activity overlapping
            successor_activities = list(self.activity_network.successors(activity_name))
            if successor_activities:
                activity_overlap = self.activity_network.edges[activity_name, successor_activities[0]]['overlap_to_previous_activity'] # overlapping activity types are always same

                outgoing_dependencies = []
                if activity_overlap == 0:
                    task_name = self.generate_task_name(num_tasks - 1, activity_name) # last node
                    outgoing_dependencies.append(task_name)
                    num_overlapped_tasks = 0
                else:
                    num_not_overlapped_tasks = math.ceil(num_tasks * (1 - activity_overlap))
                    num_overlapped_tasks = num_tasks - num_not_overlapped_tasks
                    for layer in list(nx.bfs_layers(self.task_graph, self.generate_task_name(0, activity_name))): # start bfs from first activity node
                        layer.sort()
                        random.shuffle(layer)
                        for task in layer:
                            if len(outgoing_dependencies) == num_not_overlapped_tasks:
                                break
                            if task not in outgoing_dependencies:
                                outgoing_dependencies.append(task)

                # recursive network generation
                for successor_activity_name in successor_activities:
                    self.recursive_task_network_generation(successor_activity_name, outgoing_dependencies, final_task, num_overlapped_tasks)


    def collect_overlap_connections(self, activity_name, num_overlapped_tasks, previous_final_task):
        bfs_layers = list(nx.bfs_layers(self.task_graph, self.generate_task_name(0, activity_name)))
        
        overlapped_tasks = []
        for layer in bfs_layers:
            layer.sort()
            random.shuffle(layer)
            for task in layer:
                if len(overlapped_tasks) == num_overlapped_tasks:
                    break
                if self.task_graph.nodes[task]['activity_name'] == activity_name:
                    overlapped_tasks.append(task)
    
        for task in overlapped_tasks:
            relevant_successors = [succ for succ in self.task_graph.successors(task) if self.task_graph.nodes[succ]['activity_name'] == activity_name]
            
            if not all(succ in overlapped_tasks for succ in relevant_successors):
                self.final_overlap_edges.append((previous_final_task, task))

        
        
    def generate_non_random_task_network_for_activity(self, activity_name, task_times, incoming_dependencies=None):
        n_tasks = len(task_times) - 2 
        start_task_name = self.add_task(0, activity_name, task_times[0], incoming_dependencies) # start task
        
        if fully_linear_tasks:
            parallelization = 0
        else:
            parallelization = self.task_parallelization[self.activity_network.nodes[activity_name]['activity_type']]
        longest_path = math.ceil((1 - parallelization) * n_tasks)

        # create paths not longer than the longest path
        remaining_tasks = n_tasks
        task_number = 1

        while remaining_tasks > 0:
            prev_task_name = start_task_name
            for _ in range(longest_path):
                task_name = self.add_task(task_number, activity_name, task_times[task_number])
                self.task_graph.add_edge(prev_task_name, task_name)
                prev_task_name = task_name
                
                task_number += 1
                remaining_tasks -= 1
                if remaining_tasks == 0:
                    break

        # Connect any tasks with no successors with the the last node
        last_task = n_tasks + 1
        last_task_name = self.add_task(last_task, activity_name, task_times[last_task], final_task=True)
        for task in range(last_task):
            task_name = self.generate_task_name(task, activity_name)
            if not list(self.task_graph.successors(task_name)):
                self.task_graph.add_edge(task_name, last_task_name)
                
        return last_task_name
            
        

        
    def generate_random_task_network_for_activity(self, activity_name, task_times, incoming_dependencies=None):
        n_tasks = len(task_times)
        self.add_task(0, activity_name, task_times[0], incoming_dependencies) # Start task
        last_task = n_tasks - 1
        
        for current_task in range(n_tasks - 1):
            current_task_name = self.generate_task_name(current_task, activity_name)
            
            # check already placed tasks for activity
            filtered_tasks = [self.task_graph.nodes[task]['task_number'] for task in self.task_graph.nodes if self.task_graph.nodes[task]['activity_name'] == activity_name]
            last_placed_task = sorted(filtered_tasks)[-1]
            if last_placed_task > current_task:
                for next_task in range(current_task + 1, last_placed_task + 1):
                    next_task_name = self.generate_task_name(next_task, activity_name)
                    # handle max in- or outgoing edges
                    if max_out is not None and self.task_graph.out_degree(current_task_name) >= max_out:
                        break
                    if max_in is not None and self.task_graph.in_degree(next_task_name) >= max_in:
                        continue
                    if random.random() < reconnect_probability:
                        self.task_graph.add_edge(current_task_name, next_task_name)
            
            if max_out is not None and self.task_graph.out_degree(current_task_name) >= max_out:
                continue
                
            # Add new task and connect
            next_task = last_placed_task + 1
            overlap_probability = self.task_overlap_probabilities[self.activity_network.nodes[activity_name]['activity_type']]
            if next_task != last_task:
                if not list(self.task_graph.successors(current_task_name)) or random.random() < overlap_probability:
                    next_task_name = self.add_task(next_task, activity_name, task_times[next_task])
                    self.task_graph.add_edge(current_task_name, next_task_name)
                else:
                    continue
                
                # check additional dependencies
                for next_task in range(last_placed_task + 2, last_task):
                    if max_out is not None and self.task_graph.out_degree(current_task_name) >= max_out:
                        break
                    if random.random() < overlap_probability:
                        next_task_name = self.add_task(next_task, activity_name, task_times[next_task])
                        self.task_graph.add_edge(current_task_name, next_task_name)
                    else:
                        break
        
        # Connect any tasks with no successors with the the last node
        last_task_name = self.add_task(last_task, activity_name, task_times[last_task], final_task=True)
        for task in range(last_task):
            task_name = self.generate_task_name(task, activity_name)
            if not list(self.task_graph.successors(task_name)):
                self.task_graph.add_edge(task_name, last_task_name)
        
        return last_task_name
        
    
    
    def generate_task_times(self, total_effort):
        if not self.random_times:
            n_tasks = int(total_effort / nominal_task_effort)
            task_effort = total_effort / n_tasks
            task_times = [task_effort] * n_tasks
        else:
            task_times = []
            remaining_effort = total_effort
            while remaining_effort > min_task_effort:
                task_effort = random.triangular(min_task_effort, min(max_task_effort, remaining_effort), (min_task_effort + max_task_effort) / 2)
                task_times.append(task_effort)
                remaining_effort -= task_effort
            
            if remaining_effort > 0:
                task_times.append(remaining_effort)
                
        return task_times

    def generate_task_name(self, task_number, activity_name):
        return f"{task_number}_{activity_name}"
    
    
    def collect_activity_tasks(self, activity_name):
        start_task = self.generate_task_name(0, activity_name)
        bfs_layers = list(nx.bfs_layers(self.task_graph, start_task))
        
        collected_tasks = []

        for layer in bfs_layers:
            layer.sort()
            random.shuffle(layer)
            activity_layer_tasks = [task for task in layer if self.task_graph.nodes[task]['activity_name'] == activity_name]
            collected_tasks.extend(activity_layer_tasks)
        self.activity_network.nodes[activity_name]['tasks'] = collected_tasks
    
    
    def plot_task_graph(self):
        color_map = {
            'Definition': 'lightblue',
            'Design': 'lightgreen',
            'Testing': 'orange',
            'Integration': 'lightcoral'
        }
        
        node_colors = [color_map[self.task_graph.nodes[node]['activity_type']] for node in self.task_graph.nodes()]

        for layer, nodes in enumerate(nx.topological_generations(self.task_graph)):
            for node in nodes:
                self.task_graph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(self.task_graph, subset_key="layer")

        # Define labels
        labels = {}
        for node in self.task_graph.nodes():
            task_number = self.task_graph.nodes[node]['task_number']
            final_task = self.task_graph.nodes[node].get('final_task', False)
            activity_name = self.task_graph.nodes[node]['activity_name']
            if task_number == 0:
                labels[node] = f"Start {activity_name}"
            elif final_task:
                labels[node] = f"Complete {activity_name}"

        plt.figure(figsize=(12, 8))
        nx.draw(self.task_graph, pos, with_labels=False, node_size=500, node_color=node_colors, font_size=10, font_color='black', arrows=True)
        nx.draw_networkx_labels(self.task_graph, pos, labels, font_size=10, font_color="black")
        plt.title("Task Graph")
        plt.show()

        
if __name__ == "__main__":
    architecture_graph = ArchitectureGraph(test_data=True).architecture
    tools = Tools(test_data=True)
    
    activity_network = ActivityNetwork(architecture_graph, tools, random_seed=42)
    
    # seperate start condition for integration (goodness of test) --> Decision making

    
    task_network = TaskNetwork(activity_network.activity_graph, architecture_graph, randomize_structure=True, randomize_task_times=False, random_seed=42)
    
    task_network.plot_task_graph()