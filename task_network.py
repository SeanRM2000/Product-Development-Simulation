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
    def __init__(self, activity_network: ActivityNetwork, architecture_graph: ArchitectureGraph, randomize_structure=False, randomize_task_times=False, random_seed=None):
        self.activity_network = activity_network.activity_graph
        self.architecture_graph = architecture_graph.architecture
        self.final_activity = activity_network.final_activity
        
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.random_structure = randomize_structure
        self.random_times = randomize_task_times
        
        if self.random_structure:
            self.task_overlap_probabilities = overlap_prob
        else:
            self.task_parallelization = task_parallelization
    
        # generate graph
        self.task_times = self.generate_task_times()
        
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
        self.calc_rank_pos_weights()
        self.calc_importance_of_tasks()
        

    
    def add_task(self, task_number, activity_name, effort, dependencies=None, first_task=False, final_task=False):
        task_name = self.generate_task_name(task_number, activity_name)
        
        architecture_element = self.activity_network.nodes[activity_name]['architecture_element']
        activity_type = self.activity_network.nodes[activity_name]['activity_type']
        knowledge_req = self.architecture_graph.nodes[architecture_element]['knowledge_req']
        
        learning_rate = self.activity_network.nodes[activity_name]['learning_rate']
        
        self.task_graph.add_node(
            task_name,
            task_number=task_number,
            first_task=first_task,
            final_task=final_task,
            activity_name=activity_name,
            architecture_element=architecture_element,
            activity_type=activity_type,
            knowledge_req=knowledge_req,
            nominal_effort=effort,
            learning_rate=learning_rate,
            importance=0,
            task_status='Waiting',
            assigned_to=None,
            completed=False,
            repetitions=0,
            quality=None,
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
    
    def calc_rank_pos_weights(self):
        final_task = self.activity_network.nodes[self.final_activity]['tasks'][-1]
        total_effort = sum(self.task_graph.nodes[task]['nominal_effort'] for task in self.task_graph.nodes)
        
        def calculate_dependent_effort(task, nodes_to_exclude=[]):
            dependent_tasks = nx.descendants(self.task_graph, task) | {task}
            return sum(self.task_graph.nodes[dependent]['nominal_effort'] for dependent in dependent_tasks if dependent not in nodes_to_exclude)
        
        def find_nearest_connected_task(task):
            ancestors = list(nx.ancestors(self.task_graph, task))
            selected_ancestor = (None, float('inf')) # distance to final task
            for ancestor in ancestors:
                if nx.has_path(self.task_graph, ancestor, final_task):
                    path_length = nx.shortest_path_length(self.task_graph, ancestor, final_task)
                    if selected_ancestor[1] > path_length:
                        selected_ancestor = (ancestor, path_length)
                    elif selected_ancestor[1] == path_length:
                        selected_ancestor = random.choice([selected_ancestor, (ancestor, path_length)])
            return selected_ancestor[0]

        for task in self.task_graph.nodes:
            if nx.has_path(self.task_graph, task, final_task):
                total_dependent_effort = calculate_dependent_effort(task)
            else:
                connected_ancestor = find_nearest_connected_task(task)

                nodes_to_exclude = set()
                
                num_tasks = self.activity_network.nodes[self.task_graph.nodes[task]['activity_name']]['num_tasks']
                for path in nx.all_simple_paths(self.task_graph, source=connected_ancestor, target=task, cutoff=num_tasks):
                    nodes_to_exclude.update(path)
                    
                nodes_to_exclude.discard(connected_ancestor)
                nodes_to_exclude.discard(task)
                
                total_dependent_effort = calculate_dependent_effort(connected_ancestor, nodes_to_exclude)

            
            rank_pos_weight = total_dependent_effort / total_effort
            self.task_graph.nodes[task]['rank_pos_weight'] = round(rank_pos_weight, 4)
            
    def calc_importance_of_tasks(self):
        for task in self.task_graph.nodes:
            architecture_element = self.task_graph.nodes[task]['architecture_element']
            complexity = self.architecture_graph.nodes[architecture_element]['development_complexity'] / upper_limit_knowledge_scale
            req_importance = self.architecture_graph.nodes[architecture_element]['req_importance']
            rank_pos_weight = self.task_graph.nodes[task]['rank_pos_weight']
            
            importance = round(prioritization_weights['rank_pos_weight'] * rank_pos_weight + prioritization_weights['complexity'] * complexity + prioritization_weights['req_importance'] * req_importance, 4) 
            self.task_graph.nodes[task]['importance'] = importance


    def recursive_task_network_generation(self, activity_name, incoming_dependencies=None, previous_final_task=None, num_overlapped_tasks=0):
        # check if tasks for activity were already created
        existing_tasks = [task for task in self.task_graph.nodes if self.task_graph.nodes[task]['activity_name'] == activity_name]
        if existing_tasks:
            # only add dependencies when tasks are already created
            for dependency in incoming_dependencies:
                self.task_graph.add_edge(dependency, self.generate_task_name(0, activity_name))
            self.collect_overlap_connections(activity_name, num_overlapped_tasks, previous_final_task)
            
        else:
            task_times = self.task_times[activity_name]
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
            successor_activities = sorted(list(self.activity_network.successors(activity_name)))
            random.shuffle(successor_activities)
            for successor in successor_activities:
                activity_overlap = self.activity_network.edges[activity_name, successor]['overlap_to_previous_activity']

                outgoing_dependencies = []
                if activity_overlap == 0:
                    task_name = self.generate_task_name(num_tasks - 1, activity_name) # last node
                    outgoing_dependencies.append(task_name)
                    num_overlapped_tasks = 0
                else:
                    num_not_overlapped_tasks = math.floor(num_tasks * (1 - activity_overlap)) # of predecessor
                    num_overlapped_tasks = math.ceil(len(self.task_times[successor]) * (1 - activity_overlap)) # of successor
                    
                    subgraph_nodes = [node for node, data in self.task_graph.nodes(data=True) if data['activity_name'] == activity_name]
                    subgraph = self.task_graph.subgraph(subgraph_nodes).copy()
                    sorted_layers = list(nx.topological_generations(subgraph))
                    for layer in sorted_layers: 
                        layer.sort()
                        random.shuffle(layer)
                        for task in layer:
                            if len(outgoing_dependencies) == num_not_overlapped_tasks:
                                break
                            if task not in outgoing_dependencies and self.task_graph.nodes[task]['activity_name'] == activity_name:
                                outgoing_dependencies.append(task)

                self.recursive_task_network_generation(successor, outgoing_dependencies, final_task, num_overlapped_tasks)


    def collect_overlap_connections(self, activity_name, num_overlapped_tasks, previous_final_task):
        
        subgraph_nodes = [node for node, data in self.task_graph.nodes(data=True) if data['activity_name'] == activity_name]
        subgraph = self.task_graph.subgraph(subgraph_nodes).copy()
        sorted_layers = list(nx.topological_generations(subgraph))
        
        overlapped_tasks = []
        for layer in sorted_layers:
            layer.sort()
            random.shuffle(layer)
            if len(overlapped_tasks) == num_overlapped_tasks:
                break
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
        start_task_name = self.add_task(0, activity_name, task_times[0], incoming_dependencies, first_task=True) # start task
        
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
        self.add_task(0, activity_name, task_times[0], incoming_dependencies, first_task=True) # Start task
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
        
    
    
    def generate_task_times(self):
        task_times = {}
        for activity, activity_data in self.activity_network.nodes.items():
            total_effort = activity_data['effort']
            if self.random_times:
                task_times[activity] = []
                remaining_effort = total_effort
                while remaining_effort > min_task_effort:
                    task_effort = round(random.triangular(min_task_effort, min(max_task_effort, remaining_effort), (min_task_effort + max_task_effort) / 2), 4)
                    task_times[activity].append(task_effort)
                    remaining_effort -= task_effort
                
                if remaining_effort > 0:
                    task_times[activity].append(remaining_effort)
                    
            else:
                n_tasks = int(total_effort / nominal_task_effort)
                task_effort = round(total_effort / n_tasks, 4)
                task_times[activity] = [task_effort] * n_tasks
                
        return task_times

    def generate_task_name(self, task_number, activity_name):
        return f"{task_number}_{activity_name}"
    
    
    def collect_activity_tasks(self, activity_name):
        # Collects tasks in the order of a topological generation for quick access during simulation
        
        subgraph_nodes = [node for node, data in self.task_graph.nodes(data=True) if data['activity_name'] == activity_name]
        subgraph = self.task_graph.subgraph(subgraph_nodes).copy()
        sorted_layers = list(nx.topological_generations(subgraph))

        collected_tasks = []

        for layer in sorted_layers:
            layer.sort()
            random.shuffle(layer)
            collected_tasks.extend(layer)
        self.activity_network.nodes[activity_name]['tasks'] = collected_tasks

    
    def plot_task_graph(self, labels_to_show=[]):
        color_map = {
            'System_Design': 'blue',
            'LF_System_Simulation': 'lightgreen',
            'Design': 'lightblue',
            'Component_Simulation': 'lightgreen',
            'Virtual_Integration': 'green',
            'HF_System_Simulation': 'lightgreen',
            'Prototyping': 'orange',
            'Testing': 'orangered'
        }
        
        node_colors = [color_map[self.task_graph.nodes[node]['activity_type']] for node in self.task_graph.nodes()]

        for layer, nodes in enumerate(nx.topological_generations(self.task_graph)):
            for node in nodes:
                self.task_graph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(self.task_graph, subset_key="layer")

        # Define labels
        labels = {}
        for node in self.task_graph.nodes:
            task_number = self.task_graph.nodes[node]['task_number']
            final_task = self.task_graph.nodes[node].get('final_task', False)
            activity_name = self.task_graph.nodes[node]['activity_name']
            if labels_to_show:
                if node in labels_to_show:
                    labels[node] = node
            else:
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
    folder = 'Architecture/Inputs/Baseline'
    architecture_graph = ArchitectureGraph(folder=folder)
    tools = Tools(folder=folder)
    
    activity_network = ActivityNetwork(architecture_graph, tools, folder=folder, random_seed=42)

    
    task_network = TaskNetwork(activity_network, architecture_graph, randomize_structure=True, randomize_task_times=False, random_seed=42)
    
    task_network.plot_task_graph()