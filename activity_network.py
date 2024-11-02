import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

from architecture_graph import ArchitectureGraph
from tools import Tools

from Inputs.tuning_params import *
from Inputs.sim_inputs import *
from Inputs.sim_settings import *

class ActivityNetwork:
    def __init__(self, architecture_graph, tools, random_seed=None):
        self.architecture_graph = architecture_graph
        self.tools = tools
        self.activity_graph = nx.DiGraph()
        
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
            

        self.generate_activity_graph()

    def add_activity(self, node_name, activity_type, dependencies=None):
        activity_name = f"{node_name}_{activity_type}"
        
        complexity = self.architecture_graph.nodes[node_name]['technical_complexity']
        
        # calculate effort here instead of in architecture
        distribution = tri_distribution[activity_type]
        effort =  complexity * random.triangular(distribution[0], distribution[1], distribution[2])

        possible_tools = self.tools.get_tools(node_name, activity_type)
        
        self.activity_graph.add_node(activity_name, 
                                     activity_type=activity_type, 
                                     architecture_element=node_name,
                                     effort=effort,
                                     num_tasks=0,
                                     tasks=[],
                                     possible_tools=possible_tools,
                                     degree_of_completion=0,
                                     activity_status='Waiting',
                                     information_completeness=0
                                     )
        
        if dependencies:
            if not isinstance(dependencies, (list, tuple)):
                dependencies = [dependencies] 
            
            for dependency in dependencies:
                predeceessor_type = self.activity_graph.nodes[dependency]['activity_type']
                overlap = self.get_activity_overlap(predeceessor_type, activity_type)
                self.activity_graph.add_edge(dependency, activity_name, 
                                             overlap_to_previous_activity=overlap)

        return activity_name
    
    def get_activity_overlap(self, predecessor_activity_type, successor_activity_type):
        global activity_overlap
        
        if predecessor_activity_type == 'Definition' and successor_activity_type == 'Definition':
            return activity_overlap['def_def']
        elif predecessor_activity_type == 'Definition' and successor_activity_type == 'Design':
            return activity_overlap['def_des']
        elif predecessor_activity_type == 'Design' and successor_activity_type == 'Testing':
            return activity_overlap['des_test']
        elif predecessor_activity_type == 'Testing' and successor_activity_type == 'Integration':
            return activity_overlap['test_int']
        elif predecessor_activity_type == 'Integration' and successor_activity_type == 'Testing':
            return activity_overlap['int_test']
        else:
            raise ValueError(f'Activity paring not valid. (Predecessor: {predecessor_activity_type}; Successor: {successor_activity_type})')
    
    def get_hierarchical_successors(self, node):
        return [child for child in self.architecture_graph.successors(node) 
                if self.architecture_graph.edges[node, child].get('dependency_type') == 'hierarchical']

    def get_hierarchical_predecessors(self, node):
        return [parent for parent in self.architecture_graph.predecessors(node) 
                if self.architecture_graph.edges[parent, node].get('dependency_type') == 'hierarchical']

    def generate_activity_graph(self):
        stack = []

        # Start from root node
        root_nodes = [node for node in self.architecture_graph.nodes() if not any(self.get_hierarchical_predecessors(node))]
        if len(root_nodes) > 1:
            raise ValueError("Architecture has multiple root nodes.")
        root_node = root_nodes[0]
        stack.append((root_node, 'Definition', None))

        while stack:
            node, activity_type, parent_activities = stack.pop(0)

            current_activity = self.add_activity(node, activity_type, parent_activities)

            if activity_type == 'Definition':
                children = list(self.get_hierarchical_successors(node))

                if not children: # Component
                    stack.append((node, 'Design', current_activity))
                else: # Decompose further
                    for child in children:
                        stack.append((child, 'Definition', current_activity))

            elif activity_type == 'Design' or activity_type == 'Integration':
                stack.append((node, 'Testing', current_activity))

            elif activity_type == 'Testing':
                if self.get_hierarchical_predecessors(node):
                    parent = self.get_hierarchical_predecessors(node)[0]
                    siblings = self.get_hierarchical_successors(parent)
                    
                    all_testing_activities = [f"{sibling}_Testing" for sibling in siblings]
                    if all(test_activity in self.activity_graph for test_activity in all_testing_activities):
                        stack.append((parent, 'Integration', all_testing_activities))

        return self.activity_graph

    def show_activity_graph(self):
        color_map = {
            'Definition': 'lightblue',
            'Design': 'lightgreen',
            'Testing': 'orange',
            'Integration': 'lightcoral'
        }

        node_colors = [color_map[self.activity_graph.nodes[node]['activity_type']] for node in self.activity_graph.nodes()]

        # Assign layers to nodes for multipartite layout
        for layer, nodes in enumerate(nx.topological_generations(self.activity_graph)):
            for node in nodes:
                self.activity_graph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(self.activity_graph, subset_key="layer")

        # Extract edge labels for overlaps (only the number)
        edge_labels = {
            (u, v): f"{d['overlap_to_previous_activity']:.2f}"
            for u, v, d in self.activity_graph.edges(data=True)
        }

        plt.figure(figsize=(12, 8))

        # Draw the graph with nodes and edges
        nx.draw(
            self.activity_graph, pos, with_labels=True, 
            node_size=500, node_color=node_colors, 
            font_size=10, font_color='black', arrows=True
        )

        # Draw the edge labels with only the numerical overlap
        nx.draw_networkx_edge_labels(
            self.activity_graph, pos, edge_labels=edge_labels, 
            font_size=8, font_color='black'
        )

        plt.title("High Level Activity Graph with Overlap Values")
        plt.show()



if __name__ == "__main__":
    architecture_graph = ArchitectureGraph(test_data=True)
    tools = Tools(test_data=True)

    activity_graph = ActivityNetwork(architecture_graph.architecture, tools, random_seed=42)

    activity_graph.show_activity_graph()
    
    print([data['effort'] for activity, data in activity_graph.activity_graph.nodes(data=True)])
        