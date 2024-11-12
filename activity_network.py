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
    def __init__(self, architecture_graph: ArchitectureGraph, tools: Tools, random_seed=None):
        self.architecture_graph = architecture_graph
        self.tools = tools
        self.activity_graph = nx.DiGraph()
        
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.generate_activity_graph()
        
        reduced_graph = nx.transitive_reduction(self.activity_graph)
        
        # Copy node attributes from the original graph
        for node, data in self.activity_graph.nodes(data=True):
            reduced_graph.nodes[node].update(data)

        # Copy edge attributes from the original graph
        for u, v, data in self.activity_graph.edges(data=True):
            if reduced_graph.has_edge(u, v):
                reduced_graph[u][v].update(data)

        self.activity_graph = reduced_graph
        
        self.final_activity = self.generate_activity_name(self.architecture_graph.root_node, 'Testing')
        
        

    def generate_activity_name(self, node, activity_type):
        return f"{node}_{activity_type}"

    def add_activity(self, node_name, activity_type, dependencies=None):
        activity_name = self.generate_activity_name(node_name, activity_type)
        
        complexity = self.architecture_graph.architecture.nodes[node_name]['development_complexity']
        distribution = tri_distribution[activity_type]
        effort = round(complexity * random.triangular(distribution[0], distribution[1], distribution[2]), 4)
        
        tool = self.check_for_tool(node_name, activity_type)
        if not tool:
            raise ValueError(f'Necessary tool for activity ({activity_type}) and element ({node_name}) combination does not exist.')
        
        self.activity_graph.add_node(activity_name, 
                                     activity_type=activity_type, 
                                     architecture_element=node_name,
                                     effort=effort,
                                     num_tasks=0,
                                     tasks=[],
                                     tool=tool,
                                     n_completed_tasks=0,
                                     activity_status='Waiting',
                                     assigned_to_team=''
                                     )
        
        if dependencies:
            if not isinstance(dependencies, (list, tuple)):
                dependencies = [dependencies] 
            for dependency in dependencies:
                if allow_activity_overlap:
                    overlap = activity_overlap.get(activity_type, (0, 0))[1]
                else:
                    overlap = 0
                self.activity_graph.add_edge(dependency, activity_name, 
                                             overlap_to_previous_activity=overlap)

        return activity_name


    def check_for_tool(self, node, activity):
        tool = self.tools.get_tools(node, activity)
        if len(tool) > 1:
            raise ValueError(f'Multiple tools assigned to activity and element combination: {activity}, {node}')
        else:
            return tool[0]


    def generate_activity_graph(self):
        
        def check_quantification_activity(predecessor_activity_type, node):
            quantification_activity_mapping = {
                'System_Design': 'LF_System_Simulation',
                'Design': 'Component_Simulation',
                'Virtual_Integration': 'HF_System_Simulation',
                'Prototyping': 'Testing'
            }
            quantification_activity_type = quantification_activity_mapping[predecessor_activity_type]
            tool = self.tools.get_tools(node, quantification_activity_type)
            if tool:
                return quantification_activity_type
            else:
                if node == root_node and quantification_activity_type == 'Testing':
                    raise ValueError('No overall Product/System testing activity exists')
                else:
                    return None
        
        def add_activity_according_to_overlap(node, activity_type, quant_activity, predecessor_activities):
            if allow_activity_overlap:
                if quant_activity and activity_overlap[activity_type][0] == 'quantification':
                    stack.append((node, activity_type, quant_activity))
                else:
                    stack.append((node, activity_type, predecessor_activities))
            else:
                stack.append((node, activity_type, quant_activity))
            

        # first activity
        stack = []
        root_node = self.architecture_graph.root_node
        stack.append((root_node, 'System_Design', None))
        
        
        while stack:
            node, activity_type, predecessor_activities = stack.pop(0)
            
            current_activity = self.add_activity(node, activity_type, predecessor_activities)
            
            # create corresponsing quantification activity if the tool for that activity exists
            quant_activity_type = check_quantification_activity(activity_type, node)
            if quant_activity_type:
                quant_activity = self.add_activity(node, quant_activity_type, current_activity)
            else:
                quant_activity = None
            
            
            match activity_type:
                case 'System_Design':          
                    children = list(self.architecture_graph.get_hierarchical_children(node))
                    for child in children:
                        if list(self.architecture_graph.get_hierarchical_children(child)):
                            add_activity_according_to_overlap(child, 'System_Design', quant_activity, current_activity)
                        else:
                            add_activity_according_to_overlap(child, 'Design', quant_activity, current_activity)
                
                
                case 'Design':
                    add_activity_according_to_overlap(node, 'Prototyping', quant_activity, current_activity)
                
                    parent = self.architecture_graph.get_parent(node)
                    if self.check_for_tool(parent, 'Virtual_Integration'):
                        add_activity_according_to_overlap(parent, 'Virtual_Integration', quant_activity, current_activity)
                        

                case 'Virtual_Integration':
                    parent = self.architecture_graph.get_parent(node)
                    if parent and self.check_for_tool(parent, 'Virtual_Integration'):
                        add_activity_according_to_overlap(parent, 'Virtual_Integration', quant_activity, current_activity)
                        
                         
                case 'Prototyping':
                    parent = self.architecture_graph.get_parent(node)
                    if parent:
                        add_activity_according_to_overlap(parent, 'Prototyping', quant_activity, current_activity)                       

            

        def check_children_recursivly(node):
            children = self.architecture_graph.get_hierarchical_children(node)
            predecessors =  []
            for child in children:
                if self.activity_graph.nodes.get(self.generate_activity_name(child, 'Virtual_Integration')):
                    predecessors.append(self.generate_activity_name(child, 'Virtual_Integration'))
                else:
                    predecessors.extend(check_children_recursivly(child))
                    
            return predecessors
            
        
        def check_integration_of_higher_levels(parent):
            activity = self.generate_activity_name(parent, 'Virtual_Integration')
            if self.activity_graph.nodes.get(activity):
                return activity
            else:
                new_parent = self.architecture_graph.get_parent(parent)
                if new_parent:
                    return check_integration_of_higher_levels(new_parent)
                else:
                    return None
            
            
        for node, data in self.activity_graph.nodes(data=True):
            if data['activity_type'] == 'Prototyping':
                element = data['architecture_element']
                
                if allow_activity_overlap:
                    overlap_with_virtual_integration = self.architecture_graph.architecture.nodes[element]['wait_for_virtual_integration']
                else:
                    overlap_with_virtual_integration = 'full system'
                
                match overlap_with_virtual_integration:
                    
                    case 'full system':
                        virtual_full_system_integration = self.generate_activity_name(root_node, 'Virtual_Integration')
                        if self.activity_graph.nodes.get(virtual_full_system_integration):
                            predecessors = [virtual_full_system_integration]
                        else:
                            predecessors = check_children_recursivly(root_node)
                              
                    case 'yes':
                        element = self.activity_graph.nodes[node]['architecture_element']
                        parent = self.architecture_graph.get_parent(element)       
                        predecessors = [check_integration_of_higher_levels(parent)]
                        if not predecessors:
                            predecessors = check_children_recursivly(node)
                            
                    case 'no':
                        predecessors = [] # no virtual integration activity is a predecessor
                    
                    case _:
                        raise ValueError('Variable \'wait_for_virtual_integration\' in architecture has to be "full system", "yes", or "no".')
                    
                    
                for predecessor in predecessors:
                    # check if quantification activity has to be used
                    if activity_overlap['Prototyping'][0] == 'quantification' or not allow_activity_overlap:
                        quant_activity = [s for s in self.activity_graph.successors(predecessor) 
                                          if self.activity_graph.nodes[s].get('activity_type') in {'Component_Simulation', 'HF_System_Simulation'}]
                        if len(quant_activity) > 1:
                            raise ValueError(f'Multiple quantification activities found for \'{predecessor}\'')
                        if quant_activity:
                            predecessor = quant_activity[0]
                    
                    if allow_activity_overlap:
                        overlap = activity_overlap.get('Prototyping')[1]
                    else:
                        overlap = 0 
                    
                    self.activity_graph.add_edge(predecessor, node, overlap_to_previous_activity=overlap)
            
            
            

    def show_activity_graph(self):
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

        node_colors = [color_map.get(self.activity_graph.nodes[node]['activity_type'], 'grey') for node in self.activity_graph.nodes()]

        for layer, nodes in enumerate(nx.topological_generations(self.activity_graph)):
            for node in nodes:
                self.activity_graph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(self.activity_graph, subset_key="layer")

        edge_labels = {
            (u, v): f"{d['overlap_to_previous_activity']:.2f}"
            for u, v, d in self.activity_graph.edges(data=True)
        }

        plt.figure(figsize=(12, 8))
        nx.draw(
            self.activity_graph, pos, with_labels=True, 
            node_size=500, node_color=node_colors, 
            font_size=10, font_color='black', arrows=True
        )

        nx.draw_networkx_edge_labels(
            self.activity_graph, pos, edge_labels=edge_labels, 
            font_size=8, font_color='black'
        )

        plt.title("High Level Activity Graph with Overlap Values")
        plt.show()


if __name__ == "__main__":
    architecture_graph = ArchitectureGraph(test_data=True)
    tools = Tools(test_data=True)

    activity_graph = ActivityNetwork(architecture_graph, tools, random_seed=42)
    activity_graph.show_activity_graph()