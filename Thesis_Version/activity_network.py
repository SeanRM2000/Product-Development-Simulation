import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import json
import matplotlib as mpl

from architecture_graph import ArchitectureGraph
from organization_graph import OrganizationalGraph
from tools import Tools

from Inputs.tuning_params import *
from Inputs.sim_settings import *

class ActivityNetwork:
    def __init__(self, architecture_graph: ArchitectureGraph, tools: Tools, org: OrganizationalGraph=None, folder=None, random_seed=None):
        self.architecture_graph = architecture_graph
        self.tools = tools
        self.org = org
        self.activity_graph = nx.DiGraph()
        
        file_path = ('Architecture/Inputs/Baseline/activities.json') if folder else 'Inputs/test_data/test_activities.json'
        
        with open(file_path, 'r') as file:
            activity_data = json.load(file)
        
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        
        self._generate_activity_graph(activity_data)
        
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

    def _add_activity(self, node_name, activity_type, activity_data, dependencies=None):
        activity_name = self.generate_activity_name(node_name, activity_type)
        
        complexity = self.architecture_graph.architecture.nodes[node_name]['technical_complexity']
        distribution = activity_data['tri_distribution'] # h per complexity
        learning_rate = activity_data['learning_rate'] # reduce effort after repeating tasks (Wright model) --> doubling number of repetetions leads to 1-x reduction 
        
        # time reduction
        spread_red=0.02
        
        min = distribution[0] * effort_factor
        max = distribution[1] * effort_factor
        mode = distribution[2] * effort_factor
        
        spread_min = (mode - min) * spread_red
        spread_max = (max - mode) * spread_red
        min = mode - spread_min
        max = mode + spread_max
        
        effort = round(complexity * random.triangular(min, max, mode), 4)
        
        if ((self.architecture_graph.get_hierarchical_children(node_name) and activity_type == 'Testing') or
            (not self.architecture_graph.get_hierarchical_children(node_name) and activity_type == 'Prototyping')):
            effort = round(effort * testing_increase_factor_systems, 4)
        
        if activity_type in {'Testing', 'Prototyping'}:
            effort = round(effort * physical_effort_factor, 4)
        
        # decreased effort for design and prototyping activies that are supplier activities
        if self.architecture_graph.architecture.nodes[node_name].get('procure', False) and activity_type == 'Design':
            effort = round(effort * supplier_effort_factor, 4)
            
        tool = self.check_for_tool(node_name, activity_type)
        
        if not tool:
            raise ValueError(f'Necessary tool for activity ({activity_type}) and element ({node_name}) combination does not exist.')
        
        
        self.activity_graph.add_node(activity_name, 
                                     activity_type=activity_type, 
                                     architecture_element=node_name,
                                     effort=effort,
                                     learning_rate=learning_rate,
                                     num_tasks=0,
                                     tasks=[],
                                     tool=tool,
                                     
                                     n_completed_tasks=0,
                                     activity_status='Waiting',
                                     second_order_rework_reduction=1,
                                     
                                     cost=0,
                                     total_work_effort=0,
                                     total_rework_effort=0
                                     )
        
        
        if fixed_assignments:
            for agent in self.org.all_agents:
                for activity, elements in self.org.get_agent(agent)['responsibilities'].items():
                    if activity == activity_type:
                        for element in elements:
                            if element == node_name:
                                self.activity_graph.nodes[activity_name]['assigned_to'] = agent
        else:
            self.activity_graph.nodes[activity_name]['assigned_to_team'] = ''
        
        
        if dependencies:
            if not isinstance(dependencies, (list, tuple)):
                dependencies = [dependencies] 
            for dependency in dependencies:
                if allow_activity_overlap:
                    overlap_data = activity_data.get('activity_overlap')
                    if overlap_data:
                        overlap = overlap_data['value']
                    else:
                        overlap = 0
                else:
                    overlap = 0
                self.activity_graph.add_edge(dependency, activity_name, 
                                             overlap_to_previous_activity=overlap)

        return activity_name


    def check_for_tool(self, node, activity):
        tool = self.tools.get_tools(node, activity)
        if len(tool) > 1:
            raise ValueError(f'Multiple tools assigned to activity and element combination: {activity}, {node}')
        elif len(tool) == 0:
            return None
        else:
            return tool[0]


    def _generate_activity_graph(self, activity_data):
        
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
                    raise ValueError('No overall Product/System testing activity could be created becasue the tool is missing.')
                else:
                    return None
        
        def add_activity_according_to_overlap(node, activity_type, quant_activity, predecessor_activities):
            if allow_activity_overlap:
                if quant_activity and activity_data[activity_type]['activity_overlap']['to']== 'quantification':
                    stack.append((node, activity_type, quant_activity))
                else:
                    stack.append((node, activity_type, predecessor_activities))
            else:
                if quant_activity:
                    stack.append((node, activity_type, quant_activity))
                else:
                    stack.append((node, activity_type, predecessor_activities))
            

        # first activity
        stack = []
        root_node = self.architecture_graph.root_node
        stack.append((root_node, 'System_Design', None))
        
        
        while stack:
            node, activity_type, predecessor_activities = stack.pop(0)
            
            current_activity = self._add_activity(node, activity_type, activity_data[activity_type], predecessor_activities)
            
            # create corresponsing quantification activity if the tool for that activity exists
            quant_activity_type = check_quantification_activity(activity_type, node)
            if quant_activity_type:
                quant_activity = self._add_activity(node, quant_activity_type, activity_data[quant_activity_type], current_activity)
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
            higher_level_element = []
            activity = self.generate_activity_name(parent, 'Virtual_Integration')
            
            if self.activity_graph.nodes.get(activity):
                higher_level_element.append(activity)
            else:
                new_parent = self.architecture_graph.get_parent(parent)
                if new_parent:
                    higher_level_element.extend(check_integration_of_higher_levels(new_parent))
                    
            return higher_level_element
            
            
        for node, data in self.activity_graph.nodes(data=True):
            if data['activity_type'] == 'Prototyping':
                element = data['architecture_element']
                
                if True:#allow_activity_overlap:
                    prototype_start_condition = self.architecture_graph.architecture.nodes[element]['prototype_start_condition']
                else:
                    prototype_start_condition = 'Full Virtual Integration'
                
                match prototype_start_condition:
                    
                    case 'Full Virtual Integration':
                        virtual_full_system_integration = self.generate_activity_name(root_node, 'Virtual_Integration')
                        if self.activity_graph.nodes.get(virtual_full_system_integration):
                            predecessors = [virtual_full_system_integration]
                        else:
                            predecessors = check_children_recursivly(root_node)
                              
                    case 'Higher Level Virtual Integration':
                        element = self.activity_graph.nodes[node]['architecture_element']
                        parent = self.architecture_graph.get_parent(element)       
                        predecessors = check_integration_of_higher_levels(parent)
                        if not predecessors:
                            predecessors = check_children_recursivly(element)
                            
                    case 'Lower Level Virtual Integration':
                        #predecessors = check_children_recursivly(element)
                        predecessor = []
                            
                    case 'Component Development':
                        predecessors = [] # no virtual integration activity is a predecessor
                    
                    case _:
                        raise ValueError('Variable for \'prototype_start_condition\' in architecture has to be "Full Virtual Integration", "Higher Level Virtual Integration", "Lower Level Virtual Integration", or "Component Development".')
                    
                for predecessor in predecessors:
                    # check if quantification activity has to be used
                    if activity_data['Prototyping']['activity_overlap']['to'] == 'quantification' or not allow_activity_overlap:
                        quant_activity = [s for s in self.activity_graph.successors(predecessor) 
                                          if self.activity_graph.nodes[s].get('activity_type') in {'Component_Simulation', 'HF_System_Simulation'}]
                        if len(quant_activity) > 1:
                            raise ValueError(f'Multiple quantification activities found for \'{predecessor}\'')
                        if quant_activity:
                            predecessor = quant_activity[0]
                    
                    if allow_activity_overlap:
                        overlap = activity_data['Prototyping']['activity_overlap']['value']
                    else:
                        overlap = 0 
                    
                    self.activity_graph.add_edge(predecessor, node, overlap_to_previous_activity=overlap)
            
            
            

    def show_activity_graph(self):
        abbreviations = {
            'Drone': '1',
            'Air Frame': '1.1',
            'Propulsion System': '1.2',
            'Flight Control System': '1.3',
            'Main Body': '1.1.1',
            'Landing Gear': '1.1.3',
            'Arms': '1.1.2',
            'Battery': '1.2.2',
            'Motor': '1.2.3',
            'Propeller': '1.2.1',
            'Control Software': '1.3.3',
            'Sensor Suite': '1.3.1',
            'Controller': '1.3.2'
        }
        
        color_map = {
            'System_Design': 'navy',
            'LF_System_Simulation': 'teal',
            'Design': 'cornflowerblue',
            'Component_Simulation': 'darkgreen',
            'Virtual_Integration': 'limegreen',
            'HF_System_Simulation': 'purple',
            'Prototyping': 'orange',
            'Testing': 'orangered'
        }
        color_map = {
            'System_Design': '#e60049',
            'LF_System_Simulation': '#0bb4ff',
            'Design': '#50e991',
            'Component_Simulation': '#9b19f5',
            'Virtual_Integration': '#ffa300',
            'HF_System_Simulation': '#dc0ab4',
            'Prototyping': '#e6d800',
            'Testing': '#00bfa0'
        }

        plt.rcParams["font.family"] = "Times New Roman"
        mpl.rcParams['svg.fonttype'] = 'none'
        
        node_colors = [color_map.get(self.activity_graph.nodes[node]['activity_type'], 'grey') for node in self.activity_graph.nodes()]
        #node_colors = ['#D9D9D9' for _ in self.activity_graph.nodes()]

        for layer, nodes in enumerate(nx.topological_generations(self.activity_graph)):
            for node in nodes:
                self.activity_graph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(self.activity_graph, subset_key="layer")
        
        with open("Misc/Baseline/layout.json", "r") as f:
            new_pos = json.load(f)
        #    pos = new_pos
            
        min_x, min_y = np.min(list(pos.values()), axis=0)
        max_x, max_y = np.max(list(pos.values()), axis=0)

        for node in pos:
            pos[node] = [(pos[node][0] - min_x) / (max_x - min_x), 
                        (pos[node][1] - min_y) / (max_y - min_y)]  # Normalized positions
            
        print(pos)

        edge_labels = {
            (u, v): f"{d['overlap_to_previous_activity']:.2f}"
            for u, v, d in self.activity_graph.edges(data=True)
            if d['overlap_to_previous_activity'] > 0
        }
        
        node_labels = {
            node: abbreviations.get(self.activity_graph.nodes[node].get('architecture_element', ''), '')
            for node in self.activity_graph.nodes()
        }

        plt.figure(figsize=(6, 4))
        nx.draw(
            self.activity_graph, pos,
            node_size=350, node_color=node_colors,
            font_size=12, font_color='black', arrows=True
        )
        # Add node labels with more visible styling
        for node, (x, y) in pos.items():
            plt.text(
                x, y, node_labels[node],
                fontsize=9, ha='center', va='center', color='black', fontweight='bold',
                #bbox=dict(facecolor='white', boxstyle='round,pad=0.1', edgecolor='white')
            )

            nx.draw_networkx_edge_labels(
                self.activity_graph, pos, edge_labels=edge_labels, 
                font_size=8, font_color='black'
            )
            
        legend_handles = [
            plt.Line2D(
                [0], [0], marker='o', color='w', label=activity_type.replace('_', ' '),
                markersize=10, markerfacecolor=color
            )
            for activity_type, color in color_map.items()
        ]
        plt.legend(
            handles=legend_handles, loc='upper left', fontsize=8.5,frameon=False, handletextpad=0.3
        )    
        plt.margins(0)
        plt.savefig('Generated Activities.svg', format='svg', bbox_inches='tight', pad_inches=0)
        plt.title("High Level Activity Graph with Overlap Values")
        plt.show()


if __name__ == "__main__":
    folder = 'Architecture/Inputs/DOE3 - New Tool/DOE3-1'
    folder = 'Architecture/Inputs/Baseline'
    
    architecture_graph = ArchitectureGraph(folder=folder)
    tools = Tools(architecture=architecture_graph.architecture, folder=folder)
    org_graph = OrganizationalGraph(architecture_graph, folder=folder)

    activity_graph = ActivityNetwork(architecture_graph, tools, org=org_graph, folder=folder, random_seed=None)
    activity_graph.show_activity_graph()
    
    
    for node, data in activity_graph.activity_graph.nodes(data=True):
        print(f'{node}: {data['effort']}')
    #    print(node, ': ', activity_graph.activity_graph.nodes[node].get('assigned_to', None))