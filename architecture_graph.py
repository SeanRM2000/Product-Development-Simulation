import networkx as nx
import math
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from Inputs.tuning_params import *
from Inputs.sim_inputs import *

class ArchitectureGraph:
    def __init__(self, test_data=False):
        self.architecture = nx.DiGraph()
        
        self._load_architecture(test_data)
        
        root_nodes = [node for node in self.architecture.nodes() if not self.get_parent(node)]
        if len(root_nodes) > 1:
            raise ValueError("Architecture has multiple root nodes.")
        self.root_node = root_nodes[0]


    def _add_element(self, name, knowledge_req, req_importance, wait_for_virtual_integration):
        self.architecture.add_node(name,
                                   wait_for_virtual_integration=wait_for_virtual_integration,
                                   knowledge_req=knowledge_req,
                                   overall_complexity=0,
                                   development_complexity=0,
                                   req_importance=req_importance,
                                   interfaces={},
                                   completion=0,
                                   definition_quality=[0],
                                   perceived_definition_quality=[0],
                                   design_quality=[0],
                                   perceived_design_quality=[0],
                                   overall_quality=[0],
                                   perceived_overall_quality=[0]
                                   )

    def _add_hierarchical_dependency(self, parent, child):
        self.architecture.add_edge(parent, child, dependency_type='hierarchical')

    def _add_interface_dependency(self, node1_name, node2_name, interface_severity):
        interface_complexity = self._calculate_interface_complexity(node1_name, node2_name, interface_severity)
        self.architecture.add_edge(node1_name, node2_name, 
                    dependency_type='interface',
                    complexity=interface_complexity,
                    definition_quality=[0],
                    perceived_definition_quality=[0],
                    design_quality=[0]
                    )

    def _calculate_interface_complexity(self, node1_name, node2_name, interface_severity):
        node1 = self.architecture.nodes[node1_name]
        node2 = self.architecture.nodes[node2_name]

        node1_kv = np.array(node1['knowledge_req'])
        node2_kv = np.array(node2['knowledge_req'])

        dot_product = np.dot(node1_kv, node2_kv)
        magnitude_node1 = np.sqrt(node1_kv.dot(node1_kv))
        magnitude_node2 = np.sqrt(node2_kv.dot(node2_kv))

        cos_theta = dot_product / (magnitude_node1 * magnitude_node2)
        sin_theta = math.sqrt(1 - cos_theta ** 2)

        return 0.5 * interface_severity * upper_limit_knowledge_scale ** sin_theta


    def _calculate_complexity(self):
        
        def calculate_technical_complexity(node):
            knowledge_req = self.architecture.nodes[node]['knowledge_req']
            return math.sqrt(sum(kri ** 2 for kri in knowledge_req) / len(knowledge_req))
        
        def calc_hierarchical_complexity(node, level=1):
            children = self.get_hierarchical_children(node)
            complexity = level * calculate_technical_complexity(node)
            
            for child in children:
                complexity += calc_hierarchical_complexity(child, level+1)
                
            return complexity
        
        def sum_interface_complexities(node):
            components = self.get_all_components(node)
            interface_complexities = 0
            for component in components:
                for dep_component in components:
                    if component != dep_component:
                        interface_data = self.architecture.get_edge_data(component, dep_component)
                        if interface_data:
                            interface_complexities += interface_data['complexity']
            
            return interface_complexities
        
        
        for node in self.architecture.nodes():
            hierarchical_complexity = calc_hierarchical_complexity(node)
            interface_complexities = sum_interface_complexities(node)
            self.architecture.nodes[node]['overall_complexity'] = hierarchical_complexity + interface_complexities
        
        for node in self.architecture.nodes():
            overall_complexity = self.architecture.nodes[node]['overall_complexity']
            irrelevant_complexity = 0
            for child in self.get_hierarchical_children(node):
                irrelevant_complexity += self.architecture.nodes[child]['overall_complexity']
                
            self.architecture.nodes[node]['development_complexity'] = overall_complexity - irrelevant_complexity

    
    
    def _aggregate_interfaces(self):
        
        def check_for_interface(node, dep_node):
            hierarchical_components_node = self.get_all_components(node)
            hierarchical_components_dep_node = self.get_all_components(dep_node)
            
            interfaces = []
            for component_node in hierarchical_components_node:
                for component_dep_node in hierarchical_components_dep_node:
                    if component_node != component_dep_node:
                        interface_data = self.architecture.get_edge_data(component_node, component_dep_node)
                        if interface_data:
                            interfaces.append((component_node, component_dep_node))
            
            return interfaces
        
        
        for node in self.architecture.nodes():
            interfaces = self.architecture.nodes[node]['interfaces']

            for dep_node in self.architecture.nodes():
                # no pairing of hierarchically dependent elements
                if dep_node == node or dep_node in self.get_all_ancestors(node) or node in self.get_all_ancestors(dep_node):
                    continue
                
                # no pairing of components with subsystems
                if ((self.get_hierarchical_children(node) and not self.get_hierarchical_children(dep_node)) or
                    (self.get_hierarchical_children(dep_node) and not self.get_hierarchical_children(node))):
                    continue
                
                dep_interfaces = check_for_interface(node, dep_node)
                
                if dep_interfaces:
                    for interface in dep_interfaces:
                        if interface in list(interfaces.values()):
                            original_node = next((k for k, v in interfaces.items() if v == interface), None)
                            
                            # only use the higher level implicit link
                            if original_node in self.get_all_ancestors(dep_node):
                                continue
                            else:
                                del interfaces[original_node]

                        interfaces[dep_node] = interface
                 
    
    def get_all_components(self, node):
        children = self.get_hierarchical_children(node)
        if not children:
            return [node]
        
        leaf_nodes = []
        for child in children:
            leaf_nodes.extend(self.get_all_components(child))
        
        return leaf_nodes
    
    
    def get_parent(self, node):
        parents = [predecessor for predecessor in self.architecture.predecessors(node) if self.architecture.edges[predecessor, node].get('dependency_type') == 'hierarchical']
        if len(parents) > 1:
            raise ValueError("Multiple parents found.")
        elif len(parents) == 0:
            return None 
        return parents[0]
    
    
    def get_hierarchical_children(self, node):
        return [child for child in self.architecture.successors(node) 
                if self.architecture.edges[node, child].get('dependency_type') == 'hierarchical']
    
    
    def get_all_ancestors(self, node):
        parents = []
        current_node = node
        
        while True:
            parent = self.get_parent(current_node)
            if parent is None:
                break
            parents.append(parent)
            current_node = parent

        return parents


    def show_architecture(self, show_plot=False):

        
        # Plot
        if show_plot:     
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Interface architecture
            nodes_to_include = [node for node in self.architecture.nodes() if not self.get_hierarchical_children(node)]
            
            # colors
            color_map = {}
            parents = set()
            for node in nodes_to_include:
                parent = self.get_parent(node)
                parents.add(parent)
                 
            num_colors = len(parents)
            cmap = plt.get_cmap('tab20')
            colors = [cmap(i / num_colors) for i in range(num_colors)]
            
            for parent, color in zip(parents, colors):
                color_map[parent] = color
                for successor in self.architecture.successors(parent):
                    if self.architecture.edges[parent, successor].get('dependency_type') == 'hierarchical':
                        color_map[successor] = color
                    
                    
            for node in self.architecture.nodes():
                if node not in color_map:
                    color_map[node] = "#808080"  # Gray
            
            interfaces_graph = self.architecture.subgraph(nodes_to_include)
            pos_interface = nx.kamada_kawai_layout(interfaces_graph)
            nx.draw(interfaces_graph, pos_interface, ax=axes[1], with_labels=True, node_size=500, 
                    node_color=[color_map[node] for node in interfaces_graph], 
                    font_size=10, font_color='black', arrows=True)
            axes[1].set_title("Interface architecture")
            patches = [mpatches.Patch(color=color, label=node) for node, color in color_map.items() if self.get_hierarchical_children(node) and color != '#808080']
            axes[1].legend(handles=patches, title="Subsystems", loc="best")
            
            # Hierarchical architecture
            hierarchy_edges = [(u, v) for u, v, data in self.architecture.edges(data=True) if data.get('dependency_type') == 'hierarchical']
            hierarchy_graph = self.architecture.edge_subgraph(hierarchy_edges).copy()

            for layer, nodes in enumerate(nx.topological_generations(hierarchy_graph)):
                for node in nodes:
                    hierarchy_graph.nodes[node]["layer"] = layer
            pos_hierarchy = nx.multipartite_layout(hierarchy_graph, subset_key="layer")
            nx.draw(hierarchy_graph, pos_hierarchy, ax=axes[0], with_labels=True, node_size=500, 
                    node_color=[color_map[node] for node in hierarchy_graph], 
                    font_size=10, font_color='black', arrows=True)
            axes[0].set_title("Hierarchical architecture")

            # Show the plots
            plt.tight_layout()
            plt.show()
    
        # Print
        print("\nProduct Graph with Interfaces:")
        root_node = [node for node in self.architecture.nodes() if not any(self.architecture.predecessors(node))][0]
        self.print_hierarchy(root_node)

    def print_hierarchy(self, node, level=0):
        indent = '    ' * level
        node_data = self.architecture.nodes[node]
        print(f"{indent}{node} (Overall Complexity: {node_data['overall_complexity']:.2f}, Development Complexity: {node_data['development_complexity']:.2f},  Req. Importance: {node_data['req_importance']})")

        # Print interfaces
        interfaces = []
        for _, neighbor, edge_data in self.architecture.edges(node, data=True):
            if edge_data['dependency_type'] == 'interface':
                interfaces.append((neighbor, edge_data['complexity']))

        if interfaces:
            print(f"{indent}    Interfaces:")
            for interface, complexity in interfaces:
                print(f"{indent}    -> {interface}: Complexity = {complexity:.2f}")

        # Recursively print children in the hierarchy
        for child in self.architecture.successors(node):
            if self.architecture.edges[node, child].get('dependency_type') == 'hierarchical':
                self.print_hierarchy(child, level + 1)
                


    def _load_architecture(self, test_data):
        def recursivly_build_graph(name, knowledge_req, req_importance, wait_for_virtual_integration, structure=None):
            knowledge_req = list(knowledge_req.values())
            self._add_element(name, knowledge_req, req_importance, wait_for_virtual_integration)

            if structure:
                for child_name, child_structure in structure.items():
                    recursivly_build_graph(
                                        child_name, 
                                        knowledge_req=child_structure.get('knowledge_vector'), 
                                        req_importance=child_structure.get('req_importance'),
                                        wait_for_virtual_integration=child_structure.get('wait_for_virtual_integration'),
                                        structure=child_structure.get('children'))
                    self._add_hierarchical_dependency(name, child_name)

        # Read data
        dsm_file_path = 'Inputs/test_data/test_dsm.csv' if test_data else 'Inputs/interface_dsm.csv'
        achritecture_file_path = 'Inputs/test_data/test_architecture.json' if test_data else 'Inputs/drone_architecture.json'
        
        with open(achritecture_file_path, 'r') as file:
            architecture_data = json.load(file)
        dsm = pd.read_csv(dsm_file_path, index_col=0)

        product_name = list(architecture_data.keys())[0]
        recursivly_build_graph( 
            name=product_name,
            knowledge_req=architecture_data[product_name].get("knowledge_vector"),
            req_importance=architecture_data[product_name].get("req_importance"),
            wait_for_virtual_integration=architecture_data[product_name].get("wait_for_virtual_integration"),
            structure=architecture_data[product_name].get("children")
        )

        # add interface dependencies
        for node_name1 in dsm.columns:
            for node_name2 in dsm.columns:
                n_interfaces = dsm.loc[node_name1, node_name2]
                if node_name1 != node_name2 and n_interfaces > 0:
                    self._add_interface_dependency(node_name1, node_name2, n_interfaces)

        self._aggregate_interfaces()
        
        self._calculate_complexity()
                



if __name__ == "__main__":
    architecture_graph = ArchitectureGraph(test_data=True)

    architecture_graph.show_architecture(show_plot=True)
    
    for node, data in architecture_graph.architecture.nodes(data=True):
        print(node, ': ', data['interfaces'])