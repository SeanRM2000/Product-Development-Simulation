import networkx as nx
import math
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Inputs.tuning_params import *


class ArchitectureGraph:
    def __init__(self, folder=None):
        self.architecture = nx.DiGraph()
        
        self._load_architecture(folder)
        
        root_nodes = [node for node in self.architecture.nodes() if not self._get_parent(node)]
        if len(root_nodes) > 1:
            raise ValueError("Architecture has multiple root nodes.")
        self.root_node = root_nodes[0]
        
        self._get_all_structural_relationships()


    def _add_element(self, name, knowledge_req, req_importance, novelty, prototype_start_condition, procure, is_rootnode=False):
        self.architecture.add_node(
            name,
            prototype_start_condition=prototype_start_condition,
            procure=procure,
            responsible_designer='',
                                                
            knowledge_req=knowledge_req,
            overall_complexity=0,
            development_complexity=0,
            technical_complexity=0,
            req_importance=req_importance,
            novelty=novelty,
            
            interfaces={},
            
            completion=False,
            
            previous_definition_quality=0,
            definition_quality=0,
            
            previous_design_quality=0,
            design_quality=0,
            
            previous_overall_quality=0,
            overall_quality=0,
            
            
            perceived_quality={
                'LF_System_Simulation': 0,
                'Component_Simulation': 0,
                'HF_System_Simulation': 0,
                'Testing': 0
                },
            previous_perceived_quality={
                'LF_System_Simulation': 0,
                'Component_Simulation': 0,
                'HF_System_Simulation': 0,
                'Testing': 0
                },
            perceived_definition_quality=None
            )
        
        # root node (overall system) has def_quality
        if is_rootnode:
            self.architecture.nodes[name]['definition_quality'] = 1

    def _add_hierarchical_dependency(self, parent, child):
        self.architecture.add_edge(parent, child, dependency_type='hierarchical')

    def _add_interface_dependency(self, node1_name, node2_name, interface_severity):
        interface_complexity = self._calculate_interface_complexity(node1_name, node2_name, interface_severity)
        self.architecture.add_edge(node1_name, node2_name, 
                    dependency_type='interface',
                    severity=interface_severity,
                    complexity=round(interface_complexity, 4),
                    
                    system_definition_quality=0, # not used currently
                    definition_quality=0,
                    product_knowledge_used=0,
                    info_used=[],
                    old_info=None,
                    
                    perceived_interface_quality=None
                    )

        
    def calc_interface_quality(self, edge):
        edge_data = self.architecture.edges[edge]
        
        if not edge_data['info_used'] and edge_data['old_info'] == None:
            raise ValueError(f'Interface Quality for {edge} was called without information for it existing.')
        
        if not edge_data['info_used']:
            return edge_data['product_knowledge_used'] * (edge_data['definition_quality'] + (1 - edge_data['definition_quality'] ) * edge_data['old_info'])
        else:
            sum_info_used = 0
            n = len(edge_data['info_used'])
            for comp, cons in edge_data['info_used']:
                sum_info_used += comp * cons
            
            return edge_data['product_knowledge_used'] * (edge_data['definition_quality'] + (1 - edge_data['definition_quality'] ) * sum_info_used / n) 
        
        


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

        return 0.5 * ( interface_severity / upper_interface_severity_limit ) * upper_limit_knowledge_scale ** sin_theta


    def _calculate_complexity(self):
        
        def calculate_technical_complexity(node):
            knowledge_req = self.architecture.nodes[node]['knowledge_req']
            return math.sqrt(sum(kri ** 2 for kri in knowledge_req) / len(knowledge_req))
        
        def calc_hierarchical_complexity(node, level=1):
            children = self._get_hierarchical_children(node)
            complexity = level * calculate_technical_complexity(node)
            
            for child in children:
                complexity += calc_hierarchical_complexity(child, level+1)
                
            return complexity
        
        def sum_interface_complexities(node):
            components = self._get_all_components(node)
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
            self.architecture.nodes[node]['overall_complexity'] = round(hierarchical_complexity + interface_complexities, 4)
        
        for node in self.architecture.nodes():
            technical_complexity = calculate_technical_complexity(node)
            
            children = self._get_hierarchical_children(node)
            
            child_complexities = 0
            for child in children:
                child_complexities += calculate_technical_complexity(child)
                interfaces = self.architecture.nodes[child]['interfaces']
                for dep_node, edges in interfaces.items():
                    if dep_node in children:    ################################################# maybe include all interfaces
                        for edge in edges:
                            child_complexities += self.architecture.edges[edge]['complexity']
                
            self.architecture.nodes[node]['development_complexity'] = round(technical_complexity + child_complexities, 4) 
            self.architecture.nodes[node]['technical_complexity'] = round(technical_complexity, 4) 
    
    
    def _aggregate_interfaces(self):
        
        def check_for_interface(node, dep_node):
            hierarchical_components_node = self._get_all_components(node)
            hierarchical_components_dep_node = self._get_all_components(dep_node)
            
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
                if dep_node == node or dep_node in self._get_all_ancestors(node) or node in self._get_all_ancestors(dep_node):
                    continue
                
                # no pairing of components with subsystems
                if ((self._get_hierarchical_children(node) and not self._get_hierarchical_children(dep_node)) or
                    (self._get_hierarchical_children(dep_node) and not self._get_hierarchical_children(node))):
                    continue
                
                dep_interfaces = check_for_interface(node, dep_node)
                
                if dep_interfaces:
                    for interface in dep_interfaces:
                        if interface in list(interfaces.values()):
                            original_node = next((k for k, v in interfaces.items() if v == interface), None)
                            
                            # only use the higher level implicit link
                            if original_node in self._get_all_ancestors(dep_node):
                                continue
                            else:
                                del interfaces[original_node]

                        if not interfaces.get(dep_node):
                            interfaces[dep_node] = []
                        interfaces[dep_node].append(interface)
                 
    
    def _get_all_structural_relationships(self):
        for node, data in self.architecture.nodes(data=True):
            data['all_components'] = self._get_all_components(node)
            data['all_decendants'] = self._get_all_hierarchical_descendants(node)
            data['parent'] = self._get_parent(node)
            data['children'] = self._get_hierarchical_children(node)
            data['ancestors'] = self._get_all_ancestors(node)
            
            if not data['children']:
                total_interface_complexity = 0
                for _, interfaces_dep in self.architecture.nodes[node]['interfaces'].items():
                    for edge in interfaces_dep:
                        total_interface_complexity += self.architecture.edges[edge]['complexity']
                data['total_interface_complexity'] = total_interface_complexity
    
    def _get_all_components(self, node):
        children = self._get_hierarchical_children(node)
        if not children:
            return [node]
        
        leaf_nodes = []
        for child in children:
            leaf_nodes.extend(self._get_all_components(child))
        
        return leaf_nodes
        
    def get_all_components(self, node):
        return self.architecture.nodes[node]['all_components']
    
    def _get_all_hierarchical_descendants(self, node):
        descendants = []
        children = self._get_hierarchical_children(node)
        
        for child in children:
            descendants.append(child)
            descendants.extend(self._get_all_hierarchical_descendants(child))
    
        return descendants
    
    def get_all_hierarchical_descendants(self, node):
        return self.architecture.nodes[node]['all_decendants']
    
    def _get_parent(self, node):
        parents = [predecessor for predecessor in self.architecture.predecessors(node) if self.architecture.edges[predecessor, node].get('dependency_type') == 'hierarchical']
        if len(parents) > 1:
            raise ValueError("Multiple parents found.")
        elif len(parents) == 0:
            return None 
        return parents[0]
    
    def get_parent(self, node):
        return self.architecture.nodes[node]['parent']
    
    
    def _get_hierarchical_children(self, node):
        return [child for child in sorted(self.architecture.successors(node))
                if self.architecture.edges[node, child].get('dependency_type') == 'hierarchical']
    
    def get_hierarchical_children(self, node):
        return self.architecture.nodes[node]['children']
    
    
    def _get_all_ancestors(self, node):
        parents = []
        current_node = node
        
        while True:
            parent = self._get_parent(current_node)
            if parent is None:
                break
            parents.append(parent)
            current_node = parent

        return parents
    
    def get_all_ancestors(self, node):
        return self.architecture.nodes[node]['ancestors']
    
    
    def hierarchical_distance(self, from_node, to_node):
        if to_node == 'root':
            to_node = self.root_node
            
        # Traverse upward from the starting node to the root
        current_node = from_node
        distance = 0
        
        while current_node:
            if current_node == to_node:
                return distance
            current_node = self._get_parent(current_node)
            distance += 1

        # If the target node is not found, return None
        return None



    def show_architecture(self, show_plot=False):

        
        # Plot
        if show_plot:     
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Interface architecture
            nodes_to_include = [node for node in self.architecture.nodes() if not self._get_hierarchical_children(node)]
            
            # colors
            color_map = {}
            parents = set()
            for node in nodes_to_include:
                parent = self._get_parent(node)
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
            
            pos_interface = nx.spring_layout(interfaces_graph, weight='severity') # alternativ kadam kawai with different thicknesses
            nx.draw(interfaces_graph, pos_interface, ax=axes[1], with_labels=True, node_size=500, 
                    node_color=[color_map[node] for node in interfaces_graph], 
                    font_size=10, font_color='black', arrows=True)
            
            edge_complexities = nx.get_edge_attributes(interfaces_graph, 'severity')
            nx.draw_networkx_edge_labels(interfaces_graph, pos_interface, ax=axes[1], 
                                         edge_labels=edge_complexities, font_size=8)
        
            axes[1].set_title("Interface architecture")
            patches = [mpatches.Patch(color=color, label=node) for node, color in color_map.items() if self._get_hierarchical_children(node) and color != '#808080']
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
                


    def _load_architecture(self, folder):
        def recursivly_build_graph(name, knowledge_req, req_importance, novelty, prototype_start_condition, procure, structure=None, is_rootnode=False):
            knowledge_req = list(knowledge_req.values())
            self._add_element(name, knowledge_req, req_importance, novelty, prototype_start_condition, procure, is_rootnode=is_rootnode)

            if structure:
                for child_name, child_structure in structure.items():
                    recursivly_build_graph(
                                        child_name, 
                                        knowledge_req=child_structure.get('knowledge_vector'), 
                                        req_importance=child_structure.get('req_importance'),
                                        novelty=child_structure.get('novelty'),
                                        prototype_start_condition=child_structure.get('prototype_start_condition'),
                                        procure=child_structure.get('procure', False),
                                        structure=child_structure.get('children'))
                    self._add_hierarchical_dependency(name, child_name)

        # Read data
        if folder and ('Architecture/Inputs/' in folder or 'Hypercube' in folder): # use product folder if data is from architecture
            base_folder='Architecture/Inputs/Product'

        dsm_file_path = (base_folder + '/interface_dsm.csv') if folder else 'Inputs/test_data/test_dsm.csv'
        
        tool_file = folder + '/tools.json'
        with open(tool_file, 'r') as file:
            tool_data = json.load(file)
        
        if folder and 'HFSystemSimulator' in tool_data:
            achritecture_file_path = (base_folder + '/architecture_added_tool.json')
        else:
            achritecture_file_path = (base_folder + '/architecture.json') if folder else 'Inputs/test_data/test_architecture.json'
        
        with open(achritecture_file_path, 'r') as file:
            architecture_data = json.load(file)
        dsm = pd.read_csv(dsm_file_path, index_col=0)

        product_name = list(architecture_data.keys())[0]
        recursivly_build_graph( 
            name=product_name,
            knowledge_req=architecture_data[product_name].get("knowledge_vector"),
            req_importance=architecture_data[product_name].get("req_importance"),
            novelty=architecture_data[product_name].get("novelty"),
            prototype_start_condition=architecture_data[product_name].get("prototype_start_condition"),
            procure=architecture_data[product_name].get("procure", False),
            structure=architecture_data[product_name].get("children"),
            is_rootnode=True
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
    folder = 'Architecture/Inputs/Baseline'
    architecture_graph = ArchitectureGraph(folder=folder)

    architecture_graph.show_architecture(show_plot=True)
    
    print(architecture_graph.get_all_hierarchical_descendants('Arms'))
    #for node, data in architecture_graph.architecture.nodes(data=True):
     #   print(node, ': ', data['interfaces'])
       # print(f'{node}: {data['overall_complexity']}')
      #  print(f'{node}: {data['development_complexity']}')
        #print(f'{node}: {data['technical_complexity']}')