import json
from architecture_graph import ArchitectureGraph

class Tools:
    def __init__(self, architecture: ArchitectureGraph, folder=None):
        self.tool_list = {}
        self.knowledge_base = {}
        
        file_path = (folder + '/tools.json') if folder else 'Inputs/test_data/test_tools.json'
        
        with open(file_path, 'r') as file:
            tool_data = json.load(file)
        
        self.load_data(tool_data, architecture)

    def load_data(self, tool_data, architecture):
        for name, details in tool_data.items():
                
            if details['type'] == 'knowledgebase':
                for param_key, param_value in details["parameters"].items():
                    if param_key != 'pk_completeness':
                        self.knowledge_base[param_key] = param_value
                
                self.knowledge_base['product_knowledge_completeness']['reqs'] = {}
                self.knowledge_base['product_knowledge_completeness']['design'] = {}
                for node, data in sorted(list(architecture.nodes(data=True))):
                    self.knowledge_base['previous_product_knowledge'][node] = (1 - data['novelty']) * details['parameters']['pk_completeness']
                    self.knowledge_base['product_knowledge_completeness']['reqs'][node] = 0
                    self.knowledge_base['product_knowledge_completeness']['design'][node] = 0
                
            else:
                self.tool_list[name] = {
                    "type": details["type"],
                    "activities": details["activities"],
                    "architecture_elements": details["mapped_elements"],
                }
                for param_key, param_value in details["parameters"].items():
                    self.tool_list[name][param_key] = param_value


    def get_tools(self, architecture_element, activity):
        possible_tools = []
        for tool_name, tool_data in self.tool_list.items():
            if architecture_element in tool_data["architecture_elements"] and activity in tool_data["activities"]:
                possible_tools.append(tool_name)
        return possible_tools


if __name__ == "__main__":
    folder = 'Inputs/drone'
    
    architecture = ArchitectureGraph(folder=folder).architecture
    
    tools = Tools(architecture=architecture, folder=folder)
    print(tools.tool_list)
    print(tools.knowledge_base)
