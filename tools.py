import json

class Tools:
    def __init__(self, folder=None):
        self.tool_list = {}
        self.knowledge_base = {}
        
        file_path = (folder + '/tools.json') if folder else 'Inputs/test_data/test_tools.json'
        
        with open(file_path, 'r') as file:
            tool_data = json.load(file)
        
        self.load_data(tool_data)

    def load_data(self, tool_data):
        for name, details in tool_data.items():
            if details['type'] == 'knowledgebase':
                self.knowledge_base = details['parameters']
                
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
    tools = Tools()
    print(tools.tool_list)
