import os
import json
import copy
import pandas as pd

def update_digital_literacy(data, new_value):
    """
    Recursively update any dictionary that has a key "digital_literacy".
    Within that dictionary, if there is a key "EngineeringTools", update its value.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "digital_literacy" and isinstance(value, dict):
                if "EngineeringSupportTools" in value: # EngineeringTools, EngineeringSupportTools
                    value["EngineeringSupportTools"] = new_value
            else:
                update_digital_literacy(value, new_value)
    elif isinstance(data, list):
        for item in data:
            update_digital_literacy(item, new_value)
    return data

def update_agents_digital_literacy(data, new_value, agent_names):
    """
    Recursively update the digital literacy for only the agents whose name is in agent_names.
    Specifically, update the 'EngineeringTools' field under 'digital_literacy' if the dictionary
    has a 'name' key and that name is in the provided agent_names list.
    """
    if isinstance(data, dict):
        if "name" in data and "digital_literacy" in data and isinstance(data["digital_literacy"], dict):
            if data["name"] in agent_names:
                if "EngineeringTools" in data["digital_literacy"]:
                    data["digital_literacy"]["EngineeringTools"] = new_value
        for key, value in data.items():
            update_agents_digital_literacy(value, new_value, agent_names)
    elif isinstance(data, list):
        for item in data:
            update_agents_digital_literacy(item, new_value, agent_names)
    return data

def update_tools(data, accuracy_factor, tool_complexity=None, interop=0.5):
    """
    For each tool (except the one with key "KnowledgeBase"), update:
    - The 'use_complexity' parameter (if exists) is replaced by tool_complexity.
    - For digital tools with a non-null accuracy value, multiply accuracy by accuracy_factor.
    """
    for tool_name, tool in data.items():
        # Skip KnowledgeBase
        if tool_name == "KnowledgeBase":
            continue
        #if tool_name == 'HFSystemSimulator':
        # Update use_complexity if it exists in parameters

        if True and "parameters" in tool and tool["parameters"]["interoperability"]:
            tool["parameters"]["interoperability"] = accuracy_factor
            
        # Update accuracy for digital tools if accuracy is not None
        if False and "parameters" in tool and "accuracy" in tool["parameters"] and tool["parameters"]["accuracy"] and tool["type"] == "digital":
            if True:
                if True:
                    if accuracy_factor >= 0:
                        tool["parameters"]["accuracy"] += (1 - tool["parameters"]["accuracy"]) * accuracy_factor
                    else:
                        tool["parameters"]["accuracy"] += tool["parameters"]["accuracy"] * accuracy_factor
                else:
                    if tool["parameters"]["accuracy"] >= 0.5:
                        tool["parameters"]["accuracy"] += (1 - tool["parameters"]["accuracy"]) * accuracy_factor
                    else:
                        tool["parameters"]["accuracy"] += tool["parameters"]["accuracy"] * accuracy_factor
            else:
                if tool_name == 'HFSystemSimulator':
                    tool["parameters"]["accuracy"] = accuracy_factor
                    tool["parameters"]["use_complexity"] = tool_complexity
                    tool["parameters"]["interoperability"] = interop
    return data

def main():


    # Load the baseline JSON files
    with open('Architecture/Inputs/Baseline/organization.json', "r") as f:
        org_data_orig = json.load(f)
    with open('Architecture/Inputs/Baseline/tools.json', "r") as f:
        tools_data_orig = json.load(f)

    # Load the configuration Excel file
    # Make sure the Excel file 'configs.xlsx' is in the same directory as this script.
    configs_df = pd.read_excel("Architecture/configs.xlsx")
    
    print(configs_df)
    
    # Process each configuration row from the Excel file
    for index, row in configs_df.iterrows():
        folder_name = 'Architecture/Inputs/DOE2 - Interoperability/' + str(row["Config. No."])
        new_digital_lit = row["Digital Literacy"]
        new_accuracy_factor = row["Accuracy Change"]
        #new_interoperability = row["Interoperability"]

        # Create the new folder if it doesn't exist
        os.makedirs(folder_name, exist_ok=True)

        # Deep copy the original JSON data to avoid modifying the baseline data
        org_data = copy.deepcopy(org_data_orig)
        tools_data = copy.deepcopy(tools_data_orig)

        # Update the organization.json data
        org_data_updated = update_digital_literacy(org_data, new_digital_lit)
        #org_data_updated = update_agents_digital_literacy(org_data, new_digital_lit, ['System Manager', 'Systems Engineer 1', 'Systems Engineer 2', 'System Simulation Engineer 2'])

        # Update the tools.json data
        tools_data_updated = update_tools(tools_data, new_accuracy_factor)#, tool_complexity=new_digital_lit)#, interop=new_interoperability)

        # Write the updated JSON files into the new folder
        with open(os.path.join(folder_name, "organization.json"), "w") as f:
            json.dump(org_data_updated, f, indent=4)
        with open(os.path.join(folder_name, "tools.json"), "w") as f:
            json.dump(tools_data_updated, f, indent=4)

        print(f"Configuration '{folder_name}' created with updated JSON files.")



def doe4():
    
    def update_dl(data, new_value, type):

        if isinstance(data, dict):
            for key, value in data.items():
                if key == "digital_literacy" and isinstance(value, dict):
                    if type in value: 
                        value[type] = new_value
                else:
                    update_dl(value, new_value, type)
        elif isinstance(data, list):
            for item in data:
                update_dl(item, new_value, type)
        return data
    
    def update_new_tool(data, accuracy, tool_complexity, interop):

        for tool_name, tool in data.items():
            if tool_name == 'HFSystemSimulator':
                tool["parameters"]["accuracy"] = accuracy
                tool["parameters"]["use_complexity"] = tool_complexity
                tool["parameters"]["interoperability"] = interop
            
        return data
    
    def update_old_tools(data, accuracy_factor, interop):

        for tool_name, tool in data.items():
            if tool_name in {"KnowledgeBase", "HFSystemSimulator"}:
                continue

            if "parameters" in tool and tool["parameters"]["interoperability"]:
                tool["parameters"]["interoperability"] = interop
                
            # Update accuracy for digital tools if accuracy is not None
            if "accuracy" in tool["parameters"] and tool["parameters"]["accuracy"] and tool["type"] == "digital":
                if accuracy_factor >= 0:
                    tool["parameters"]["accuracy"] += (1 - tool["parameters"]["accuracy"]) * accuracy_factor
                else:
                    tool["parameters"]["accuracy"] += tool["parameters"]["accuracy"] * accuracy_factor


        return data
        
    
    with open('Architecture/Inputs/Baseline/organization.json', "r") as f:
        org_data_orig = json.load(f)
    with open('Architecture/Inputs/Baseline/tools_with_new.json', "r") as f:
        tools_data_with_new = json.load(f)
    with open('Architecture/Inputs/Baseline/tools.json', "r") as f:
        tools_data_orig = json.load(f)
        
    configs_df = pd.read_csv("DOE/DOE4.csv")
    
    for index, row in configs_df.iterrows():
        folder_name = 'Architecture/Inputs/DOE4 - Full/' + str(row["Config. No."])
        
        old_tool_acc = row["T_Acc (old)"]
        old_tool_inter = row["T_I (old)"]
        digital_lit_Eng = row["DL_Eng"]
        digital_lit_EKM = row["DL_EKM"]
        NewTool = row["New Tool"]
        new_tool_acc = row["T_Acc (new)"]
        new_tool_inter = row["T_I (new)"]
        new_tool_usab = row["T_Usab"]
        

        os.makedirs(folder_name, exist_ok=True)
        
        org_data = copy.deepcopy(org_data_orig)
        new_org_data = update_dl(org_data, digital_lit_EKM, 'EngineeringSupportTools')
        new_org_data = update_dl(new_org_data, digital_lit_Eng, 'EngineeringTools')
        
        with open(os.path.join(folder_name, "organization.json"), "w") as f:
            json.dump(new_org_data, f, indent=4)
        
        
        if NewTool == True:
            tools_data = copy.deepcopy(tools_data_with_new)
            new_tool_data = update_new_tool(tools_data, new_tool_acc, new_tool_usab, new_tool_inter)
        else:
            new_tool_data = copy.deepcopy(tools_data_orig)
        
        new_tool_data = update_old_tools(new_tool_data, old_tool_acc, old_tool_inter)
        
        with open(os.path.join(folder_name, "tools.json"), "w") as f:
            json.dump(new_tool_data, f, indent=4)
            

if __name__ == "__main__":
    main()
    
    #doe4()