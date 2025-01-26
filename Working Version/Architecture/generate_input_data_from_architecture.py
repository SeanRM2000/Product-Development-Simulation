import pandas as pd
import json


def generate_symmetric_dsm(input_csv, output_csv):
    """
    Generates a symmetric Design Structure Matrix (DSM) from an interfaces CSV file.
    
    Args:
        input_csv (str): Path to the input CSV file containing interfaces data.
        output_csv (str): Path to save the generated DSM as a CSV file.
    
    Returns:
        pd.DataFrame: The symmetric DSM as a Pandas DataFrame.
    """
    # Load the CSV file
    interfaces_df = pd.read_csv(input_csv)

    # Rename columns for easier reference
    interfaces_df.columns = [
        'Component_A', 'Component_B', 'Geometric_Spatial', 
        'Structural', 'Energy', 'Material', 'Information_Data'
    ]

    # Sum the severities across all interface types for each row
    interfaces_df['Total_Severity'] = interfaces_df[
        ['Geometric_Spatial', 'Structural', 'Energy', 'Material', 'Information_Data']
    ].sum(axis=1)

    # Create a symmetric DSM
    # Step 1: Initialize an empty DSM with unique components as rows and columns
    unique_components = pd.concat([interfaces_df['Component_A'], interfaces_df['Component_B']]).unique()
    dsm_matrix = pd.DataFrame(0, index=unique_components, columns=unique_components)

    # Step 2: Populate the DSM by summing severities for each pair of components
    for _, row in interfaces_df.iterrows():
        comp_a = row['Component_A']
        comp_b = row['Component_B']
        severity = row['Total_Severity']
        
        dsm_matrix.loc[comp_a, comp_b] += severity
        dsm_matrix.loc[comp_b, comp_a] += severity  # Ensure symmetry

    # Save the DSM to a CSV file
    dsm_matrix.to_csv(output_csv)
    print(f"DSM generated and saved to: {output_csv}")

    return dsm_matrix


def generate_json_structure(knowledge_requirements_path, properties_path, hierarchy_path, output_path):
    """
    Generates a JSON structure based on input CSV files.

    Parameters:
    - knowledge_requirements_path: str, path to the 'Knowledge Requirements' CSV file.
    - properties_path: str, path to the 'Properties' CSV file.
    - hierarchy_path: str, path to the 'Hierarchy' CSV file.
    - output_path: str, path where the output JSON file will be saved.
    """

    # Load the data from CSV files
    knowledge_requirements = pd.read_csv(knowledge_requirements_path)
    properties = pd.read_csv(properties_path)
    hierarchy = pd.read_csv(hierarchy_path)

    # Align the naming conventions by removing '.Knowledge Requirement' suffix
    knowledge_requirements['Name'] = knowledge_requirements['Name'].str.replace('.Knowledge Requirement', '', regex=False)

    # Clean the knowledge vector keys by removing ': Knowledge Level'
    knowledge_requirements.rename(columns=lambda x: x.replace(' : Knowledge Level', ''), inplace=True)




    # Build the JSON structure
    def build_json_structure(node_name, hierarchy, knowledge, properties):
        # Find children of the current node
        children_df = hierarchy[hierarchy['Name'] == node_name]
        if not children_df.empty and not children_df['Composition'].isnull().all():
            children_names = children_df['Composition'].iloc[0].split("\r\n")
        else:
            children_names = []

        # Fetch the node's knowledge vector
        knowledge_row = knowledge[knowledge['Name'] == node_name]
        if not knowledge_row.empty:
            knowledge_vector = knowledge_row.iloc[0].drop('Name').to_dict()
        else:
            knowledge_vector = {}

        # Fetch the node's properties
        properties_row = properties[properties['Name'] == node_name]
        if not properties_row.empty:
            properties_data = properties_row.iloc[0]
            node_properties = {
                "req_importance": properties_data["Importance : Real"],
                "prototype_start_condition": properties_data["Prototyping Start Condition : Start Condition"],
                "novelty": properties_data["Novelty : Real"]
            }
        else:
            node_properties = {}

        # Build the node structure
        node_structure = {
            "knowledge_vector": knowledge_vector,
            **node_properties
        }

        # If there are children, recursively process them
        if children_names:
            node_structure["children"] = {
                child.strip(): build_json_structure(child.strip(), hierarchy, knowledge, properties) for child in children_names
            }

        return node_structure

    # Start with the root node
    root_node = hierarchy.iloc[0]['Name']
    json_structure = {root_node: build_json_structure(root_node, hierarchy, knowledge_requirements, properties)}

    # Save the JSON structure to the specified output path
    with open(output_path, 'w') as json_file:
        json.dump(json_structure, json_file, indent=4)

    print(f"JSON structure has been successfully generated and saved to {output_path}.")





if __name__ == "__main__":
    # Example usage
    input_file = 'Interfaces.csv'  # Replace with your input file path
    output_file = 'DSM_Output.csv'  # Replace with your desired output file path
    dsm_result = generate_symmetric_dsm(input_file, output_file)

    # Display the result (optional)
    print(dsm_result)

    # Example usage:
    # generate_json_structure('Knowledge Requirements.csv', 'Properties.csv', 'Hierarchy.csv', 'generated_structure.json')