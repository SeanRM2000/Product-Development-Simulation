import networkx as nx
import json
import matplotlib.pyplot as plt

from Inputs.sim_inputs import *

class OrganizationalGraph:
    def __init__(self, test_data=False):
        self.organization = nx.DiGraph()

        file_path = 'Inputs/test_data/test_organization.json' if test_data else 'Inputs/organization.json'

        with open(file_path, 'r') as file:
            org_data = json.load(file)
            
        self.knowledge_items = []
        self.digital_literacy_items = []
        self.knowledge_bases = []

        project_team = list(org_data.values())[0]
        self._recursivly_build_graph(project_team)
        self._aggregate_responsibilities_and_product_knowledge()

    def _add_agent(self, data):
        name = data.get('name')
        
        knowledge_vector = data.get('knowledge_vector')
        expertise = list(knowledge_vector.values())
        if not self.knowledge_items:
            self.knowledge_items = list(knowledge_vector.keys())
        
        digital_literacy_dict = data.get('digital_literacy')
        digital_literacy = list(digital_literacy_dict.values())
        if not self.digital_literacy_items:
            self.digital_literacy_items = list(digital_literacy_dict.keys())

        familiarity = data.get('knowledge_base_familiarity')
        knowledge_base_familiarity = list(familiarity.values())
        if not self.knowledge_bases:
            self.knowledge_bases = list(familiarity.keys())

        self.organization.add_node(
            name,
            node_type='Agent',
            profession=data.get('profession'),
            expertise=expertise,
            initial_expertise=expertise.copy(),
            product_knowledge={},
            digital_literacy=digital_literacy,
            initial_digital_literacy=digital_literacy.copy(),
            knowledge_base_familiarity=knowledge_base_familiarity,
            initial_knowledge_base_familiarity=knowledge_base_familiarity.copy(),
            responsibilities=data.get('responsibilities'),
            decision_making=data.get('decision_making'),
            salary=data.get('salary'),
            availability=data.get('availability'),
            state='Idle',
            technical_task='',
            tool_in_use='',
            task_queue={},
            requests=[]
        )
        return name

    def _recursivly_build_graph(self, team, manager=None):
        team_name = team.get('Team name')
        self._add_team(team_name)

        manager_name = self._add_manager(team.get('Manager'), team_name, manager)

        if 'Subteams' in team:
            for subteam in team.get('Subteams'):
                sub_team_name = self._recursivly_build_graph(subteam, manager_name)
                self.organization.add_edge(sub_team_name, team_name, relation='part_of')
        else:
            for member in team.get('Members'):
                self._add_member(member, team_name, manager_name)
        
        return team_name

    def _add_member(self, data, team, manager):
        name = self._add_agent(data)
        self.organization.add_edge(name, team, relation='member_of')
        self.organization.add_edge(name, manager, relation='reports_to')

    def _add_manager(self, data, team, manager=None):
        name = self._add_agent(data)
        self.organization.add_edge(name, team, relation='manages')
        if manager:
            self.organization.add_edge(name, manager, relation='reports_to')
        return name

    def _add_team(self, name):
        self.organization.add_node(
            name,
            node_type='Team',
            responsibilities={}
        )

    def _aggregate_responsibilities_and_product_knowledge(self):
        all_architecture_elements = set()
        for node in nx.topological_sort(self.organization):
            if self.organization.nodes[node]['node_type'] == 'Team':
                team_responsibilities = {}

                # add all responsibilities of members to team
                for member in self.get_members(node):
                    for func, prod in self.organization.nodes[member]['responsibilities'].items():
                        if func not in team_responsibilities:
                            team_responsibilities[func] = []
                        team_responsibilities[func].extend(prod)
                        for element in prod:
                            all_architecture_elements.add(element)

                for func in team_responsibilities:
                    team_responsibilities[func] = list(set(team_responsibilities[func]))

                self.organization.nodes[node]['responsibilities'] = team_responsibilities
        
        # product knowledge
        product_knowledge = {
            'Req_Knowledge': {},
            'Design_Knowledge': {}
        }
        for element in all_architecture_elements:
            product_knowledge['Req_Knowledge'][element] = initial_req_knowledge
            product_knowledge['Design_Knowledge'][element]= inital_design_knowledge
        
        for member in self.get_all_agents():
            self.organization.nodes[member]['product_knowledge'] = product_knowledge


    def plot_organization(self):
        agent_nodes = [n for n, attr in self.organization.nodes(data=True) if attr['node_type'] == 'Agent']
        
        subgraph = self.organization.subgraph(agent_nodes).copy()

        pos = nx.spring_layout(subgraph)
        plt.figure(figsize=(12, 8))

        nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color='skyblue')
        nx.draw_networkx_edges(subgraph, pos, arrows=True)
        nx.draw_networkx_labels(subgraph, pos, font_size=10, font_family='sans-serif')

        edge_labels = nx.get_edge_attributes(subgraph, 'relation')
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='black')

        plt.title("Organizational Graph (Agents Only)", fontsize=16)
        plt.axis('off')
        plt.show()


    def get_all_teams(self):
        return [node for node, attr in self.organization.nodes(data=True) if attr['node_type'] == 'Team']
    
    def get_subordinates(self, manager):
        return [node for node in self.organization.predecessors(manager) if self.organization.edges[node, manager]['relation'] == 'reports_to']

    def get_members(self, team):
        return [node for node in self.organization.predecessors(team) if self.organization.nodes[node]['node_type'] == 'Agent']

    def get_manager(self, agent=None, team=None):
        if agent and team:
            raise ValueError('Provide either an agent or a team.')

        if agent:
            for predecessor in self.organization.successors(agent):
                if self.organization.edges[agent, predecessor]['relation'] == 'reports_to':
                    return predecessor

        if team:
            for predecessor in self.organization.predecessors(team):
                if self.organization.edges[predecessor, team]['relation'] == 'manages':
                    return predecessor

        return None
    
    def get_all_agents(self):
        return [node for node, attr in self.organization.nodes(data=True) if attr['node_type'] == 'Agent']
    
    
    def get_agent(self, agent):
        return self.organization.nodes[agent]
    
    def get_common_manager(self, teams):
        # Get the direct managers of each team
        direct_managers = [self.get_manager(team=team) for team in teams]

        # If all teams share the same direct manager, return that manager
        if len(set(direct_managers)) == 1:
            return direct_managers[0]

        # Use a BFS approach to find the first common manager up the hierarchy
        visited = set()
        queue = []

        # Initialize the queue with the direct managers of all teams
        for manager in direct_managers:
            if manager:
                queue.append((manager, manager))  # (current node, original manager)

        while queue:
            current, origin = queue.pop(0)  # BFS: Process the current manager
            
            if current not in visited:
                visited.add(current)

                # Check if this manager is responsible for all teams
                if all(self._is_manager_of(current, team) for team in teams):
                    return current

                # Add the next level of managers to the queue (higher in hierarchy)
                parent_manager = self.get_manager(agent=current)
                if parent_manager:
                    queue.append((parent_manager, origin))

        # No common manager found
        return None

    def _is_manager_of(self, manager, team):
        current_manager = self.get_manager(team=team)
        while current_manager:
            if current_manager == manager:
                return True
            current_manager = self.get_manager(agent=current_manager)
        return False
    
    
    def get_team(self, agent):
        return [node for node in self.organization.successors(agent) if self.organization.nodes[node]['node_type'] == 'Team'][0]


if __name__ == "__main__":

    org_graph = OrganizationalGraph(test_data=True)

    # Plot the organization graph
    org_graph.plot_organization()

    # Example usage
    teams = org_graph.get_all_teams()
    print("Teams:", teams)

    for team in teams:
        members = org_graph.get_members(team)
        print(f"Members of {team}:", members)
        print(f'Responsibilities: {org_graph.organization.nodes[team]['responsibilities']}')
        manager = org_graph.get_manager(team=team)
        print(f"Manager of {team}:", manager)
        
    print(org_graph.get_subordinates('Project Manager'))
    
    print(org_graph.get_agent('Project Manager'))