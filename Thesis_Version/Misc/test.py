import json

def check_unique_responsibilities(org_json):
    """Checks if every responsibility (activity, architecture element) exists only once in the given JSON structure."""
    responsibility_tracker = {}
    duplicates_found = False

    def process_team(team):
        """Recursively processes a team, checking responsibilities of the manager and members."""
        # Check manager's responsibilities
        manager = team.get("Manager", {})
        if manager:
            process_responsibilities(manager.get("responsibilities", {}), manager["name"])

        # Check members' responsibilities
        for member in team.get("Members", []):
            process_responsibilities(member.get("responsibilities", {}), member["name"])

        # Recursively process subteams
        for subteam in team.get("Subteams", []):
            process_team(subteam)

    def process_responsibilities(responsibilities, assignee):
        """Processes and checks unique responsibilities."""
        nonlocal duplicates_found 
        for activity, elements in responsibilities.items():
            for element in elements:
                key = (activity, element)
                if key in responsibility_tracker:
                    print(f"Duplicate responsibility found: {key} assigned to both {responsibility_tracker[key]} and {assignee}")
                    duplicates_found = True
                else:
                    responsibility_tracker[key] = assignee

    # Start processing the organization from the top-level team
    project_team = org_json.get("Project Team", {})
    if project_team:
        process_team(project_team)

    if not duplicates_found:
        print("All responsibilities are unique.")

# Example usage:
with open("Architecture\Inputs\Baseline\organization_copy.json", "r") as file:
    org_data = json.load(file)

check_unique_responsibilities(org_data)
