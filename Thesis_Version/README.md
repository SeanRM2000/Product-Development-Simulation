# Master-Thesis---Product-Development-Simulation
An agent-based simulation to evaluate product development

Potential Name: ProDevSim Studio (Planner, Manager)

USE PAPER VERSION

Architecture --> Inputs: Creation of Configurations for the Organization (Personnel, Tools), Goals, Product

<sim_run.py> for single runs

<monte_carlo.py> for monte carlo of the configurations


Sim settings:
some at the bottom of the file and some in <Inputs> --> <sim_settings.py> and <tuning_params.py>




Some remarks to additional functionalities not discussed in the thesis:
1. Task network generation: Different structures for the tasks that are part of an activity can be generated. This is done by randomly parallelizing tasks. In the thesis this was not considered and a linear sequence of activities was created. Parallelized tasks can not be used with current implementation. Therefore process rules still exist but have to be defined in a correct way for the simulation to work correctly.
2. Availability of tools and personnel was not concerned in the thesis, therefore noise was not simulated in the thesis
3. Managers are included in the simulation but do not currently behave differently than regular agent 
        --> Decision-making does not do anything (yet)
        --> Prefixed task responsibilities at the beginning (dynamic would be better especially for prototyping, testing, simulation)
4. Overlapping and other process rules were not used in the thesis due to not being hard to implement --> synchonization issues
5. Only one knowledge base can be defined right now
6. Partial Integration is not possible (all subsystems have to also be virtually integrated)
7. Some artefacts in the input data were kept to allow for the simulation to function after some changes to the logic were made. These do not have an impact on the simulation itself. (e.g. Project Manager)
8. Some properties and entities have been named differently in the thesis for better understanding
--> Generally limitations are: One-to-One assignment of all design/system design activities is required, and no overlapping of tasks is allowed