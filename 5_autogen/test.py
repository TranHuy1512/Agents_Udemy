import glob
import os
import random

agent_files = glob.glob("agent*.py")
agent_names = [os.path.splitext(file)[0] for file in agent_files]
agent_names.remove("agent")
agent_name = random.choice(agent_names)
print(f"Selecting agent for refinement: {agent_name}")
# print(agent_names)
# print(agent_files)
# agent_names = os.path.splitext(agent3.py)[0]
print(agent_name)
