from asyncio import subprocess
import os

commands =[]

for i in range(30):
    commands.append("python dimreduction.py -d 2 -m umap -id {}".format(i))

for i in range(len(commands)):
    os.system(commands[i])