
import os

commands =[]

for i in range(11):
    commands.append("python dimreduction.py -d 2 -m umap -id {}".format(29-i))

for i in range(len(commands)):
    os.system(commands[i])