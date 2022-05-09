import os

commands =[]

for i in range(29):
    commands.append("python dimreduction_old.py -d 3 -m umap -id {}".format(i))

for i in range(len(commands)):
    os.system(commands[i])

