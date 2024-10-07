# run_optimization.py

from protMCMC import protMCMC
import numpy as np
from functools import partial


import os

from apt_function import *

from pyrosetta import *
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation

from functools import partial

import numpy as np
from numpy.random import uniform
from random import sample
import random


# Assuming you have a function defined for fitness evaluation
# from your_library import your_fitness_function

# Instructions on how to use partial with a fitness function:
# To use a custom fitness function with the `protMCMC` class, define your function and use `partial` like this:
# fitness_function = partial(your_fitness_function, arg1=value1, arg2=value2)
# Then, pass it to the mc_optimize method like so:
# best_score = optimizer.mc_optimize(locked_positions=[1, 2, 3], n_iter=100, fitness_function=fitness_function, output_path='output.csv')


# Example usage

    # Use partial to preset some arguments for the fitness function
    # fitness_function = partial(your_fitness_function, arg1=value1, arg2=value2)
    
    # Instantiate your optimization class

my_pose= pose_from_pdb('2lzt.pdb')
# Get the total number of residues

fitness_function = partial(
    apt_rosetta,
    starting_pose=my_pose,
    scorefxn=pyrosetta.create_score_function("ref2015_cart.wts")
)
    # Run the optimization

optimizer = protMCMC(starting_pose=my_pose)
optimizer.mc_optimize(
        locked_positions=[],  # Specify locked positions
        n_iter=100,  # Number of iterations
        fitness_function=fitness_function,  # Replace with your fitness function if needed
        output_path='output.csv',  # Specify your output CSV file path
    use_esm=True)
#print(f"Best score: {best_score}")


