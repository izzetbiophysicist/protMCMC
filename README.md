## protMCMC - Protein Optimization using Monte Carlo Markov Chain with PyRosetta and ESM2

This program applies a Monte Carlo Markov Chain (MCMC) approach to optimize protein sequences, leveraging PyRosetta for structural manipulation and ESM2 for amino acid prediction. It offers flexibility in guiding mutations using either random choices or machine learning-based predictions from the ESM2 model.

key Modules:

apt_function.py:
  Contains essential utility functions for protein modeling and fitness evaluations.
  Fitness Functions:
      apt_rosetta(): Evaluates a protein sequence using the PyRosetta energy function.
      apt_esm() and apt_esm_penalty(): Uses the ESM2 model to predict amino acid probabilities for a given sequence, with optional penalties for repeated residues or based on Shannon entropy.

protMCMC.py:
    Defines the protMCMC class to handle the MCMC optimization process.
    Key Methods:
        mc_optimize(): Runs the MCMC optimization. Mutations can be guided by ESM2 or generated randomly. The acceptance of mutations is based on the Metropolis criterion.
        get_all_sequences_and_scores(): Returns all generated sequences and their respective scores.
        get_best_score_and_sequence(): Retrieves the best sequence along with its score from the optimization.

runMCMC.py:
    Example script for running the MCMC optimization. It uses partial to define a fitness function and sets up the optimization run.
