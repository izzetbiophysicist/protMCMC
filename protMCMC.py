# prot_mcmc.py

import random
import numpy as np
import csv

import numpy as np
from numpy.random import uniform
from random import sample

from threading import Thread
from time import sleep
from pyrosetta import *
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation
from datetime import datetime
import pandas as pd
from apt_function import correct_multi_input, batchs_to_run

from pyrosetta import *
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation


#### ESM stuff
import torch
import esm
import random
import math

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

model.eval()  # disables dropout for deterministic results

# Initialize PyRosetta
pyrosetta.init()

# Assume these are defined or imported from other modules
# from your_library import alphabet, insert_mask, batch_converter, model
def insert_mask(sequence, position, mask="<mask>"):
    """
    Replaces a character in a given position of a sequence with a mask.

    Parameters:
    - sequence (str or list): The sequence to replace the character in.
    - position (int): The position in the sequence where the character should be replaced.
    - mask (str): The mask to insert (default is "<mask>").

    Returns:
    - str or list: The sequence with the mask replacing the character at the specified position.
    """
    
    if not (0 <= position < len(sequence)):
        raise ValueError("Position is out of bounds.")
    
    if isinstance(sequence, str):
        return sequence[:position] + mask + sequence[position + 1:]
    elif isinstance(sequence, list):
        return sequence[:position] + [mask] + sequence[position + 1:]
    else:
        raise TypeError("Sequence must be a string or list.")

def complete_mask(input_sequence, posi, temperature=1.0):
    # Standard amino acids
    standard_aa = [alphabet.get_idx(aa) for aa in ['A', 'R', 'N', 'D', 'C', 'Q', 
                                                   'E', 'G', 'H', 'I', 'L', 'K', 
                                                   'M', 'F', 'P', 'S', 'T', 'W', 
                                                   'Y', 'V']]

    # Insert <mask> at the desired position
    data = [("protein1", insert_mask(input_sequence, posi, mask="<mask>"))]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Predict masked tokens
    with torch.no_grad():
        token_probs = model(batch_tokens, repr_layers=[33])["logits"]

    # Apply temperature scaling
    token_probs /= temperature
    softmax = torch.nn.Softmax(dim=-1)
    probabilities = softmax(token_probs)

    # Get the index of the <mask> token
    mask_idx = (batch_tokens == alphabet.mask_idx).nonzero(as_tuple=True)

    # Zero out probabilities for non-standard amino acids
    for token_idx in range(probabilities.size(-1)):
        if token_idx not in standard_aa:
            probabilities[:, :, token_idx] = 0.0

    # Sample from the probability distribution for the masked position
    predicted_token = torch.multinomial(probabilities[mask_idx], num_samples=1).squeeze(-1)

    # Get the predicted amino acid
    predicted_residue = alphabet.get_tok(predicted_token.item())

    print(f"ESM Mutation at position {posi}: {predicted_residue}")
    
    return predicted_residue


# Define the optimization class
class protMCMC:
    def __init__(self, starting_pose):
        self.starting_pose = starting_pose
        self.scores = []  # List to store the scores (or likelihoods) of each pose
        self.sequences = []  # List to store the sequences of each pose

    def mc_optimize(self, locked_positions, n_iter=100, temp=1.0,  use_esm=False, 
                    fitness_function=None, output_path='output.csv', **fitness_kwargs):
        # Extract sequence from the starting pose
        sequence = self.starting_pose.sequence()
        best_score = fitness_function(sequence, starting_pose=self.starting_pose, **fitness_kwargs) if fitness_function else None
        current_score = best_score
        best_sequence = sequence

        # Open the output CSV file for writing
        with open(output_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Iteration', 'Sequence', 'Score'])  # Write header

            for i in range(n_iter):
                # Select a random position for mutation that is not locked
                posi = random.randint(1, len(best_sequence))
                print("TRYING "+str(posi))
                while posi in locked_positions:
                    posi = random.randint(1, len(best_sequence))

                # If using ESM, predict the mutation at the selected position
                if use_esm:
                    mutated_residue = complete_mask(best_sequence, posi-1, temperature=temp)
                else:
                    # Get the current residue at the selected position
                    current_residue = self.starting_pose.residue(posi).name1()

                    # Sample a new residue that's different from the current one
                    mutated_residue = random.choice(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                                                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
                    while mutated_residue == current_residue:
                        print(f"Resampling for position {posi} because {mutated_residue} is the same as current residue {current_residue}")
                        mutated_residue = random.choice(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                                                          'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])

                print(f"Selected position {posi}, Mutated residue: {mutated_residue} (Current: {self.starting_pose.residue(posi).name1()})")

                # Generate a new sequence
                new_sequence = list(best_sequence)
                new_sequence[posi - 1] = mutated_residue  # Adjust 1-based to 0-based index for sequence
                trial_sequence = ''.join(new_sequence)

                # Evaluate the score using the provided fitness function, if available
                if fitness_function:
                    trial_score = fitness_function(trial_sequence, starting_pose=self.starting_pose, **fitness_kwargs)
                else:
                    trial_score = None  # Placeholder if no fitness function is provided

                # Save the score and sequence
                self.scores.append(trial_score)
                self.sequences.append(trial_sequence)

                # Acceptance criterion using Metropolis
                delta_score = trial_score - current_score if trial_score is not None else 0
                if delta_score < 0 or (trial_score is not None and np.exp(-delta_score / temp) > random.random()):
                    current_score = trial_score
                    best_sequence = trial_sequence

                    # Update the best score if necessary
                    if current_score is not None and (best_score is None or current_score < best_score):
                        best_score = current_score

                # Write the current iteration's results to the CSV
                writer.writerow([i + 1, trial_sequence, trial_score])

                print(f"Iteration {i + 1}/{n_iter}: Current Score = {current_score}, Best Score = {best_score}")

        return best_score  # Return the best score

    # Method to retrieve all sequences and scores
    def get_all_sequences_and_scores(self):
        return self.sequences, self.scores

    # Method to retrieve the best score and sequence
    def get_best_score_and_sequence(self):
        if self.scores:
            best_idx = np.argmin(self.scores)
            return self.sequences[best_idx], self.scores[best_idx]
        return None, None
