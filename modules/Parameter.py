from dataclasses import dataclass


@dataclass
class Parameter:
    # Number of Individuals in the Population
    population_count = 100

    # Percentage of worst fit Individuals to replace
    gap_percentage = 0.3

    # Generation Count
    generations = 100000

    # Probability of a Mutation
    mutation_prob = 0.3