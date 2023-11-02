import numpy as np
import random
from sklearn.metrics import mean_squared_error
from modules.Dataset import Dataset
from modules.Parameter import Parameter

class Population:
    def __init__(self):
        self.individuals = None
        self.lower_bounds = None
        self.upper_bounds = None
        self.sigma = 1
        self.errors = None

    def initialize(self):
        feature_count = Dataset.X_train.shape[1]
        self.individuals = np.random.uniform(-10, 10, 
                                            (Parameter.population_count, feature_count + 1))
        self.lower_bounds = np.full(feature_count + 1, -10)
        self.upper_bounds = np.full(feature_count + 1, 10)

        # Initialize self.errors with high values
        self.errors = np.full(Parameter.population_count, np.inf)


    def compute_generation_fitness(self):
        gap_size = int(Parameter.gap_percentage * Parameter.population_count)
        parents = self.individuals[:-gap_size]
        children = self.individuals[-gap_size:]

        y_preds_children = np.dot(Dataset.X_train, children[:, :-1].T) + children[:, -1]
        errors_children = np.mean((Dataset.y_train[:, None] - y_preds_children) ** 2, axis=0)
        
        # Combine errors of parents and children, and then sort
        combined_errors = np.concatenate([self.errors[:-gap_size], errors_children])
        sorted_indices = np.argsort(combined_errors)
        
        self.individuals = self.individuals[sorted_indices]
        self.errors = combined_errors[sorted_indices]

        best_individual = self.individuals[0]
        bound_gap = 5  
        self.lower_bounds = np.minimum(self.lower_bounds, best_individual - bound_gap)
        self.upper_bounds = np.maximum(self.upper_bounds, best_individual + bound_gap)

        if np.std(self.errors) < 0.1:
            self.sigma *= 1.2

    def generate_next_gen(self):
        gap_size = int(Parameter.gap_percentage * Parameter.population_count)
        parents = self.individuals[:-gap_size]
        children = []

        while len(children) < gap_size:
            parent1, parent2 = random.choices(parents, k=2)
            alpha = random.random()
            child = alpha * parent1 + (1 - alpha) * parent2
            
            if random.random() < Parameter.mutation_prob:
                mutation_point = random.randint(0, len(child) - 1)
                mutation_value = np.random.normal(0, self.sigma)
                child[mutation_point] += mutation_value
                child[mutation_point] = np.clip(child[mutation_point], 
                                                self.lower_bounds[mutation_point], 
                                                self.upper_bounds[mutation_point])
            children.append(child)

        self.individuals = np.vstack([parents, np.array(children)])

    def compute_best_individual_test_fitness(self):
        best_individual = self.individuals[0]
        y_pred_test = np.dot(Dataset.X_test, best_individual[:-1]) + best_individual[-1]
        error_test = mean_squared_error(Dataset.y_test, y_pred_test)
        
        formula = ' + '.join(f'{weight:.2f}x{i+1}' for i, weight in enumerate(best_individual[:-1]))
        formula += f' + {best_individual[-1]:.2f}'
        print('')
        print('Test Dataset - Best Individual')
        print(f"Best individual: {error_test:.2f} MSE")
        print(f"Equation: {formula}")
