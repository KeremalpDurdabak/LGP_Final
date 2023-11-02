class Display:
    @staticmethod
    def report_generation_performance(population, generation):
        best_individual_error = population.errors[0]
        best_individual = population.individuals[0]
        formula = ' + '.join(f'{weight:.2f}x{i+1}' for i, weight in enumerate(best_individual[:-1]))
        formula += f' + {best_individual[-1]:.2f}'
        print(f"Gen {generation}: Best individual error: {best_individual_error:.2f} with formula: {formula}")
