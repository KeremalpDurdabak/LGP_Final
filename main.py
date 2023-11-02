from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from modules.Parameter import Parameter
from modules.Dataset import Dataset
from modules.Population import Population
from modules.Display import Display

def sklearn_linear_regression():
    model = LinearRegression()
    model.fit(Dataset.X_train, Dataset.y_train)
    y_pred = model.predict(Dataset.X_test)
    error = mean_squared_error(Dataset.y_test, y_pred)

    print('')
    print('Test Dataset - Scikit-learn')
    
    print(f"Optimized Model: {error:.2f} MSE")

    # Print the equation
    equation = " + ".join([f"{weight:.2f}x{i+1}" for i, weight in enumerate(model.coef_)])
    equation += f" + {model.intercept_:.2f}"
    print(f"Equation: {equation}")

    print('')

def evolve(population, dataset):

    for generation in range(1, Parameter.generations + 1):

        population.compute_generation_fitness()

        Display.report_generation_performance(population,generation)

        # Parent Selection -> Variation Operators (Crossover, Mutation) -> Child Replacement
        population.generate_next_gen()

    #Compute best individual's score on the test dataset
    population.compute_best_individual_test_fitness()
    sklearn_linear_regression()
    
    # Display overall metrics as graphs    
    #Display.report_overall_performance()

if __name__ == "__main__":
    dataset = Dataset()
    dataset.set_real_estate()

    population = Population()
    population.initialize()

    try:
        evolve(population, dataset)
    except KeyboardInterrupt:
        population.compute_best_individual_test_fitness()
        sklearn_linear_regression()