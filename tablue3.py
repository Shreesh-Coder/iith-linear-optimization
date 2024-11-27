import numpy as np
import pandas as pd

def read_csv_input(file_path):
    """
    Reads the CSV file and parses the initial feasible point, cost vector,
    constraint matrix, and constraint vector.
    """
    df = pd.read_csv(file_path, header=None)
    data = df.values

    m_plus_2, n_plus_1 = data.shape
    m = m_plus_2 - 2
    n = n_plus_1 - 1

    # Initial feasible point z (not used in simplex, but parsed as per specification)
    z = data[0, :-1].astype(float)

    # Cost vector c
    c = data[1, :-1].astype(float)

    # Constraint vector b
    b = data[2:, -1].astype(float)

    # Constraint matrix A
    A = data[2:, :-1].astype(float)

    return z, c, A, b

def simplex(c, A, b):
    """
    Performs the simplex algorithm using the tableau method with Bland's Rule
    to handle degeneracy.
    Returns the sequence of vertices and their objective function values.
    """
    m, n = A.shape

    # Initialize the tableau
    # The tableau has m + 1 rows and n + m + 1 columns
    # Last row is the objective function
    # Slack variables are added to convert inequalities to equalities
    tableau = np.zeros((m + 1, n + m + 1))

    # Fill constraint coefficients and slack variables
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b

    # Fill the objective function
    tableau[m, :n] = -c

    # Basis: Initially, slack variables are the basic variables
    basis = list(range(n, n + m))

    vertices = []
    objective_values = []

    while True:
        # Record current basic variables' values
        basic_vars = tableau[:m, -1]
        current_vertex = np.zeros(n + m)
        for i in range(m):
            current_vertex[basis[i]] = basic_vars[i]
        vertices.append(current_vertex.copy())

        # Compute current objective value
        z = tableau[m, -1]
        objective_values.append(z)

        # Check for optimality (no negative coefficients in objective row)
        # Use a small epsilon to handle numerical precision
        epsilon = 1e-8
        if all(tableau[m, :-1] >= -epsilon):
            break

        # Determine entering variable using Bland's Rule:
        # Choose the smallest index with a negative coefficient in the objective row
        entering_candidates = [j for j in range(n + m) if tableau[m, j] < -epsilon]
        if not entering_candidates:
            break  # Optimal

        entering = min(entering_candidates)

        # Determine leaving variable using minimum ratio test with Bland's Rule
        ratios = []
        for i in range(m):
            if tableau[i, entering] > epsilon:
                ratios.append(tableau[i, -1] / tableau[i, entering])
            else:
                ratios.append(np.inf)

        min_ratio = min(ratios)
        leaving_candidates = [i for i, ratio in enumerate(ratios) if ratio == min_ratio]

        if not leaving_candidates:
            raise Exception("Linear program is unbounded.")

        # Choose the leaving variable with the smallest index (Bland's Rule)
        leaving = min(leaving_candidates)

        # Pivot
        pivot = tableau[leaving, entering]
        tableau[leaving, :] = tableau[leaving, :] / pivot
        for i in range(m + 1):
            if i != leaving:
                tableau[i, :] = tableau[i, :] - tableau[i, entering] * tableau[leaving, :]

        # Update basis
        basis[leaving] = entering

    return vertices, objective_values

def print_results(vertices, objective_values, n, m):
    """
    Prints the sequence of vertices and their corresponding objective function values.
    """
    print("Sequence of vertices visited and their objective function values:")
    for idx, (vertex, z) in enumerate(zip(vertices, objective_values)):
        var_values = vertex[:n]  # Assuming first n variables are original variables
        print(f"Step {idx + 1}: Vertex {var_values}, Objective Value = {z}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simplex Algorithm Implementation with Degeneracy Handling')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    args = parser.parse_args()

    # Read input
    z, c, A, b = read_csv_input(args.csv_file)

    m, n = A.shape

    # Perform simplex
    vertices, objective_values = simplex(c, A, b)

    # Print results
    print_results(vertices, objective_values, n, m)

if __name__ == "__main__":
    main()
