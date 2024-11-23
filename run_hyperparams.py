import subprocess
import itertools

# Define the grid of hyperparameters
hyperparams_grid = {
    'lr': [0.1, 0.01, 0.001, 0.0001],
    'optimizer': ['adam', 'sgd'],
    'gradient_accumulation_steps': [2, 4, 6, 8],
    'clip_grad_norm': [None, 3, 5],
}

def generate_hyperparam_combinations(grid):
    # Generate all combinations of hyperparameters
    return list(itertools.product(*grid.values()))

def run_command_with_hyperparams(hyperparams):
    # Construct the command with the current combination of hyperparameters
    command = f"python main.py "  # Start the command
    
    # Add hyperparameters to the command
    for key, value in zip(hyperparams_grid.keys(), hyperparams):
        if key == 'clip_grad_norm' and value is None:
            continue
        command += f"--{key} {value} "
    
    # Add fixed arguments to the command
    command += "--template draftrec --/Users/jk/Documents/project_data --use_parallel true"

    # Execute the command
    print(f"Running command: {command}")
    
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def run_all_experiments():
    # Generate all hyperparameter combinations
    hyperparam_combinations = generate_hyperparam_combinations(hyperparams_grid)
    
    # Run each combination
    for hyperparams in hyperparam_combinations:
        run_command_with_hyperparams(hyperparams)

if __name__ == "__main__":
    run_all_experiments()