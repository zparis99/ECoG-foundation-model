import yaml
import itertools
import subprocess
from datetime import datetime


def read_config(config_path):
    """Read the parameter sweep configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_job_prefix(param_names, combination):
    """Generate a descriptive prefix based on the parameters being varied."""
    parts = []
    for name, value in zip(param_names, combination):
        # Convert parameter name from kebab-case to snake_case for the prefix
        name = name.replace("-", "_")
        # Handle different types of values
        if isinstance(value, (int, float)):
            # Format numbers without decimal places if they're integers
            value_str = f"{value:g}"
        else:
            value_str = str(value)
        parts.append(f"{name}{value_str}")

    return "-".join(parts)


def generate_parameter_combinations(config):
    """Generate all possible combinations of parameters."""
    # Separate base config from parameter sweeps
    base_config = config.get("base-config-file")

    # Get parameter names and their values for sweeping
    param_dict = {k: v for k, v in config.items() if isinstance(v, list)}

    # Generate all combinations
    param_names = list(param_dict.keys())
    param_values = [param_dict[name] for name in param_names]

    return base_config, param_names, itertools.product(*param_values)


def submit_job(combination, param_names, base_config, job_index):
    """Submit a single job with the given parameter combination."""
    # Create a descriptive prefix based on the parameters
    prefix = generate_job_prefix(param_names, combination)

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"{prefix}-{timestamp}"

    # Build the command
    cmd = ["sbatch"]
    cmd.extend(["--job-name", job_name])
    cmd.extend(["submit.sh", "ECoG_MAE/main.py"])
    cmd.extend(["--job-name", job_name])
    cmd.extend(["--config-file", base_config])

    # Add parameter combinations
    for param_name, param_value in zip(param_names, combination):
        cmd.extend([f"--{param_name}", str(param_value)])

    # Execute the command
    print(f"Submitting job {job_index + 1}:")
    print(f"Job Name: {job_name}")
    print("Command:", " ".join(str(x) for x in cmd))
    print("-" * 80)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")


def main():
    # Setup argument parser for this script
    import argparse

    parser = argparse.ArgumentParser(description="Run parameter sweep jobs")
    parser.add_argument(
        "config_file", help="Path to parameter sweep configuration file"
    )
    args = parser.parse_args()

    # Read config
    config = read_config(args.config_file)

    # Generate combinations
    base_config, param_names, combinations = generate_parameter_combinations(config)

    # Submit jobs for each combination
    for i, combination in enumerate(combinations):
        submit_job(combination, param_names, base_config, i)


if __name__ == "__main__":
    main()
