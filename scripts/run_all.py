import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()

    scripts_to_run = [
        "scripts.evaluate_active_units",
        "scripts.evaluate_fid",
        "scripts.evaluate_is",
        "scripts.evaluate_kid",
        "scripts.model_summary",
        "scripts.plot_curves",
        "scripts.plot_interpolation",
        "scripts.plot_latent"
    ]

    for script_module in scripts_to_run:
        print(f"\n---> Running module: {script_module}")
        
        command = [
            sys.executable, 
            "-m", script_module, 
            "--run-name", args.run_name
        ]

        try:
            subprocess.run(command, check=True)
            print(f"--->Successfully finished {script_module}.")
        except subprocess.CalledProcessError as e:
            print(f"--->[ERROR] {script_module} failed with exit code {e.returncode}.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("All evaluation scripts executed successfully!")


if __name__ == "__main__":
    main()
