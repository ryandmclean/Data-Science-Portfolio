import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Error running command: {cmd}")

def main():
    # Build warehouse tables
    run(["dbt", "run", "--project-dir", "learning_app_warehouse"])

    # Run tests
    # run(["dbt", "test", "--project-dir", "learning_app_warehouse"])

    # export outputs to csv
    exports = [
        ("event_stream", "data/exports/dl_event_stream.csv"),
        ("dim_user", "data/exports/dw_dim_user.csv"),
        ("fct_play_attempts", "data/exports/dw_fct_play_attempts.csv"),
        ("dim_user_device", "data/exports/dw_dim_user_device.csv"),
        ("fct_user_play_summary", "data/exports/dm_fct_user_play_summary.csv"),
    ]
    for model, output_path in exports:
        run([
            "dbt", 
            "run-operation", 
            "export_model_to_csv",
            "--project-dir", "learning_app_warehouse",
            "--args", f"{{model_name: {model}, output_path: {output_path}}}",
        ])

if __name__ == "__main__":
    main()