"""
@file run_experiment.py
@brief Multi-seed orchestration for ablation study training runs.

Iterates over baseline configs and seeds, calling train() for each combination.
Creates distinct output directories per run: logs/<baseline>/seed_<N>/

Each baseline config (in configs/baselines/) contains only override keys
(baseline_name, include_covariance, policy_type, output dirs). These are
merged on top of the shared base config (configs/train_config.yaml) so that
all baselines share identical PPO hyperparameters, sensor noise, etc.

Supports parallel execution via the `parallel_workers` config parameter or
`--workers` CLI flag. Each parallel worker uses a separate CARLA port
(base_port + 3 * worker_index) to avoid conflicts.

This script runs INSIDE the training Docker container where CARLA, ROS 2, and
PyTorch/SB3 are available. Use `make docker-experiment` or `docker compose exec`
to invoke it. Dry-run mode works locally without Docker for planning.

Usage examples (inside training container or via Make):

    # Dry run locally (no Docker needed) - plan all runs
    make experiment-dry

    # Dry run inside container
    make docker-experiment-dry

    # Run full ablation study inside container (4 baselines x 10 seeds)
    make docker-experiment

    # Or invoke directly inside the training container shell:
    docker compose exec training python scripts/run_experiment.py \\
        --configs configs/baselines/*.yaml --dry-run

    # Run just vanilla_ppo with 3 seeds
    docker compose exec training python scripts/run_experiment.py \\
        --configs configs/baselines/vanilla_ppo.yaml --seeds 0 1 2

    # Run all baselines, override to 5 seeds
    docker compose exec training python scripts/run_experiment.py \\
        --configs configs/baselines/vanilla_ppo.yaml \\
                  configs/baselines/input_uncertainty.yaml \\
                  configs/baselines/output_uncertainty.yaml \\
                  configs/baselines/full_method.yaml \\
        --seeds 0 1 2 3 4

    # Run with 3 parallel workers (each gets its own CARLA port)
    docker compose exec training python scripts/run_experiment.py \\
        --configs configs/baselines/*.yaml --workers 3

    # Filter to specific baselines
    docker compose exec training python scripts/run_experiment.py \\
        --configs configs/baselines/*.yaml \\
        --baseline-filter vanilla_ppo full_method \\
        --dry-run
"""

import argparse
import copy
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    @brief Load a YAML config file.
    @param config_path: Path to YAML file.
    @return Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


def merge_configs(
    base: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """
    @brief Deep-merge overrides into a copy of base config.

    Nested dicts are merged recursively. Non-dict values in overrides
    replace the corresponding base values.

    @param base: Base configuration (e.g. train_config.yaml).
    @param overrides: Override configuration (e.g. baseline YAML).
    @return Merged configuration dictionary.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _assign_carla_port(
    config: Dict[str, Any], worker_index: int
) -> Dict[str, Any]:
    """
    @brief Assign a CARLA port to a run based on its worker index.

    Each CARLA instance uses 3 consecutive ports (world, streaming, RPC).
    Worker 0 uses base_port, worker 1 uses base_port+3, etc.

    @param config: Run configuration dictionary.
    @param worker_index: Zero-based index of the parallel worker.
    @return Config with updated carla_port.
    """
    run_config = copy.deepcopy(config)
    base_port = run_config.get("carla_port", 2000)
    run_config["carla_port"] = base_port + (3 * worker_index)
    return run_config


def run_single(
    config: Dict[str, Any], seed: int, dry_run: bool = False
) -> None:
    """
    @brief Run a single training experiment with the given config and seed.
    @param config: Fully merged config dict (will be copied and modified).
    @param seed: Random seed for this run.
    @param dry_run: If True, print the command but do not execute.
    """
    run_config = copy.deepcopy(config)
    baseline_name = run_config.get("baseline_name", "unknown")

    # Set seed-specific output directories
    run_config["seed"] = seed
    run_config["log_dir"] = f"./logs/{baseline_name}/seed_{seed}"
    run_config["checkpoint_dir"] = f"./checkpoints/{baseline_name}/seed_{seed}"

    if dry_run:
        print(
            f"  [DRY RUN] baseline={baseline_name}, seed={seed}, "
            f"log_dir={run_config['log_dir']}, "
            f"checkpoint_dir={run_config['checkpoint_dir']}, "
            f"policy_type={run_config.get('policy_type', 'evidential')}, "
            f"include_covariance="
            f"{run_config.get('include_covariance', True)}, "
            f"carla_port={run_config.get('carla_port', 2000)}"
        )
        return

    # Import here to avoid loading torch/SB3 during dry runs
    from uncertainty_rl.training.train_ppo import train

    print(f"\n{'=' * 60}")
    print(
        f"Starting: {baseline_name}, seed={seed}, "
        f"port={run_config.get('carla_port', 2000)}"
    )
    print(f"{'=' * 60}\n")

    start_time = time.monotonic()
    train(run_config)
    elapsed = time.monotonic() - start_time

    print(f"\nCompleted {baseline_name}/seed_{seed} in {elapsed:.1f}s")


def _run_worker(args: Tuple[Dict[str, Any], int, bool]) -> str:
    """
    @brief Worker function for parallel execution via ProcessPoolExecutor.
    @param args: Tuple of (config, seed, dry_run).
    @return Status message string.
    """
    config, seed, dry_run = args
    baseline_name = config.get("baseline_name", "unknown")
    try:
        run_single(config, seed, dry_run=dry_run)
        return f"OK: {baseline_name}/seed_{seed}"
    except Exception as e:
        return f"FAILED: {baseline_name}/seed_{seed}: {e}"


def main() -> None:
    """
    @brief Main entry point for the orchestration script.
    """
    parser = argparse.ArgumentParser(
        description="Run ablation study training across baselines and seeds"
    )
    parser.add_argument(
        "--base-config",
        default="configs/train_config.yaml",
        help="Path to shared base config (default: configs/train_config.yaml)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Paths to baseline override YAML config files",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds to use (overrides random_seeds in config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing",
    )
    parser.add_argument(
        "--baseline-filter",
        nargs="*",
        default=None,
        help="Only run baselines whose baseline_name matches these",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel workers (overrides parallel_workers in config). "
            "Each worker uses a separate CARLA port."
        ),
    )
    args = parser.parse_args()

    # Load shared base config
    base_config = load_config(args.base_config)
    print(f"Base config: {args.base_config}")

    # Load all baseline overrides, merge with base, apply filter
    configs: List[Dict[str, Any]] = []
    for path in args.configs:
        overrides = load_config(path)
        merged = merge_configs(base_config, overrides)
        baseline_name = merged.get("baseline_name", path)

        if args.baseline_filter and baseline_name not in args.baseline_filter:
            print(f"Skipping {baseline_name} (not in filter)")
            continue

        configs.append(merged)

    if not configs:
        print("No configs matched the filter. Exiting.")
        sys.exit(1)

    # Determine parallelism
    parallel_workers: int = (
        args.workers
        if args.workers is not None
        else base_config.get("parallel_workers", 1)
    )

    # Build list of all (config, seed) pairs
    all_runs: List[Tuple[Dict[str, Any], int]] = []
    for config in configs:
        seeds: List[int] = args.seeds or config.get("random_seeds", [42])
        baseline_name = config.get("baseline_name", "unknown")
        print(f"Baseline: {baseline_name} x {len(seeds)} seeds")
        for seed in seeds:
            all_runs.append((config, seed))

    total_runs = len(all_runs)
    print(f"\nTotal planned runs: {total_runs}")
    print(f"Parallel workers: {parallel_workers}")

    if args.dry_run:
        print("\n--- DRY RUN ---\n")

    if parallel_workers <= 1:
        # Sequential execution
        completed = 0
        for config, seed in all_runs:
            completed += 1
            print(f"\n[{completed}/{total_runs}]")
            run_single(config, seed, dry_run=args.dry_run)
    else:
        # Parallel execution — assign each run a CARLA port based on its
        # position in the worker pool (worker_index cycles 0..workers-1)
        worker_args: List[Tuple[Dict[str, Any], int, bool]] = []
        for i, (config, seed) in enumerate(all_runs):
            worker_index = i % parallel_workers
            port_config = _assign_carla_port(config, worker_index)
            worker_args.append((port_config, seed, args.dry_run))

        if args.dry_run:
            # Dry run: just print sequentially, no process pool needed
            for i, (config, seed, _) in enumerate(worker_args):
                print(f"\n[{i + 1}/{total_runs}]")
                run_single(config, seed, dry_run=True)
        else:
            print(
                f"\nLaunching {parallel_workers} parallel workers "
                f"(ports {base_config.get('carla_port', 2000)}"
                f"-{base_config.get('carla_port', 2000) + 3 * (parallel_workers - 1)})"
            )
            with ProcessPoolExecutor(
                max_workers=parallel_workers
            ) as executor:
                futures = {
                    executor.submit(_run_worker, wa): wa
                    for wa in worker_args
                }
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    result = future.result()
                    print(f"[{completed}/{total_runs}] {result}")

    print(f"\nAll {total_runs} runs {'planned' if args.dry_run else 'complete'}.")


if __name__ == "__main__":
    main()
