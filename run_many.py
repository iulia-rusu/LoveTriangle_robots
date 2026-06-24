from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from dqn_common import apply_overrides, parse_scalar, slugify


def parse_sweep_item(item: str) -> tuple[str, list[str]]:
    if "=" not in item:
        raise ValueError(f"Sweep must look like key=value1,value2, got: {item}")
    key, values = item.split("=", 1)
    vals = [v.strip() for v in values.split(",") if v.strip()]
    if not vals:
        raise ValueError(f"No sweep values supplied for {key}")
    for v in vals:
        parse_scalar(v)
    return key.strip(), vals


def cfg_list(cfg: dict[str, Any], key: str, default: list[Any]) -> list[Any]:
    value = cfg.get("multi_run", {}).get(key, default)
    if value is None:
        return default
    if isinstance(value, list):
        return value
    return [value]


def choose(cli_value: Any, cfg: dict[str, Any], key: str, default: Any) -> Any:
    return cli_value if cli_value is not None else cfg.get("multi_run", {}).get(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch multiple DQN training runs.")
    parser.add_argument("--launcher-config", type=str, default="config/config.yaml", help="Config file containing optional multi_run defaults.")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--out-root", type=str, default=None)

    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--start-margin", type=float, default=None)
    parser.add_argument("--random-heading", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--env-action-mode", choices=["index", "motor"], default=None)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--memory-size", type=int, default=None)
    parser.add_argument("--eps-decay", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--rollout-every", type=int, default=None)

    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Common config override passed to every run.")
    parser.add_argument("--sweep", action="append", default=[], help="Config sweep, e.g. --sweep vehicle.max_linear_speed=10,20")
    parser.add_argument("--train-script", type=str, default=None, help="Path to train.py. Defaults to train.py next to this file.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel", type=int, default=None, help="Max number of training runs to execute concurrently. Defaults to 1 (sequential).")
    parser.add_argument(
        "--multi-condition",
        action="store_true",
        help="Pass the full --conditions list to each run (so it randomizes condition per episode) instead of spawning one run per condition.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launcher_cfg = apply_overrides(load_config(args.launcher_config), args.overrides)

    this_dir = Path(__file__).resolve().parent
    train_script = Path(args.train_script) if args.train_script else this_dir / "train.py"
    out_root = Path(choose(args.out_root, launcher_cfg, "out_root", "outputs/dqn_runs"))

    configs = args.configs if args.configs is not None else cfg_list(launcher_cfg, "configs", [args.launcher_config])
    conditions = args.conditions if args.conditions is not None else cfg_list(
        launcher_cfg,
        "conditions",
        [launcher_cfg.get("simulation", {}).get("condition", "blocking")],
    )
    seeds = args.seeds if args.seeds is not None else [int(s) for s in cfg_list(launcher_cfg, "seeds", [launcher_cfg.get("seed", 0)])]

    multi_condition = bool(choose(args.multi_condition or None, launcher_cfg, "multi_condition", False))
    parallel = int(choose(args.parallel, launcher_cfg, "parallel", 1))
    # itertools.product still needs an iterable for the condition axis; in
    # multi-condition mode we collapse it to a single "all conditions" entry
    # so we get one run (per config/seed/sweep combo) instead of one per condition.
    condition_axis = [tuple(str(c) for c in conditions)] if multi_condition else conditions

    sweep_items = [parse_sweep_item(s) for s in args.sweep]
    cfg_sweeps = launcher_cfg.get("multi_run", {}).get("sweeps", {}) or {}
    for key, values in cfg_sweeps.items():
        sweep_items.append((str(key), [str(v) for v in values]))

    sweep_keys = [k for k, _ in sweep_items]
    sweep_values = [vals for _, vals in sweep_items]
    sweep_combos = list(itertools.product(*sweep_values)) if sweep_items else [()]

    commands: list[list[str]] = []
    for config, condition, seed, combo in itertools.product(configs, condition_axis, seeds, sweep_combos):
        sweep_overrides = [f"{k}={v}" for k, v in zip(sweep_keys, combo)]
        sweep_name = "_".join(slugify(x) for x in sweep_overrides) if sweep_overrides else "base"
        config_name = slugify(Path(str(config)).stem)

        is_multi = isinstance(condition, tuple)
        cond_label = "multi-" + "-".join(condition) if is_multi else str(condition)
        run_name = f"{config_name}_cond-{slugify(cond_label)}_seed-{seed}_{sweep_name}"
        out_dir = out_root / run_name

        cmd = [
            sys.executable,
            str(train_script),
            "--config", str(config),
            "--seed", str(seed),
            "--out-dir", str(out_dir),
        ]
        if is_multi:
            cmd += ["--conditions", *condition]
        else:
            cmd += ["--condition", str(condition)]

        optional_args = {
            "--episodes": choose(args.episodes, launcher_cfg, "episodes", None),
            "--lr": choose(args.lr, launcher_cfg, "lr", None),
            "--gamma": choose(args.gamma, launcher_cfg, "gamma", None),
            "--batch-size": choose(args.batch_size, launcher_cfg, "batch_size", None),
            "--memory-size": choose(args.memory_size, launcher_cfg, "memory_size", None),
            "--eps-decay": choose(args.eps_decay, launcher_cfg, "eps_decay", None),
            "--save-every": choose(args.save_every, launcher_cfg, "save_every", None),
            "--rollout-every": choose(args.rollout_every, launcher_cfg, "rollout_every", None),
            "--start-margin": choose(args.start_margin, launcher_cfg, "start_margin", None),
            "--env-action-mode": choose(args.env_action_mode, launcher_cfg, "env_action_mode", None),
        }
        for flag, value in optional_args.items():
            if value is not None:
                cmd += [flag, str(value)]

        bool_args = {
            "--cpu": choose(args.cpu, launcher_cfg, "cpu", None),
            "--random-start": choose(args.random_start, launcher_cfg, "random_start", None),
            "--random-heading": choose(args.random_heading, launcher_cfg, "random_heading", None),
        }
        for flag, value in bool_args.items():
            if value is True:
                cmd.append(flag)
            elif value is False:
                cmd.append("--no-" + flag[2:])

        # args.overrides were already applied to launcher_cfg; pass them to train too,
        # because each target config also needs the same common overrides.
        for override in list(args.overrides) + sweep_overrides:
            cmd += ["--set", override]
        commands.append(cmd)

    print(f"Prepared {len(commands)} run(s). parallel={parallel}")
    for i, cmd in enumerate(commands, start=1):
        print(f"[{i}/{len(commands)}] {' '.join(cmd)}")

    if args.dry_run:
        return

    def run_one(item: tuple[int, list[str]]) -> int:
        i, cmd = item
        print(f"\n[{i}/{len(commands)}] starting", flush=True)
        result = subprocess.run(cmd)
        print(f"[{i}/{len(commands)}] finished (exit={result.returncode})", flush=True)
        return result.returncode

    indexed = list(enumerate(commands, start=1))
    if parallel <= 1:
        for item in indexed:
            code = run_one(item)
            if code != 0:
                raise subprocess.CalledProcessError(code, item[1])
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            codes = list(pool.map(run_one, indexed))
        failed = [indexed[i][0] for i, code in enumerate(codes) if code != 0]
        if failed:
            raise RuntimeError(f"Run(s) {failed} failed (non-zero exit).")


if __name__ == "__main__":
    main()
