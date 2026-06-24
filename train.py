from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datetime import datetime
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seed
from src.environment import BraitenbergEnv

from dqn_common import (
    DQNAgent,
    append_metrics,
    apply_overrides,
    get_action_space,
    get_device,
    get_vehicle_xy,
    reset_env,
    resolve_condition,
    save_checkpoint,
    save_json,
    save_policy,
    save_rollout_plot,
    save_training_curves,
    seed_everything,
    step_env,
    to_state_tensor,
)


def cfg_value(cfg: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def choose(cli_value: Any, cfg: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return cli_value if cli_value is not None else cfg_value(cfg, section, key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN policy for BraitenbergEnv.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="If given, sample a random condition per episode from this list instead of a single fixed --condition.",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--memory-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--eps-start", type=float, default=None)
    parser.add_argument("--eps-end", type=float, default=None)
    parser.add_argument("--eps-decay", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)

    parser.add_argument(
        "--env-action-mode",
        choices=["index", "motor"],
        default=None,
        help="Use 'index' if env.step accepts a DQN action index. Use 'motor' if env.step expects [left_motor, right_motor].",
    )

    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--start-margin", type=float, default=None)
    parser.add_argument("--random-heading", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--rollout-every", type=int, default=None, help="0 disables rollout plots during training.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. --set simulation.max_steps=1000")
    return parser.parse_args()


def run_greedy_rollout(
    agent: DQNAgent,
    env: BraitenbergEnv,
    cfg: dict[str, Any],
    action_space: list[Any],
    action_mode: str,
    max_steps: int,
    device: torch.device,
) -> tuple[dict[str, list[float]], list[float]]:
    agent.policy_net.eval()
    state, _, _ = reset_env(env, cfg, random_start=False)
    state_t = to_state_tensor(state, device)

    trajectory = {"x": [], "y": [], "red_x": [], "red_y": [], "green_x": [], "green_y": []}
    rewards: list[float] = []

    for _ in range(max_steps):
        action_idx = agent.select_action(state_t, explore=False)
        observation, reward, terminated, _ = step_env(env, action_idx, action_space, action_mode)
        rewards.append(float(reward))

        x, y = get_vehicle_xy(env)
        trajectory["x"].append(x)
        trajectory["y"].append(y)
        if getattr(env, "red_pos", None) is not None:
            trajectory["red_x"].append(float(env.red_pos[0]))
            trajectory["red_y"].append(float(env.red_pos[1]))
        if getattr(env, "green_pos", None) is not None:
            trajectory["green_x"].append(float(env.green_pos[0]))
            trajectory["green_y"].append(float(env.green_pos[1]))

        if terminated:
            break
        state_t = to_state_tensor(observation, device)

    agent.policy_net.train()
    return trajectory, rewards


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    conditions_list = args.conditions if args.conditions is not None else cfg.get("training", {}).get("conditions", None)
    multi_condition = bool(conditions_list)
    if multi_condition:
        conditions_list = [str(c) for c in conditions_list]
        condition = resolve_condition(cfg, conditions_list[0])
        condition_label = "multi-" + "-".join(conditions_list)
    else:
        condition = resolve_condition(cfg, args.condition)
        condition_label = condition

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    if callable(set_seed):
        set_seed(seed)
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    use_cpu = bool(choose(args.cpu, cfg, "training", "cpu", False))
    device = get_device(prefer_cpu=use_cpu)
    
    out_dir = Path(choose(args.out_dir, cfg, "training", "out_dir", "outputs/dqn"))
    #add condiiton and timestamp
    out_dir = out_dir / str(condition_label)  / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = choose(args.run_name, cfg, "training", "run_name", None)
    if run_name:
        out_dir = out_dir / str(run_name)

    ckpt_dir = out_dir / "checkpoints"
    policy_dir = out_dir / "policies"
    figure_dir = out_dir / "figures"
    metrics_path = out_dir / "training_metrics.csv"

    batch_size = int(choose(args.batch_size, cfg, "training", "batch_size", 128))
    memory_size = int(choose(args.memory_size, cfg, "training", "memory_size", 10_000))
    gamma = float(choose(args.gamma, cfg, "training", "gamma", 0.99))
    eps_start = float(choose(args.eps_start, cfg, "training", "eps_start", 0.9))
    eps_end = float(choose(args.eps_end, cfg, "training", "eps_end", 0.01))
    eps_decay = int(choose(args.eps_decay, cfg, "training", "eps_decay", 50_000))
    tau = float(choose(args.tau, cfg, "training", "tau", 0.005))
    lr = float(choose(args.lr, cfg, "training", "lr", 3e-4))
    hidden_dim = int(choose(args.hidden_dim, cfg, "training", "hidden_dim", 128))
    action_mode = str(choose(args.env_action_mode, cfg, "training", "env_action_mode", "index"))

    random_start = bool(choose(args.random_start, cfg, "training", "random_start", False))
    start_margin = float(choose(args.start_margin, cfg, "training", "start_margin", 0.0))
    random_heading = bool(choose(args.random_heading, cfg, "training", "random_heading", True))

    save_every = int(choose(args.save_every, cfg, "training", "save_every", 100))
    rollout_every = int(choose(args.rollout_every, cfg, "training", "rollout_every", 0))

    env = BraitenbergEnv(cfg, condition=condition)
    initial_state, _, _ = reset_env(
        env,
        cfg,
        rng=rng,
        random_start=random_start,
        start_margin=start_margin,
        random_heading=random_heading,
    )
    action_space = get_action_space(env)
    n_observations = len(initial_state)
    n_actions = len(action_space)

    agent = DQNAgent(
        n_observations=n_observations,
        n_actions=n_actions,
        device=device,
        lr=lr,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        memory_size=memory_size,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        hidden_dim=hidden_dim,
    )

    num_episodes = args.episodes
    if num_episodes is None:
        cfg_episodes = cfg_value(cfg, "training", "episodes", None)
        if cfg_episodes is not None:
            num_episodes = int(cfg_episodes)
        else:
            num_episodes = 10_000 if device.type in {"cuda", "mps"} else 50

    max_steps = int(cfg["simulation"]["max_steps"])
    episode_rewards: list[float] = []
    episode_durations: list[int] = []
    best_reward = -float("inf")

    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_training_cfg = {
        "seed": seed,
        "device": str(device),
        "condition": condition_label,
        "multi_condition": multi_condition,
        "conditions": conditions_list,
        "episodes": num_episodes,
        "batch_size": batch_size,
        "memory_size": memory_size,
        "gamma": gamma,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_decay": eps_decay,
        "tau": tau,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "env_action_mode": action_mode,
        "random_start": random_start,
        "start_margin": start_margin,
        "random_heading": random_heading,
        "save_every": save_every,
        "rollout_every": rollout_every,
        "n_observations": n_observations,
        "n_actions": n_actions,
        "action_space": action_space,
    }
    save_json(
        out_dir / "run_config.json",
        {
            "cli_args": vars(args),
            "resolved_training": resolved_training_cfg,
            "config_after_overrides": cfg,
        },
    )

    print(
        f"Training on {device}; episodes={num_episodes}; condition={condition_label}; "
        f"random_start={random_start}; random_heading={random_heading}; action_mode={action_mode}; "
        f"n_actions={n_actions}",
        flush=True,
    )

    for episode in range(1, int(num_episodes) + 1):
        if multi_condition:
            episode_condition = str(rng.choice(conditions_list))
            env.condition = episode_condition
        else:
            episode_condition = condition

        state, _, start_pose = reset_env(
            env,
            cfg,
            rng=rng,
            random_start=random_start,
            start_margin=start_margin,
            random_heading=random_heading,
            
        )
        state_t = to_state_tensor(state, device)
        total_reward = 0.0
        last_loss = None
        terminated = False

        for t in range(max_steps):
            action_idx = agent.select_action(state_t, explore=True)
            observation, reward, terminated, _ = step_env(env, action_idx, action_space, action_mode)
            total_reward += float(reward)

            reward_t = torch.tensor([float(reward)], device=device)
            next_state_t = None if terminated else to_state_tensor(observation, device)
            agent.memory.push(state_t, action_idx, next_state_t, reward_t)
            state_t = next_state_t

            loss = agent.optimise_model()
            if loss is not None:
                last_loss = loss

            if terminated:
                break

        duration = t + 1
        episode_rewards.append(total_reward)
        episode_durations.append(duration)

        if total_reward > best_reward:
            best_reward = total_reward
            save_policy(policy_dir / "best_policy.pt", agent, cfg, action_space, action_mode)

        append_metrics(
            metrics_path,
            {
                "episode": episode,
                "condition": episode_condition,
                "reward": total_reward,
                "duration": duration,
                "epsilon": agent.epsilon(),
                "loss": "" if last_loss is None else last_loss,
                "best_reward": best_reward,
                "terminated": terminated,
                "done_reason": getattr(getattr(env, "last_info", None), "done_reason", None),
                "start_x": start_pose.x,
                "start_y": start_pose.y,
                "start_heading": start_pose.heading,
            },
        )

        if save_every > 0 and episode % save_every == 0:
            save_checkpoint(
                ckpt_dir / f"checkpoint_ep{episode:06d}.pt",
                agent,
                episode,
                episode_rewards,
                episode_durations,
                cfg,
                action_space,
                best_reward,
                action_mode,
            )
            save_training_curves(figure_dir, episode_rewards, episode_durations)

        if rollout_every > 0 and episode % rollout_every == 0:
            trajectory, rollout_rewards = run_greedy_rollout(
                agent, env, cfg, action_space, action_mode, max_steps, device
            )
            save_rollout_plot(
                figure_dir / f"rollout_ep{episode:06d}.png",
                trajectory,
                rollout_rewards,
                cfg,
                title=f"Episode {episode} rollout",
            )

        if episode == 1 or episode % 10 == 0:
            print(
                f"episode={episode:05d} reward={total_reward:9.3f} duration={duration:5d} "
                f"eps={agent.epsilon():.4f} best={best_reward:9.3f}",
                flush=True,
            )

    save_policy(policy_dir / "final_policy.pt", agent, cfg, action_space, action_mode)
    save_checkpoint(
        ckpt_dir / "final_model.pt",
        agent,
        int(num_episodes),
        episode_rewards,
        episode_durations,
        cfg,
        action_space,
        best_reward,
        action_mode,
    )
    save_training_curves(figure_dir, episode_rewards, episode_durations)
    print(f"Done. Outputs saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
