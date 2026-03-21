#!/usr/bin/env python3
"""
DQN Training Script for JourneyEscape Atari Environment
======================================================

This script trains a Deep Q-Network (DQN) agent to play JourneyEscape using Stable Baselines3.
It supports both MLPPolicy and CNNPolicy, extensive hyperparameter tuning, and comprehensive logging.

Usage:
    python train.py --policy CNN --experiment_name exp1
    python train.py --policy MLP --learning_rate 0.001 --gamma 0.95
"""
from datetime import datetime
from typing import Any, Dict, Optional

# Import and register ALE environments
import argparse
import os
import json
import csv
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TransformObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

# Try to register ALE envs if needed (best-effort)
try:
    # gymnasium provides Atari via gymnasium.make with ALE; no explicit register normally required
    
    pass
except Exception as e:
    print(f"Warning: Could not register ALE environments: {e}")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Utility functions
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class HyperparameterExperiment: # Logs hyperparameter results to CSV
    def __init__(self, csv_path: str = "hyperparameter_results.csv"):
        self.csv_path = csv_path
        self.header = [
            "learning_rate",
            "gamma",
            "batch_size",
            "epsilon_start",
            "epsilon_end",
            "exploration_fraction",
            "avg_reward",
            "std_reward",
            "min_reward",
            "max_reward",
            "avg_episode_length",
            "total_eval_episodes",
            "training_time_seconds",
            "total_timesteps",
            "policy_type",
            "hyperparameter_set_name",
            "experiment_name",
            "timestamp",
        ]
        # create file with header if missing
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def log_experiment(self, hyperparams: Dict[str, Any], results: Dict[str, Any], policy_type: str, hyperparameter_set_name: str, experiment_name: str):
        row = [
            hyperparams.get("learning_rate", ""),
            hyperparams.get("gamma", ""),
            hyperparams.get("batch_size", ""),
            hyperparams.get("exploration_initial_eps", "") or hyperparams.get("epsilon_start", ""),
            hyperparams.get("exploration_final_eps", "") or hyperparams.get("epsilon_end", ""),
            hyperparams.get("exploration_fraction", ""),
            results.get("avg_reward", ""),
            results.get("std_reward", ""),
            results.get("min_reward", ""),
            results.get("max_reward", ""),
            results.get("avg_episode_length", ""),
            results.get("total_eval_episodes", ""),
            results.get("training_time_seconds", ""),
            results.get("total_timesteps", ""),
            policy_type,
            hyperparameter_set_name,
            experiment_name,
            datetime.utcnow().isoformat(),
        ]
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


class TrainingLogger: # Saves training summaries to JSON
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        ensure_dir(self.log_dir)

    def save_summary(self, path: str, data: Dict[str, Any]):
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def create_environment(env_name: str = "ALE/JourneyEscape-v5", n_envs: int = 1, seed: int = SEED):
    """
    Create and wrap the Atari environment with proper ALE registration.
    Returns a VecFrameStack-wrapped vectorized environment and the resolved env id.
    """
    env_alternatives = [
        env_name,
        env_name.replace("ALE/", ""),
        env_name.replace("-v5", "-v4"),
        env_name.replace("-v5", ""),
    ]

    last_exc = None
    for alt in dict.fromkeys(env_alternatives):
        try:
            print(f"Creating env: {alt} (n_envs={n_envs})")
            vec_env = make_atari_env(alt, n_envs=n_envs, seed=seed)
            vec_env = VecFrameStack(vec_env, n_stack=4)
            return vec_env, alt
        except Exception as e:
            last_exc = e
            print(f"Failed to create {alt}: {e}")
    raise RuntimeError(f"Could not create any env variant. Last error: {last_exc}")


def create_dqn_agent(
    env, policy_type: str, hyperparams: Dict[str, Any], tensorboard_log: Optional[str] = None
):
    """
    Instantiate a stable-baselines3 DQN model with given hyperparameters.
    policy_type: 'CNN' or 'MLP' -> mapped to 'CnnPolicy' or 'MlpPolicy'
    hyperparams keys used: learning_rate, gamma, batch_size, exploration_fraction,
                           exploration_initial_eps, exploration_final_eps, buffer_size, learning_starts
    """
    policy = "CnnPolicy" if policy_type.lower().startswith("c") else "MlpPolicy"

    model_kwargs = dict(
        policy=policy,
        env=env,
        learning_rate=hyperparams.get("learning_rate", 1e-4),
        gamma=hyperparams.get("gamma", 0.99),
        batch_size=int(hyperparams.get("batch_size", 32)),
        buffer_size=int(hyperparams.get("buffer_size", 100000)),
        learning_starts=int(hyperparams.get("learning_starts", 10000)),
        exploration_fraction=float(hyperparams.get("exploration_fraction", 0.1)),
        exploration_initial_eps=float(hyperparams.get("exploration_initial_eps", 1.0)),
        exploration_final_eps=float(hyperparams.get("exploration_final_eps", 0.05)),
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=SEED,
    )

    model = DQN(**model_kwargs)
    return model
def to_numpy(obs):
    # For LazyFrames, convert to numpy array safely for SB3 predictions
    try:
        return np.array(obs)
    except Exception:
        return obs

def train_agent(
    env_id: str,
    policy_type: str,
    hyperparams: Dict[str, Any],
    total_timesteps: int,
    experiment_name: str,
    save_dir: str = "models",
    tensorboard_dir: str = "logs/tensorboard",
    eval_episodes: int = 10,
):
    """
    Train a DQN agent for a given Atari environment, using a specified policy type and hyperparameters.

    Parameters:
    env_id (str): Atari environment ID.
    policy_type (str): 'CNN' or 'MLP' -> mapped to 'CnnPolicy' or 'MlpPolicy'.
    hyperparams (Dict[str, Any]): Hyperparameter dictionary with keys learning_rate, gamma, batch_size, exploration_fraction,
        exploration_initial_eps, exploration_final_eps, buffer_size, learning_starts, n_envs, eval_freq, checkpoint_freq.
    total_timesteps (int): Total training timesteps.
    experiment_name (str): Experiment name (used for saving models and logging).
    save_dir (str, optional): Directory for saving models (default: 'models').
    tensorboard_dir (str, optional): Directory for saving tensorboard logs (default: 'logs/tensorboard').
    eval_episodes (int, optional): Number of evaluation episodes (default: 10).

    Returns:
        model (DQN): Trained DQN agent.
        results (Dict[str, Any]): Dictionary containing evaluation results and experiment metadata.
    """
    ensure_dir(save_dir)
    ensure_dir(tensorboard_dir)

    train_env, used_env_id = create_environment(env_id, n_envs=int(hyperparams.get("n_envs", 8)))
    eval_env = make_atari_env(used_env_id, n_envs=1, seed=SEED)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    tb_log_path = os.path.join(tensorboard_dir, experiment_name)
    model = create_dqn_agent(train_env, policy_type, hyperparams, tensorboard_log=tensorboard_dir)

    # callbacks
    ckpt_dir = os.path.join(save_dir, experiment_name, "checkpoints")
    ensure_dir(ckpt_dir)
    checkpoint_callback = CheckpointCallback(save_freq=int(hyperparams.get("checkpoint_freq", 100000)), save_path=ckpt_dir, name_prefix="dqn_ckpt")
    eval_save_dir = os.path.join(save_dir, experiment_name, "eval")
    ensure_dir(eval_save_dir)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_save_dir,
        log_path=eval_save_dir,
        eval_freq=int(hyperparams.get("eval_freq", 50000)),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    print(f"Starting training: env={used_env_id}, policy={policy_type}, timesteps={total_timesteps}, experiment={experiment_name}")
    start = datetime.utcnow()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=experiment_name)
    duration = (datetime.utcnow() - start).total_seconds()

    # primary model save
    final_model_path = os.path.join(save_dir, "dqn_model.zip")
    # also save per-experiment copy
    exp_model_path = os.path.join(save_dir, f"{experiment_name}_dqn_model.zip")
    model.save(final_model_path)
    model.save(exp_model_path)
    print(f"Saved final model to: {final_model_path} and {exp_model_path}")

    eval_single = create_eval_env(used_env_id)
    # evaluate_policy returns (mean, std) for SB3's evaluate_policy
    avg_reward, std_reward = evaluate_policy(model, eval_single, n_eval_episodes=eval_episodes, deterministic=True, return_episode_rewards=False)
    # manually collect per-episode metrics for min/max and lengths
    episode_rewards = []
    episode_lengths = []
    for _ in range(eval_episodes):
        obs, _ = eval_single.reset()
        obs = to_numpy(obs)
        done = False
        ep_r = 0.0
        ep_len = 0
        while True:
            action, _ = model.predict(to_numpy(obs), deterministic=True)
            obs, reward, terminated, truncated, info = eval_single.step(int(action))
            ep_r += float(reward)
            ep_len += 1
            if terminated or truncated:
                break
        episode_rewards.append(ep_r)
        episode_lengths.append(ep_len)

    eval_single.close()

    results = {
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "total_eval_episodes": int(eval_episodes),
        "training_time_seconds": float(duration),
        "total_timesteps": int(total_timesteps),
    }

    # append to hyperparameter_results.csv
    csv_path = "hyperparameter_results.csv"
    header = [
        "learning_rate",
        "gamma",
        "batch_size",
        "epsilon_start",
        "epsilon_end",
        "exploration_fraction",
        "avg_reward",
        "std_reward",
        "min_reward",
        "max_reward",
        "avg_episode_length",
        "total_eval_episodes",
        "training_time_seconds",
        "total_timesteps",
        "policy_type",
        "hyperparameter_set_name",
        "experiment_name",
        "timestamp",
    ]
    row = [
        hyperparams.get("learning_rate", ""),
        hyperparams.get("gamma", ""),
        hyperparams.get("batch_size", ""),
        hyperparams.get("exploration_initial_eps", ""),
        hyperparams.get("exploration_final_eps", ""),
        hyperparams.get("exploration_fraction", ""),
        results["avg_reward"],
        results["std_reward"],
        results["min_reward"],
        results["max_reward"],
        results["avg_episode_length"],
        results["total_eval_episodes"],
        results["training_time_seconds"],
        results["total_timesteps"],
        policy_type,
        hyperparams.get("hyperparameter_set_name", "manual"),
        experiment_name,
        datetime.utcnow().isoformat(),
    ]

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"Appended results to {csv_path}")
    return model, results


def create_eval_env(env_id: str = "ALE/JourneyEscape-v5", seed: int = SEED):
    """
    Create a single-environment Monitor-wrapped environment for evaluation.
    Applies the SAME preprocessing as the training environment.
    """
    # Try without ALE wrapper fallback
    try:
        env = gym.make(env_id)
    except Exception:
        env = gym.make(env_id.replace("ALE/", ""))


    # 1. Resize the frame to 84×84 (standard for DQN)
    env = ResizeObservation(env, (84, 84))

    # 2. Convert to grayscale
    env = GrayScaleObservation(env)

    # 3. Stack 4 frames (equivalent to your Atari FrameStackingWrapper)
    env = FrameStack(env, num_stack=4)
    # Convert LazyFrames -> numpy arrays so SB3's predict doesn't choke on LazyFrames
    env = TransformObservation(env, to_numpy)

    # 4. Record stats (episode rewards, lengths)
    env = Monitor(env)

    # SEED for reproducibility
    env.reset(seed=seed)

    return env


def evaluate_agent(agent, env, n_eval_episodes: int = 10):
    """
    Evaluate the agent on the environment.

    Args:
        agent (stable_baselines3.BaseRLModel): The agent to evaluate.
        env (gym.Env): The environment to evaluate on.
        n_eval_episodes (int, optional): The number of episodes to evaluate on. Defaults to 10.

    Returns:
        dict: A dict containing the average reward and standard deviation of the rewards over the evaluation episodes.
    """
    avg_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    return {"avg_reward": float(avg_reward), "std_reward": float(std_reward)}


def get_predefined_hyperparameter_sets():
    """
    Returns a dict of named hyperparameter sets to run/record.
    Each group member should run 10 different combinations (as per assignment).
    """
    return {
        "baseline": {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "buffer_size": 100000,
            "learning_starts": 50000,
            "n_envs": 8,
            "eval_freq": 50000,
            "checkpoint_freq": 100000,
            "hyperparameter_set_name": "baseline",
        },
        "high_lr": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "buffer_size": 100000,
            "learning_starts": 50000,
            "n_envs": 8,
            "eval_freq": 50000,
            "checkpoint_freq": 100000,
            "hyperparameter_set_name": "high_lr",
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train a DQN Atari agent (Stable-Baselines3).")
    p.add_argument("--env", type=str, default="ALE/JourneyEscape-v5", help="Gymnasium Atari env id")
    p.add_argument("--policy", type=str, default="CNN", choices=["CNN", "MLP"], help="Policy type")
    p.add_argument("--total_timesteps", type=int, default=100000, help="Total training timesteps")
    p.add_argument("--experiment_name", type=str, default=f"exp_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}", help="Experiment name")
    p.add_argument("--hyperparameter_set", type=str, default="baseline", help="Named hyperparameter set to use")
    # allow overriding common hyperparameters
    p.add_argument("--learning_rate", type=float, help="learning rate override")
    p.add_argument("--gamma", type=float, help="discount factor override")
    p.add_argument("--batch_size", type=int, help="batch size override")
    p.add_argument("--exploration_initial_eps", type=float, help="epsilon start")
    p.add_argument("--exploration_final_eps", type=float, help="epsilon end")
    p.add_argument("--exploration_fraction", type=float, help="exploration fraction")
    p.add_argument("--n_envs", type=int, help="number of parallel envs for training")
    p.add_argument("--eval_episodes", type=int, default=10, help="evaluation episodes")
    p.add_argument("--save_dir", type=str, default="models", help="directory to save models")
    return p.parse_args()


def main():
    """
    Main entry point for training a DQN Atari agent.

    This function parses CLI arguments, loads predefined hyperparameter sets,
    applies CLI overrides, and calls train_agent() with the constructed
    hyperparameter set.

    It also saves a small JSON summary for the experiment.
    """
    args = parse_args()
    predefined = get_predefined_hyperparameter_sets()
    hyperparams = predefined.get(args.hyperparameter_set, {}).copy()
    # apply CLI overrides
    for k in ["learning_rate", "gamma", "batch_size", "exploration_initial_eps", "exploration_final_eps", "exploration_fraction", "n_envs"]:
        v = getattr(args, k, None)
        if v is not None:
            hyperparams[k] = v

    hyperparams.setdefault("hyperparameter_set_name", args.hyperparameter_set)

    ensure_dir(args.save_dir)
    ensure_dir("logs")
    ensure_dir("logs/tensorboard")

    model, results = train_agent(
        env_id=args.env,
        policy_type=args.policy,
        hyperparams=hyperparams,
        total_timesteps=args.total_timesteps,
        experiment_name=args.experiment_name,
        save_dir=args.save_dir,
        tensorboard_dir="logs/tensorboard",
        eval_episodes=args.eval_episodes,
    )

    # Save a small JSON summary for the experiment
    summary = {
        "experiment_name": args.experiment_name,
        "env": args.env,
        "policy": args.policy,
        "hyperparameters": hyperparams,
        "results": results,
        "timestamp": datetime.utcnow().isoformat(),
    }
    summary_path = os.path.join("models", f"{args.experiment_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote experiment summary to {summary_path}")


if __name__ == "__main__":
    main()