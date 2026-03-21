#!/usr/bin/env python3
"""
This script loads a trained DQN model and plays the game with visualization.
"""

import argparse
import os
import time


import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder


def create_play_environment(render_mode: str = "human", record_video: bool = False):
    """
    Create the Atari environment for playing with rendering enabled.
    
    Args:
        render_mode: "human" for GUI display, "rgb_array" for video recording
        record_video: Whether to record video of gameplay
    """
    print(f"🎮 Creating JourneyEscape environment for playing...")

    # Create environment with Atari wrappers
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    
    def make_env():
        env = gym.make("ALE/JourneyEscape-v5", render_mode=render_mode)
        env = AtariWrapper(env, frame_skip=4, screen_size=84, 
                          terminal_on_life_loss=False, clip_reward=False)
        return env
    
    env = DummyVecEnv([make_env])

    # Add frame stacking
    env = VecFrameStack(env, n_stack=4)

    # Add video recording if requested
    if record_video:
        video_folder = "videos"
        os.makedirs(video_folder, exist_ok=True)
        env = VecVideoRecorder(
            env,
            video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=10000,
            name_prefix="journeyescape-gameplay",
        )
        print(f"📹 Video recording enabled - saving to {video_folder}/")

    print(f"✅ Environment created successfully!")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    return env


def play_episode(
    model: DQN,
    env,
    episode_num: int,
    render: bool = True,
    delay: float = 0.0,
    verbose: bool = True,
):
    """
    Play a single episode with the trained agent.
    
    Args:
        model: Trained DQN model
        env: Gymnasium environment
        episode_num: Episode number for display
        render: Whether to render the game
        delay: Delay between steps (seconds) for better visualization
        verbose: Print detailed step information
    
    Returns:
        dict: Episode statistics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎮 Episode {episode_num}")
        print(f"{'='*60}")

    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False
    step_count = 0

    start_time = time.time()

    while not done:
        # Use greedy policy (deterministic=True) for best performance
        action, _states = model.predict(obs, deterministic=True)

        # Take action in environment
        obs, reward, done, info = env.step(action)

        # Handle vectorized environment returns
        if isinstance(reward, (list, tuple, np.ndarray)):
            step_reward = reward[0]
            done_val = done[0] if isinstance(done, (list, tuple, np.ndarray)) else done
            truncated = (
                info[0].get("TimeLimit.truncated", False)
                if info and len(info) > 0
                else False
            )
        else:
            step_reward = reward
            done_val = done
            truncated = info.get("TimeLimit.truncated", False) if info else False

        episode_reward += step_reward
        episode_length += 1
        step_count += 1

        # Verbose step logging
        if verbose and step_count % 100 == 0:
            print(
                f"   Step {step_count}: Reward = {episode_reward:.0f}, Action = {action[0] if isinstance(action, np.ndarray) else action}"
            )

        # Add delay for better visualization
        if delay > 0 and render:
            time.sleep(delay)

        done = done_val or truncated

    episode_time = time.time() - start_time

    # Episode summary
    stats = {
        "episode": episode_num,
        "reward": episode_reward,
        "length": episode_length,
        "time": episode_time,
    }

    print(f"\n📊 Episode {episode_num} Summary:")
    print(f"   Total Reward: {episode_reward:.0f}")
    print(f"   Episode Length: {episode_length} steps")
    print(f"   Time: {episode_time:.2f} seconds")
    print(f"   Average Reward per Step: {episode_reward/episode_length:.2f}")

    return stats


def play_multiple_episodes(
    model_path: str,
    num_episodes: int = 5,
    render: bool = True,
    delay: float = 0.0,
    record_video: bool = False,
    verbose: bool = True,
):
    """
    Play multiple episodes with the trained agent.
    
    Args:
        model_path: Path to the saved model (.zip file)
        num_episodes: Number of episodes to play
        render: Whether to render the game
        delay: Delay between steps for visualization
        record_video: Whether to record gameplay video
        verbose: Print detailed information
    """
    print("\n" + "="*80)
    print("🎮 JOURNEYESCAPE DQN AGENT GAMEPLAY")
    print("="*80)
    print(f"\n📁 Loading model from: {model_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load the trained model
    try:
        model = DQN.load(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Policy: {model.policy.__class__.__name__}")
        print(f"   Device: {model.device}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Create environment
    render_mode = "human" if render else "rgb_array"
    env = create_play_environment(render_mode=render_mode, record_video=record_video)

    # Play episodes
    all_stats = []

    try:
        for episode in range(1, num_episodes + 1):
            stats = play_episode(
                model=model,
                env=env,
                episode_num=episode,
                render=render,
                delay=delay,
                verbose=verbose,
            )
            all_stats.append(stats)

            # Small delay between episodes
            if episode < num_episodes:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n⚠️  Gameplay interrupted by user")

    finally:
        env.close()
        print("\n🔒 Environment closed")

    # Print overall statistics
    if all_stats:
        print("\n" + "="*80)
        print("📊 OVERALL STATISTICS")
        print("="*80)

        rewards = [s["reward"] for s in all_stats]
        lengths = [s["length"] for s in all_stats]

        print(f"\nTotal Episodes Played: {len(all_stats)}")
        print(f"\nRewards:")
        print(f"   Average: {np.mean(rewards):.2f}")
        print(f"   Std Dev: {np.std(rewards):.2f}")
        print(f"   Min: {np.min(rewards):.2f}")
        print(f"   Max: {np.max(rewards):.2f}")
        print(f"\nEpisode Lengths:")
        print(f"   Average: {np.mean(lengths):.0f} steps")
        print(f"   Min: {np.min(lengths):.0f} steps")
        print(f"   Max: {np.max(lengths):.0f} steps")

        print("\n" + "="*80)


def main():
    """Main function to parse arguments and play the game."""
    parser = argparse.ArgumentParser(
        description="Play JourneyEscape with trained DQN agent"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/dqn_model.zip",
        help="Path to the trained model (.zip file)",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to play"
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Disable rendering (faster but no visualization)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between steps in seconds (for slower visualization)",
    )
    parser.add_argument(
        "--record_video", action="store_true", help="Record gameplay video"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Reduce verbosity (less output)"
    )

    args = parser.parse_args()

    # Play the game
    try:
        play_multiple_episodes(
            model_path=args.model_path,
            num_episodes=args.episodes,
            render=not args.no_render,
            delay=args.delay,
            record_video=args.record_video,
            verbose=not args.quiet,
        )
        print("\n✅ Gameplay completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error during gameplay: {e}\n")
        raise


if __name__ == "__main__":
    main()
