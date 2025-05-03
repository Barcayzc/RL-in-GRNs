"""
A minimal end-to-end test that

1.  Instantiates a fresh PBN Gym environment
2.  Builds DDQN + PER agent
3.  Runs a short training loop (a few thousand steps)
4.  Prints running statistics and a quick evaluation score
"""

import sys, pathlib
# Get the project root (one level up from scripts/)
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# import sys, pathlib
# PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
# sys.path.append(str(PROJECT_ROOT))

from pbn_env import make_env
from dqn import DDQNPERAgent

import numpy as np
import tqdm 
from collections import deque

# ────────────────────────────────────────────────────────────────────────────
# Local imports
# ────────────────────────────────────────────────────────────────────────────
# from pbn_env import make_env
# from ddqn_per_agent import DDQNPERAgent


# ╭──────────────────────────────────────────────────────────────────────────╮
# 1) Hyper-parameters for this *quick* smoke-train
# ╰──────────────────────────────────────────────────────────────────────────╯
SEED              = 0
TOTAL_STEPS       = 20_000            # tiny compared to full paper (100k+)
EVAL_EPISODES     = 50
PRINT_EVERY       = 1_000
TARGET_UPDATE_EVERY = 1_000


# ╭──────────────────────────────────────────────────────────────────────────╮
# 2) Environment & agent
# ╰──────────────────────────────────────────────────────────────────────────╯
env          = make_env(seed=SEED)
state_dim    = env.observation_space.n
action_dim   = env.action_space.n

agent = DDQNPERAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    gamma=0.99,
    learning_rate=1e-3,
    epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
    batch_size=64,
    memory_capacity=100_000,
    target_update_freq=TARGET_UPDATE_EVERY,
)

# ╭──────────────────────────────────────────────────────────────────────────╮
# 3) Online training loop
# ╰──────────────────────────────────────────────────────────────────────────╯
state, _ = env.reset(seed=SEED)
episode_return, episode_len = 0.0, 0
returns_window = deque(maxlen=10)     # rolling average

for global_step in tqdm.tqdm(range(1, TOTAL_STEPS + 1)):
    # ───── act ────────────────────────────────────────────────────────────
    action = agent.act(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # ───── store + learn ──────────────────────────────────────────────────
    agent.store_transition(state, action, reward, next_state, done)
    agent.train_step()

    # ───── bookkeeping per step ───────────────────────────────────────────
    episode_return += reward
    episode_len    += 1
    state = next_state

    if done:
        returns_window.append(episode_return)
        state, _      = env.reset()
        episode_return, episode_len = 0.0, 0

    # ───── progress printout ──────────────────────────────────────────────
    if global_step % PRINT_EVERY == 0:
        avg_return = np.mean(returns_window) if returns_window else 0.0
        print(
            f"Step {global_step:,} | "
            f"avg 10 return = {avg_return:6.2f} | "
            f"ε = {agent.epsilon:5.3f} | "
            f"β = {agent.beta:5.3f} | "
            f"buffer = {agent.memory.size():,}"
        )


# ╭──────────────────────────────────────────────────────────────────────────╮
# 4) Quick evaluation pass *without* exploration
# ╰──────────────────────────────────────────────────────────────────────────╯
def evaluate(agent, env, n_episodes=50):
    """run greedy policy and report mean return & success-rate."""
    greedy_eps = agent.epsilon
    agent.epsilon = 0.0               # purely greedy
    total_ret, successes = 0.0, 0

    for ep in range(n_episodes):
        s, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            a = agent.act(s)
            s, r, term, trunc, _ = env.step(a)
            done   = term or trunc
            ep_ret += r
        total_ret += ep_ret
        successes += (ep_ret >= env.successful_reward)   # crude success proxy

    agent.epsilon = greedy_eps
    return total_ret / n_episodes, successes / n_episodes


mean_ret, success_rate = evaluate(agent, env, EVAL_EPISODES)
print("\n──────────────────────────────────────────")
print(f"Eval over {EVAL_EPISODES} episodes:")
print(f"  • mean return      : {mean_ret:7.2f}")
print(f"  • success-rate     : {success_rate*100:5.1f} %")
print("──────────────────────────────────────────\n")
