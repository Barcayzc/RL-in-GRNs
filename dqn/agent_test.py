# scripts/smoke_test_agent.py

import sys, os

# Insert the project root into sys.path so that `pbn_env` can be found
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if root not in sys.path:
    sys.path.insert(0, root)

import numpy as np
from ddqn_per_agent import DDQNPERAgent
from pbn_env import make_env


def smoke_test():
    # 1) Build the env
    env = make_env(seed=0)

    # 2) Inspect spaces
    state_dim  = env.observation_space.n  # e.g. 100
    action_dim = env.action_space.n       # e.g. 101
    print(f"State dim = {state_dim}, Action dim = {action_dim}")

    # 3) Build the agent
    agent = DDQNPERAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=32,
        memory_capacity=1000,
        target_update_freq=50,
    )

    # 4) Warm up: one env reset
    state, _ = env.reset()
    print("Initial state (first 10 bits):", state[:10].astype(int))

    # 5) One episode of random+train steps
    for step in range(20):
        # a) Action (epsilon-greedy)
        action = agent.act(state)

        # b) Env step
        next_state, reward, terminated, truncated, info = env.step(action)

        # c) Store and train
        agent.store_transition(state, action, reward, next_state, terminated)
        agent.train_step()

        # d) Next state (with auto-reset on done)
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

        # e) Print a line or two
        print(f"Step {step:2d}: a={action:3d}, r={reward:+.1f}, eps={agent.epsilon:.3f}")

    # 6) Final sanity checks
    print("Final epsilon:", agent.epsilon)
    print("Replay buffer size:", agent.memory.size())

    print("✅ Smoke test completed without errors.")


if __name__ == "__main__":
    smoke_test()
