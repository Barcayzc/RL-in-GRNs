'''train.py
Runs a full DDQN + PER training loop on the PBN environment built by
`pbn_env.make_env()`.

Usage (default hyper‑params are paper‑like but trimmed for laptop HW):

    python train.py               # quick run
    python train.py --steps 150000 --bs 256  # closer to paper
'''

import sys, pathlib
# Get the project root (one level up from scripts/)
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pathlib import Path
import argparse
import json
import time
from collections import deque

import numpy as np
from tqdm import trange

from pbn_env import make_env             # factory from your module
from dqn.ddqn_per_agent import DDQNPERAgent  # agent we implemented

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def moving_avg(vals, k=100):
    if len(vals) < k:
        return np.mean(vals)
    return np.mean(list(vals)[-k:])

# ──────────────────────────────────────────────────────────────────────────────
# Main training procedure
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    # 1) Env & agent -----------------------------------------------------------
    env = make_env(seed=args.seed)
    state_dim  = env.observation_space.n
    action_dim = env.action_space.n

    agent = DDQNPERAgent(
        state_dim      = state_dim,
        action_dim     = action_dim,
        gamma          = 0.99,
        learning_rate  = 1e-4,        # slightly lower than paper, stabler
        epsilon        = 1.0,
        epsilon_min    = 0.05,
        epsilon_decay  = 0.99995,     # so ~exp(-0.99995*steps)
        batch_size     = args.bs,
        memory_capacity= args.buffer,
        target_update_freq = 2000,
        beta_increment = 1/args.steps
    )

    # Book‑keeping -------------------------------------------------------------
    episodic_returns   = []
    perturbations_hist = []
    ep_return          = 0.0
    ep_perturbs        = 0
    rewards_window     = deque(maxlen=100)

    # Reset once at start
    state, _ = env.reset()

    # Progress bar -------------------------------------------------------------
    pbar = trange(args.steps, dynamic_ncols=True)
    start_time = time.time()

    for global_step in pbar:
        # 2) select action & step env ----------------------------------------
        action = agent.act(state)
        next_s, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # store + train --------------------------------------------------------
        agent.store_transition(state, action, r, next_s, done)
        agent.train_step()

        # book‑keeping --------------------------------------------------------
        ep_return += r
        ep_perturbs += (action != 0)
        state = next_s

        if done:
            # episode wrap‑up
            episodic_returns.append(ep_return)
            perturbations_hist.append(ep_perturbs)
            rewards_window.append(ep_return)

            # reset
            state, _ = env.reset()
            ep_return = 0.0
            ep_perturbs = 0

        # live bar update every 1 000 steps
        if (global_step+1) % 1000 == 0:
            pbar.set_postfix({
                'len(buf)': agent.memory.size(),
                'eps':      f"{agent.epsilon:.3f}",
                'β':        f"{agent.beta:.2f}",
                'R̄100':    f"{moving_avg(rewards_window):+.2f}"
            })

    elapsed = time.time() - start_time
    print(f"Finished {args.steps} steps in {elapsed/60:.1f} min")

    # 3) save artefacts --------------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    agent.save(out_dir / 'q_network')

    log = {
        'steps':               args.steps,
        'batch_size':          args.bs,
        'buffer_cap':          args.buffer,
        'returns':             episodic_returns,
        'perturbations':       perturbations_hist,
    }
    (out_dir / 'training_log.json').write_text(json.dumps(log))
    print(f"Saved model & log to {out_dir}")

# ──────────────────────────────────────────────────────────────────────────────
# CLI entry
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDQN+PER training for PBN control")
    parser.add_argument('--steps',  type=int,   default=50_000, help='total env interactions (paper used 150k–670k)')
    parser.add_argument('--bs',     type=int,   default=128,    help='batch size')
    parser.add_argument('--buffer', type=int,   default=200_000,help='replay buffer capacity')
    parser.add_argument('--seed',   type=int,   default=0,      help='PRNG seed')
    parser.add_argument('--out',    type=str,   default='runs/run_0', help='output checkpoint dir')
    args = parser.parse_args()

    train(args)
