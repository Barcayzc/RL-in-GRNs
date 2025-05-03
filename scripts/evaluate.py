#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────────────
# evaluate.py
#
# Evaluate a *trained* DDQN-PER agent on a Gym-PBN environment.
# Produces:
#   • success-rate
#   • mean interventions / successful episode
#   • empirical SSD improvement      (random policy  ➜  trained policy)
#   • plots: episode–return histogram + SSD (top-20 states) bar-chart
#   • JSON with all numeric metrics
# Everything is written to      evals/eval_<timestamp>/ …
# ────────────────────────────────────────────────────────────────────────────

import sys
import os
import math
import importlib
from pathlib import Path
from collections import Counter
import warnings
import json
import datetime
import argparse

# insert project root on sys.path so we can import our packages
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if root not in sys.path:
    sys.path.insert(0, root)

# now import our env builder and QNetwork class
from pbn_env import make_env
from dqn.q_network import QNetwork
from dqn.ddqn_per_agent import DDQNPERAgent 

# other imports…
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ────────────────────────────────────────────────────────────────────────────
# Helper utils
# ────────────────────────────────────────────────────────────────────────────


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def greedy_action(q_net: tf.keras.Model, obs: np.ndarray) -> int:
    """Return arg-max Q-value action for *single* observation."""
    q_vals = q_net(obs[np.newaxis, ...], training=False).numpy()[0]
    return int(np.argmax(q_vals))


def log_json(obj: dict, path: Path):
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def plot_episode_return_hist(returns, save_path: Path):
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))
    sns.histplot(returns, bins=30, kde=False)
    plt.xlabel("Episode return")
    plt.ylabel("# Episodes")
    plt.title("Reward distribution (DDQN-PER agent)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _state_id_from_info(info: dict, obs: np.ndarray) -> int:
    """
    Prefer the fast `observation_idx` exposed by gym-PBN.
    Fallback: convert bool vector to int.
    """
    if "observation_idx" in info:
        return int(info["observation_idx"])
    # slower, but works for any binary observation
    return int("".join("1" if b else "0" for b in obs.astype(int)), 2)


def plot_ssd_bar(baseline_ctr: Counter, ctrl_ctr: Counter, out: Path,
                 top_k: int = 20):
    """
    Bar-chart of top-k (by controlled frequency) states comparing
    baseline vs controlled.
    """
    # normalise -> probabilities
    baseline_total = sum(baseline_ctr.values()) or 1
    ctrl_total     = sum(ctrl_ctr.values())     or 1
    # pick states with highest *controlled* probability
    top_states = [s for s, _ in ctrl_ctr.most_common(top_k)]
    probs_base = [baseline_ctr[s] / baseline_total for s in top_states]
    probs_ctrl = [ctrl_ctr[s]    / ctrl_total     for s in top_states]

    idx = np.arange(len(top_states))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(idx - width/2, probs_base, width, label="Random policy")
    plt.bar(idx + width/2, probs_ctrl, width, label="DDQN-PER")
    plt.xticks(idx, [f"{s}" for s in top_states], rotation=60, ha="right")
    plt.ylabel("Probability")
    plt.title("Steady–state distribution (top-{} states)".format(top_k))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ───────────────────────── SSD line-plot helper ──────────────────────────
def plot_ssd_line(baseline_ctr: Counter,
                ctrl_ctr: Counter,
                out: Path):
    """
    Line plot of SSD for *all* states (0 … 2^N - 1)
    """
    baseline_total = sum(baseline_ctr.values()) or 1
    ctrl_total = sum(ctrl_ctr.values())     or 1

    # include every state observed in either run
    all_states = sorted(set(baseline_ctr) | set(ctrl_ctr))
    probs_base = [baseline_ctr[s] / baseline_total for s in all_states]
    probs_ctrl = [ctrl_ctr[s] / ctrl_total for s in all_states]

    plt.figure(figsize=(8, 5))
    plt.plot(all_states, probs_base, color="red",  lw=1.2, label="Uncontrolled PBN")
    plt.plot(all_states, probs_ctrl, color="green",lw=1.2, label="Controlled PBN via DDQN")
    plt.xlabel("State")
    plt.ylabel("Probability")
    plt.title("Steady-state Distribution for PBN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# Main evaluation routine
# ────────────────────────────────────────────────────────────────────────────
def evaluate(model_path: Path,
             n_episodes: int = 1_000,
             out_root: Path = Path("evals")) -> None:

    # -------------------- output dir ----------------------------------------
    out_dir = out_root / f"eval_{timestamp()}"
    ensure_dir(out_dir)
    print(f"[INFO] Writing results to: {out_dir}")

    # -------------------- load env & agent ---------------------------------
    env = make_env(seed=0)
    
    obs_dim = env.observation_space.shape[0]
    horizon = env.horizon
    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    print(f"[INFO] Environment horizon={env.horizon}, "
          f"obs_dim={state_dim}, actions={num_actions}")

    print(f"[INFO] Loading trained Keras model from: {model_path}")
    # q_net = tf.keras.models.load_model(model_path, compile=False)
    q_net = tf.keras.models.load_model(
        model_path, compile=False, custom_objects={"QNetwork": QNetwork}
    )

    # explicitly build it for a batch of size None
    q_net.build((None, obs_dim))

    # dummy forward‐pass to get the real output size
    dummy = tf.zeros((1, obs_dim), dtype=tf.float32)
    out_tensor = q_net(dummy)
    output_dim = int(out_tensor.shape[-1])

    if output_dim != num_actions:
        raise ValueError(
            f"Loaded model has output dim {output_dim}, "
            f"but env.action_space.n = {num_actions}"
        )

    # -------------------- baseline SSD (random) ----------------------------
    print("[INFO] Estimating baseline SSD with random policy …")
    baseline_ctr = Counter()
    
    for _ in tqdm(range(n_episodes), desc="Baseline"):
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            baseline_ctr[_state_id_from_info(info, obs)] += 1

    # -------------------- controlled rollouts ------------------------------
    success = 0
    returns, interventions = [], []
    ctrl_ctr = Counter()

    print(f"[INFO] Evaluating trained agent for {n_episodes} episodes …")

    for ep in tqdm(range(n_episodes), desc="DDQN-PER"):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        flips = 0            # interventions (non-null actions)

        while not done:
            action = greedy_action(q_net, obs.astype(np.float32))
            # if action != num_actions - 1:
            if action != 0:
                flips += 1
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_ret += r
            #ep_len += 1
            ctrl_ctr[_state_id_from_info(info, obs)] += 1

        returns.append(ep_ret)
        if term and not trunc:                              # only count *successful* completes
            success += 1
            interventions.append(flips)

    # -------------------- metrics ------------------------------------------
    success_rate   = success / n_episodes
    mean_interv    = float(np.mean(interventions)) if interventions else math.nan

    # SSD delta (KL-like diff for *reported* states)
    baseline_total = sum(baseline_ctr.values()) or 1
    ctrl_total     = sum(ctrl_ctr.values())     or 1
    all_states     = set(baseline_ctr) | set(ctrl_ctr)
    ssd_delta = {int(s): (ctrl_ctr[s]/ctrl_total) - (baseline_ctr[s]/baseline_total)
                 for s in all_states}

    metrics = dict(
        episodes          = n_episodes,
        success_rate      = success_rate,
        mean_interventions= mean_interv,
        avg_return        = float(np.mean(returns)),
        std_return        = float(np.std(returns)),
    )

    # -------------------- plots --------------------------------------------
    plot_episode_return_hist(returns, out_dir / "episode_returns.png")
    plot_ssd_bar(baseline_ctr, ctrl_ctr, out_dir / "ssd_top20.png")
    plot_ssd_line(baseline_ctr, ctrl_ctr, out_dir / "ssd_all_states.png")


    # -------------------- save TOP-20 table -------------------------------
    # compute probabilities again (we already did it for the plot)
    baseline_total = sum(baseline_ctr.values()) or 1
    ctrl_total = sum(ctrl_ctr.values())     or 1

    top_states = [s for s, _ in ctrl_ctr.most_common(20)]
    top20_info = []
    for s in top_states:
        p_rand = baseline_ctr[s] / baseline_total
        p_ctrl = ctrl_ctr[s]    / ctrl_total
        top20_info.append({
            "state_id"  : int(s),
            "p_random"  : p_rand,
            "p_control" : p_ctrl,
            "delta"     : p_ctrl - p_rand
        })


    # -------------------- export results -----------------------------------
    log_json(metrics,          out_dir / "metrics.json")
    log_json(ssd_delta,        out_dir / "ssd_delta.json")
    log_json(top20_info,       out_dir / "top20_states.json")
    # Also keep copy of training log if user provides
    print("[INFO] Evaluation finished, saved metrics & plots.")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DDQN-PER agent on a Gym-PBN env"
    )
    parser.add_argument("model", type=Path,
                        help="Path to `.keras` checkpoint (q_network)")
    parser.add_argument("-n", "--episodes", type=int, default=1000,
                        help="# evaluation episodes (default: 1000)")
    parser.add_argument("--out", type=Path, default=Path("evals"),
                        help="Root folder where eval_<timestamp> is created")
    args = parser.parse_args()

    evaluate(model_path=args.model,
             n_episodes=args.episodes,
             out_root=args.out)
