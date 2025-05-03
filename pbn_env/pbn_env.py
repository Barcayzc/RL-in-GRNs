# pbn_env/pbn_env.py
# ─────────────────────────────────────────────────────────────────────────────
"""
Factory that builds the Gym-PBN environment using the artefacts
written during data-preparation (logic functions, gene names, train
reset-pool, …).  Import and call `make_env()` from training scripts.
"""
import pickle
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import gym_PBN

# ─────────────────────────────────────────────────────────────────────────────
# Static artefact locations (adjust if you move files)
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_prepared"
# DATA_DIR  = Path("./data_prepared")
TRAIN_STATES_PATH = DATA_DIR / "train_states.npy"
LOGIC_PATH        = DATA_DIR / "logic_func_data_safe.pkl"
GENE_NAMES_PATH   = DATA_DIR / "gene_names_safe.txt"

# ─────────────────────────────────────────────────────────────────────────────
# One-time loading of gene names & logic functions
# (executed once when the module is first imported)
# ─────────────────────────────────────────────────────────────────────────────
with open(GENE_NAMES_PATH, "r") as f:
    NODE_NAMES = [ln.strip() for ln in f if ln.strip()]

with open(LOGIC_PATH, "rb") as f:
    LOGIC_MAP = pickle.load(f)

LOGIC_FUNCS = [LOGIC_MAP[name] for name in NODE_NAMES]
LOGIC_FUNC_DATA = (NODE_NAMES, LOGIC_FUNCS)

# Pre-generated pool of initial states (numpy bool array or 0/1 ints)
TRAIN_POOL = np.load(TRAIN_STATES_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# Optional user-defined “target attractor”; let env discover if None
# ─────────────────────────────────────────────────────────────────────────────
TARGET_STATE = tuple([0] * len(NODE_NAMES))   # all-zeros, just an example
GOAL_CONFIG  = {
    "all_attractors": [{TARGET_STATE}],
    "target": {TARGET_STATE},
}
HORIZON = 20


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────
def make_env(seed: int | None = None):
    """
    Returns a *fresh* gym_PBN PBN-v0 instance every call.

    Parameters
    ----------
    seed : int | None
        Gymnasium PRNG seed to pass to `env.reset(seed=seed)`.
    """
    env = gym.make(
        "gym-PBN/PBN-v0",
        logic_func_data=LOGIC_FUNC_DATA,
        goal_config=GOAL_CONFIG,
        # reward_config={
        #     "successful_reward": 5,
        #     "wrong_attractor_cost": -2,
        #     "action_cost": -1,
        # },
        reward_config={
            "step_cost":           -1,   # every time‐step you pay 1 point
            "action_cost":         -1,   # and you pay 1 more if you actually flip
            "successful_reward":    5,   # +5 for target attractor
            "wrong_attractor_cost": -2,  # -2 for landing in any other attractor
        },
    )

    # Env-specific settings
    env.horizon = HORIZON

    # Inject our custom reset-pool (env looks for `train_states`)
    env.train_states = TRAIN_POOL

    # Seed (optional)
    if seed is not None:
        env.reset(seed=seed)

    return env


# ─────────────────────────────────────────────────────────────────────────────
# Optional smoke-test when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    e = make_env(seed=0)
    obs, info = e.reset()
    print("Reset obs:", obs[:10], "...")   # show first 10 bits
    for t in range(5):
        a = e.action_space.sample()
        obs, r, term, trunc, _ = e.step(a)
        print(f"step {t:2d}: a={a:3d}  r={r:+.1f}  done={term or trunc}")
        if term or trunc:
            break
    print("\nObs space:", e.observation_space)
    print("Act space:", e.action_space)
    print("Horizon  :", e.horizon)
