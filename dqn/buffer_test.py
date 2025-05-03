import numpy as np
from collections import defaultdict
from per_buffer import PrioritizedReplayBuffer

# 1) Create a tiny buffer
buf = PrioritizedReplayBuffer(capacity=8, alpha=0.6)

# 2) Push in some “dummy” transitions
for i in range(8):
    # (state, action, reward, next_state, done)
    tr = (np.array([i]), i, float(i), np.array([i+1]), False)
    buf.store(tr)

print("Buffer size (should be 8):", len(buf))

# 3) Sample a batch
batch, idxs, weights = buf.sample(batch_size=4, beta=0.4)
print("\nSampled transitions:")
for t, idx, w in zip(batch, idxs, weights):
    print(f" idx={idx:2d}  tr={t}  w={w:.4f}")

# 4) Check shapes
assert len(batch) == 4
assert len(idxs)  == 4
assert weights.shape == (4,)

# 5) Artificially “update” their priorities (e.g. new TD‐errors = [1,2,3,4])
new_prios = [1.0, 2.0, 3.0, 4.0]
buf.update_priorities(idxs, new_prios)

# 6) Re‐sample and see if the indices with higher prios appear more often
counts = defaultdict(int)           # will auto‐zero missing keys
for _ in range(5000):
    _, (idx,), _ = buf.sample(batch_size=1, beta=1.0)
    counts[idx] += 1
# Now inspect just the ones we care about:
for idx in idxs:
    print(f" idx={idx:2d}  count={counts[idx]}")


# If everything is correctly:
#  - No exceptions should be raised
#  - The ‘counts’ for higher‐priority indices (those we gave larger new_prios)
#    should be noticeably larger than for the smaller‐priority ones.
