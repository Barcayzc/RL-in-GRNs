import random
import numpy as np
from typing import Any, List, Tuple

Transition = Tuple[Any, Any, float, Any, bool]
EPSILON = 1e-6  # small constant to avoid zero priority


class SumSegmentTree:
    """Binary indexed segment tree supporting sum queries and prefix‐sum indexing."""
    def __init__(self, capacity: int):
        # Next power of two for capacity
        self._n = 1
        while self._n < capacity:
            self._n <<= 1
        self._size = capacity
        # Tree array: [1 .. 2*n), 1-based indexing at root=1
        self._tree = np.zeros(2 * self._n, dtype=np.float32)

    def update(self, idx: int, value: float):
        """Set value at leaf idx, then update internal nodes."""
        tree_idx = idx + self._n
        self._tree[tree_idx] = value
        # Walk up and update parents
        parent = tree_idx >> 1
        while parent >= 1:
            self._tree[parent] = self._tree[2*parent] + self._tree[2*parent + 1]
            parent >>= 1

    def sum_total(self) -> float:
        """Returns sum over all leaf values."""
        return float(self._tree[1])

    def find_prefixsum_idx(self, prefix: float) -> int:
        """
        Find highest idx such that cumulative sum up to idx >= prefix.
        Returns a leaf index in [0, size).
        """
        idx = 1
        while idx < self._n:  # while not at leaf
            left = 2 * idx
            if self._tree[left] >= prefix:
                idx = left
            else:
                prefix -= self._tree[left]
                idx = left + 1
        return idx - self._n


class MinSegmentTree:
    """Similar to SumSegmentTree but supports range minimum query over priorities."""
    def __init__(self, capacity: int):
        # Next power of two for capacity
        self._n = 1
        while self._n < capacity:
            self._n <<= 1
        self._size = capacity
        # Initialize with +inf so unused leaves don't interfere
        self._tree = np.full(2 * self._n, float('inf'), dtype=np.float32)

    def update(self, idx: int, value: float):
        """Set value at leaf idx, then update internal nodes with min."""
        tree_idx = idx + self._n
        self._tree[tree_idx] = value
        parent = tree_idx >> 1
        while parent >= 1:
            self._tree[parent] = min(self._tree[2*parent], self._tree[2*parent + 1])
            parent >>= 1

    def min(self) -> float:
        """Returns minimum over all leaf values."""
        return float(self._tree[1])


class PrioritizedReplayBuffer:
    """
    Standalone Prioritized Experience Replay Buffer.

    Stores transitions with priorities, supports sampling by priority,
    and updating priorities. Internally uses a SumSegmentTree for
    proportional sampling and a MinSegmentTree for retrieving the
    minimum priority (for importance‐sampling weight normalization).
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Args:
            capacity: Maximum number of transitions to store.
            alpha: Priority exponent (0 = uniform sampling, 1 = full prioritization).
        """
        self.capacity = capacity
        self.alpha = alpha

        # Segment trees
        self._sum_tree = SumSegmentTree(capacity)
        self._min_tree = MinSegmentTree(capacity)

        # Experience storage
        self._data: List[Transition] = [None] * capacity
        self._next_idx = 0
        self._size = 0

        # Track maximal priority for new transitions
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self._size
    size = __len__

    def store(self, transition: Transition):
        """
        Adds a new transition to the buffer with maximal priority.

        Args:
            transition: A tuple (state, action, reward, next_state, done).
        """
        idx = self._next_idx
        self._data[idx] = transition

        # Assign max priority to new transition
        priority = self._max_priority ** self.alpha
        self._sum_tree.update(idx, priority)
        self._min_tree.update(idx, priority)

        # Advance pointer
        self._next_idx = (self._next_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
    add = store

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4
    ) -> Tuple[List[Transition], List[int], np.ndarray]:
        """
        Samples a batch of transitions with probabilities proportional to priority.
        Returns transitions, their indices, and importance‐sampling weights.

        Args:
            batch_size: Number of transitions to sample.
            beta: Importance-sampling exponent (0 = no correction, 1 = full correction).

        Returns:
            transitions: List of sampled transitions.
            indices: List of indices in the buffer.
            weights: Array of shape (batch_size,) of IS weights in [0,1].
        """
        assert self._size > 0, "Cannot sample from an empty buffer"

        # Total priority mass
        total_sum = self._sum_tree.sum_total()
        segment = total_sum / batch_size

        transitions = []
        indices = []
        weights = np.empty(batch_size, dtype=np.float32)

        # Minimum probability for weight normalization
        min_prob = self._min_tree.min() / total_sum
        max_weight = (min_prob * self._size) ** (-beta)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx = self._sum_tree.find_prefixsum_idx(s)

            transitions.append(self._data[idx])
            indices.append(idx)

            # Compute importance-sampling weight
            p_i = self._sum_tree._tree[idx + self._sum_tree._n] / total_sum
            w = (p_i * self._size) ** (-beta)
            weights[i] = w / max_weight  # normalize to [0, 1]

        states, actions, rewards, next_states, dones = map(
            np.array, zip(*transitions)
        )

        return (
            states.astype(np.float32),
            actions.astype(np.int32),
            rewards.astype(np.float32),
            next_states.astype(np.float32),
            dones.astype(np.float32),
            weights.astype(np.float32),
            np.array(indices, dtype=np.int32),
        )

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Updates the priorities of sampled transitions.

        Args:
            indices: List of buffer indices for the transitions.
            priorities: List of new priority values (e.g. absolute TD errors).
        """
        for idx, p in zip(indices, priorities):
            # Add a small epsilon and apply alpha exponent
            p_adjusted = (abs(p) + EPSILON) ** self.alpha
            self._sum_tree.update(idx, p_adjusted)
            self._min_tree.update(idx, p_adjusted)
            # Track max raw priority for new inserts
            self._max_priority = max(self._max_priority, abs(p) + EPSILON)
