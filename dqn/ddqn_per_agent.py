import numpy as np
import tensorflow as tf
from .q_network import QNetwork
from .per_buffer import PrioritizedReplayBuffer
from pathlib import Path


class DDQNPERAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        learning_rate=1e-3,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        batch_size=64,
        memory_capacity=100_000,
        alpha=0.6,
        beta=0.4,
        beta_increment=1e-6,
        target_update_freq=1000,
    ):
        # Dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.beta = beta
        self.beta_increment = beta_increment
        self.target_update_freq = target_update_freq

        # Replay memory with Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(capacity=memory_capacity, alpha=alpha)

        # Q-Networks: online & target
        self.q_network = QNetwork(self.state_dim, self.action_dim)
        self.target_q_network = QNetwork(self.state_dim, self.action_dim)
        # Initialize target weights
        self.target_q_network.set_weights(self.q_network.get_weights())

        # Optimizer and loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss_fn = tf.keras.losses.Huber()

        # Training step counter
        self.train_step_count = 0

    def act(self, state):
        """
        Selects an action using epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        # Greedy action from Q-network
        state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        q_values = self.q_network(state_tensor)
        return int(tf.argmax(q_values[0]).numpy())

    def store_transition(self, state, action, reward, next_state, done):
        """
        Adds a transition to the replay buffer.
        """
        transition = (state, action, reward, next_state, done)
        self.memory.add(transition)

    def train_step(self):
        """
        Samples a batch from memory and updates the Q-network.
        """
        # Do not train until enough samples
        if self.memory.size() < self.batch_size:
            return

        # Sample from PER memory
        (states, actions, rewards, next_states, dones,
         weights, indices) = self.memory.sample(self.batch_size, beta=self.beta)

        # Convert to tensors
        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones_tf = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights_tf = tf.convert_to_tensor(weights, dtype=tf.float32)

        # Double DQN target computation
        # 1) Action selection with online network
        q_next_online = self.q_network(next_states_tf)
        best_actions = tf.argmax(q_next_online, axis=1)
        # 2) Q-value evaluation with target network
        q_next_target = self.target_q_network(next_states_tf)
        batch_indices = tf.range(self.batch_size, dtype=tf.int64)
        target_q_values = tf.gather_nd(q_next_target,
                                        tf.stack([batch_indices, best_actions], axis=1))
        # Compute TD targets: r + gamma * Q_target(s', argmax Q_online)
        targets = rewards_tf + self.gamma * target_q_values * (1 - dones_tf)

        # Train online Q-network
        with tf.GradientTape() as tape:
            q_values = self.q_network(states_tf)
            # Select Q-values for taken actions
            action_mask = tf.one_hot(actions_tf, self.action_dim, dtype=tf.float32)
            q_selected = tf.reduce_sum(q_values * action_mask, axis=1)

            # Compute Huber loss per sample
            loss_unweighted = self.loss_fn(targets, q_selected)
            # Apply importance-sampling weights
            loss = tf.reduce_mean(weights_tf * loss_unweighted)

        # Backpropagation
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # Update priorities in replay buffer using absolute TD errors
        td_errors = tf.abs(targets - q_selected).numpy() + 1e-6
        self.memory.update_priorities(indices, td_errors)

        # Increment training step counter
        self.train_step_count += 1

        # Periodic target network update
        if (self.train_step_count % self.target_update_freq) == 0:
            self.update_target_network()

        # Anneal epsilon and beta
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.beta = min(1.0, self.beta + self.beta_increment)

    def update_target_network(self):
        """
        Copies online network weights to target network.
        """
        self.target_q_network.set_weights(self.q_network.get_weights())

    KERAS_SUFFIX = ".keras"

    @staticmethod
    def _add_suffix(path):
        """
        Ensure path ends with `.keras`, create parent dir, return str.
        """
        p = Path(path)
        if p.suffix != DDQNPERAgent.KERAS_SUFFIX:
            p = p.with_suffix(DDQNPERAgent.KERAS_SUFFIX)
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    # -----------------------------------------------------------------
    def save(self, path):
        """Save online Q-network in native Keras v3 format."""
        self.q_network.save(self._add_suffix(path))

    def load(self, path):
        """Load *.keras file into online net and sync target net."""
        self.q_network = tf.keras.models.load_model(self._add_suffix(path))
        self.target_q_network.set_weights(self.q_network.get_weights())