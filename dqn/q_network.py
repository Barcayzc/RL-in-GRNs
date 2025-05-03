import tensorflow as tf
from keras.saving import register_keras_serializable


@register_keras_serializable()
class QNetwork(tf.keras.Model):
    """
    Fully-connected Q-network (DDQN).

    Args
    ----
    input_dim      : int – length of state vector
    output_dim     : int – number of actions
    hidden_units   : list[int] – sizes of hidden Dense layers
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_units: list[int] | None = None,
                 **kwargs):
        super().__init__(**kwargs)

        hidden_units = hidden_units or [64, 64]          # safe default
        self.input_dim    = input_dim
        self.output_dim   = output_dim
        self.hidden_units = hidden_units                 # <-- single name!

        # Hidden layers
        self.hidden_layers = [
            tf.keras.layers.Dense(u,
                                  activation="relu",
                                  kernel_initializer="he_uniform",
                                  name=f"hidden_{i+1}")
            for i, u in enumerate(hidden_units)
        ]

        # Output (linear) layer
        self.q_out = tf.keras.layers.Dense(
            output_dim,
            activation=None,
            kernel_initializer="he_uniform",
            name="q_values"
        )

    # ------------------------------------------------------------------ #
    # Keras (de)serialisation
    # ------------------------------------------------------------------ #
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(input_dim=self.input_dim,
                        output_dim=self.output_dim,
                        hidden_units=list(self.hidden_units)))
        return cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.q_out(x)

    # Optional helper
    def build_graph(self):
        self.build((None, self.input_dim))
        return tf.keras.Model(inputs=self.input, outputs=self.call(self.input))


# Smoke-test ------------------------------------------------------------- #
if __name__ == "__main__":
    net = QNetwork(100, 101)
    _ = net(tf.zeros((1, 100)))   #  <-- builds weights
    net.summary()                 # now shows the parameter counts

    net.save("tmp_qnet.keras")
    net2 = tf.keras.models.load_model(
            "tmp_qnet.keras", custom_objects={"QNetwork": QNetwork})
    print("Reload OK:", isinstance(net2, QNetwork))



# if __name__ == "__main__":
#     # Quick smoke test
#     input_dim = 100
#     output_dim = 101
#     batch_size = 32

#     model = QNetwork(input_dim=input_dim, output_dim=output_dim)
#     # Build the model by calling it once
#     dummy_input = tf.random.uniform((batch_size, input_dim))
#     dummy_q = model(dummy_input)

#     print("Input shape:", dummy_input.shape)
#     print("Output shape:", dummy_q.shape)
#     model.summary()
