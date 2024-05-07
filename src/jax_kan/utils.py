from flax import linen as nn


class Identity(nn.Module):
    """Identity function, useful for generic modules."""

    @nn.compact
    def __call__(self, x):
        return x
