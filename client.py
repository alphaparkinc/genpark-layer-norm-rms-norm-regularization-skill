"""LayerNorm and RMSNorm Normalization Layers.
100% Python Standard Library.
"""

import math

class NormalizationLayers:
    """Standard LayerNorm and Root Mean Square Normalization (RMSNorm)."""

    @staticmethod
    def layer_norm(x: list, gamma: list = None, beta: list = None, eps: float = 1e-5) -> list:
        d = len(x)
        mean = sum(x) / d
        var = sum((v - mean) ** 2 for v in x) / d
        std = math.sqrt(var + eps)
        norm = [(v - mean) / std for v in x]
        if gamma and beta:
            norm = [norm[i] * gamma[i] + beta[i] for i in range(d)]
        return norm

    @staticmethod
    def rms_norm(x: list, gamma: list = None, eps: float = 1e-5) -> list:
        d = len(x)
        rms = math.sqrt(sum(v * v for v in x) / d + eps)
        norm = [v / rms for v in x]
        if gamma:
            norm = [norm[i] * gamma[i] for i in range(d)]
        return norm
