"""Static (non-longitudinal) causal inference estimators.

Module built progressively through week 1 of the Python for DS course.
"""

import numpy as np
import pandas as pd
from scipy.special import expit


def simulate_static_binary(
    n: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate a static confounded binary-outcome dataset.
    Data-generating process:
        X1, X2 ~ N(0, 1)    independent
        P(A=1 | X) = expit(-0.5 + 0.8*X1 + 0.5*X2)
        P(Y=1 | A, X) = expit(-1.0 + 0.7*A + 0.6*X1 - 0.4*X2)
    Args:
        n: sample size.
        seed: optional RNG seed for reproducibility.
    Returns:
        DataFrame with columns X1, X2, A, Y.
    """
    rng = np.random.default_rng(seed)

    # TODO: sample X1, X2 from N(0, 1)
    # TODO: compute P(A=1 | X), sample A
    # TODO: compute P(Y=1 | A, X), sample Y

    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)  

    p_a1 = expit(-0.5 + 0.8*X1 + 0.5*X2)
    A = rng.binomial(1, p_a1)

    p_y1 = expit(-1.0 + 0.7*A + 0.6*X1 - 0.4*X2)
    Y = rng.binomial(1, p_y1)

    # TODO: return DataFrame with columns X1, X2, A, Y
    # Python
    return pd.DataFrame({"X1": X1, "X2": X2, "A": A, "Y": Y})
