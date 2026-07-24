"""Static (non-longitudinal) causal inference estimators.

Module built progressively through week 1 of the Python for DS course.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


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

    p_a1 = expit(-0.5 + 0.8 * X1 + 0.5 * X2)
    A = rng.binomial(1, p_a1)

    p_y1 = expit(-1.0 + 0.7 * A + 0.6 * X1 - 0.4 * X2)
    Y = rng.binomial(1, p_y1)

    # TODO: return DataFrame with columns X1, X2, A, Y
    # Python
    return pd.DataFrame({"X1": X1, "X2": X2, "A": A, "Y": Y})


def fit_propensity_scikit(
    df: pd.DataFrame, treatment: str, covariates: list[str], trunc: float | None = 0.95
) -> pd.DataFrame:
    """fit a propensity score on confounded binary-outcome dataset.
    Args:
        df: the data frame with covariates and outcome,
        treatment: the treatment (A) character name in df,
        covariates: the list of covariates name to adjut on,
        trunc: the truncation parameter of the weight which by default is 0.95,
    Returns:
        df with added two columns, inverse prob treatment weights and stabilized version
    """

    df = df.copy()
    clf = LogisticRegression(random_state=0, penalty=None, C=np.inf).fit(df[covariates], df[treatment])
    weights = pd.DataFrame(1 / clf.predict_proba(df[covariates]), index=df.index)
    # treated = df[treatment] == 1
    # np.repeat(0, df.shape[0])
    # df["weights_1"] = np.repeat(0.0, df.shape[0])
    # df.loc[treated, "weights_1"] = weights.loc[treated, 1]
    # df.loc[~treated, "weights_1"] = weights.loc[~treated, 0]
    df["weights_1"] = np.where(df[treatment] == 1, weights[1], weights[0])

    return df
