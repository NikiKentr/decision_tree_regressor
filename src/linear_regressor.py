import numpy as np
from numpy.typing import NDArray
from typing import Literal

class LinearRegressor:
    def __init__(self, feature_type: Literal["lin", "quad", "rbf"] = "quad") -> None:
        self.feature_type = feature_type
        self.beta_hat = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        self.beta_hat = (
            np.linalg.inv(phi(X, self.feature_type).T @ phi(X, self.feature_type))
            @ phi(X, self.feature_type).T
            @ y
        )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return phi(X, self.feature_type) @ self.beta_hat


def phi(x: NDArray[np.float64], feat_type: Literal["lin", "quad", "rbf"] = "quad") -> NDArray[np.float64]:
    N, nx = x.shape
    if feat_type == "lin":
        return np.hstack((np.ones((N, 1)), x))
    elif feat_type == "quad":
        if nx == 1:
            return np.hstack(
                (
                    np.ones((N, 1)),
                    x,
                    x**2,
                )
            )
        elif nx == 2:
            return np.hstack(
                (
                    np.ones((N, 1)),
                    x,
                    (x[:, 0] ** 2).reshape((N, 1)),
                    (x[:, 0] * x[:, 1]).reshape((N, 1)),
                    (x[:, 1] ** 2).reshape((N, 1)),
                )
            )
        else:
            raise NotImplementedError
    elif feat_type == "rbf":
        b = lambda x, c: np.exp(-1 / 2 * (x - c) ** 2)
        rbf_list = []
        rbf_list.append(np.ones((N, 1)))
        for j in range(0, 15, 1):
            rbf_list.append(b(x, j))
        return np.hstack(rbf_list)
    else:
        raise NotImplementedError