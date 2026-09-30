"""Running mean and variance, used for observation and return normalization."""

import numpy as np


class RunningMeanStd:
    """Parallel-algorithm running moments over the leading axis of a batch."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Fold a batch ``x`` of shape ``(n, *shape)`` into the moments."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + np.square(delta) * self.count * batch_count / tot_count
        )
        self.mean = self.mean + delta * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    def state_dict(self) -> dict[str, np.ndarray | float]:
        """Moments as a checkpointable dict."""
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, np.ndarray | float]) -> None:
        """Restore moments saved by ``state_dict``."""
        self.mean = np.asarray(state["mean"], np.float64)
        self.var = np.asarray(state["var"], np.float64)
        self.count = float(state["count"])  # type: ignore[arg-type]
