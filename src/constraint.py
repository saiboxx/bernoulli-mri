from torch import Tensor


class ScoreConstrainer:
    """Ensure model weights are [0, 1] and only a given fraction is active."""

    def __init__(self, scores: Tensor) -> None:
        """Initialize a ScoreConstrainer object."""
        self.scores = scores
        self.num_scores = scores.numel()

    @property
    def max_score(self) -> float:
        """Obtain maximum score value."""
        return float(self.scores.max())  # type: ignore

    def constrain(self, dense_rate: float) -> None:
        """Constrain the scores."""
        k = int(self.num_scores * dense_rate)
        v = self.find_optimal_v(k=k)
        self.scores.data.sub_(v).clamp_(0, 1)

    def find_optimal_v(self, k: int, iterations: int = 20, eps: float = 1e-3) -> float:
        """Find an optimal v for a given k by bisection search."""
        v = 0.0
        if self._eval_v_candidate(v=v, k=k) < 0:
            return 0.0

        lower_bound = 0.0
        upper_bound = self.max_score
        for _ in range(iterations):
            v = (lower_bound + upper_bound) / 2
            v_deviation = self._eval_v_candidate(v=v, k=k)

            if abs(v_deviation) < eps:
                return max(0, v)

            if v_deviation < 0:
                upper_bound = v
            else:
                lower_bound = v

        return max(0, v)

    def _eval_v_candidate(self, v: float, k: int) -> float:
        """Compute deviation of k with the projection of the scores by v."""
        score_sum = (self.scores - v).clamp(0, 1).sum()
        return float(score_sum) - k
