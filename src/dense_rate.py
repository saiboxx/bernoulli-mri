class DenseRateScheduler:
    """Scheduler for annealing the target dense ratio of the network."""

    def __init__(self, target: float, start_epoch: int, stop_epoch: int) -> None:
        """Initialize a DenseRateScheduler object."""
        self.dense_rate = 1.0
        self.target_rate = target
        self.current_epoch = 1
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch

    def __call__(self) -> float:
        """Return dense rate on object call."""
        return self.get_dense_rate()

    def get_dense_rate(self) -> float:
        """Return dense rate."""
        return self.dense_rate

    def set_dense_rate(self) -> None:
        """Get the target remaining ratio."""
        if self.current_epoch < self.start_epoch:
            self.dense_rate = 1.0
        elif self.current_epoch >= self.stop_epoch:
            self.dense_rate = self.target_rate
        else:
            div_term = (self.current_epoch - self.start_epoch) / (
                self.stop_epoch - self.start_epoch
            )
            dr = self.target_rate + ((1 - self.target_rate) * (1 - div_term) ** 3)
            self.dense_rate = dr

    def advance(self) -> None:
        """Increase current epoch counter."""
        self.current_epoch += 1
        self.set_dense_rate()
