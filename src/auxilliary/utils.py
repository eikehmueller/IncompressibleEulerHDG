"""
Utility classes
"""

__all__ = ["Averager"]


class Averager:
    """Compute running average

    Computes the average S_n = 1/n sum_{j=1,n} x_n of n numbers on the fly
    """

    def __init__(self):
        """Initialise new instance"""
        self.reset()

    @property
    def value(self):
        """Return current value of average"""
        return self._average

    @property
    def n_samples(self):
        """Return number of processed samples since last reset"""
        return self._n_samples

    def update(self, x):
        """Include another number in the average

        :arg x: number to include
        """
        self._n_samples += 1
        self._average += (x - self._average) / self._n_samples

    def reset(self):
        """Reset the averager"""
        self._n_samples = 0
        self._average = 0

    def __repr__(self):
        """Internal string representation"""
        return f"{self.value} (averaged over {self.n_samples} samples)"
