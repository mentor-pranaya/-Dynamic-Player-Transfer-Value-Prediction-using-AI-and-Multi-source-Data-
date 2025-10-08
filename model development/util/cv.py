
import numpy as np
from typing import List, Tuple
# Placeholder for time-aware CV splitters (blocked CV, group-aware)
class BlockedTimeSeriesSplit:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
    def split(self, X, y=None, groups=None):
        n = len(X)
        fold = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = fold * (i+1)
            test_end = fold * (i+2)
            yield np.arange(0, train_end), np.arange(train_end, test_end)
