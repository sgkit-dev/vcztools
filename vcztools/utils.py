import numpy as np


def search(a, v):
    """Like `np.searchsorted`, but array `a` does not have to be sorted."""
    sorter = np.argsort(a)
    rank = np.searchsorted(a, v, sorter=sorter)
    return sorter[rank]
