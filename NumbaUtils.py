from numba import jit, int64, float32
import numpy as np

@jit(int64[:](float32[:]), nopython=True, nogil=True)
def multinomial_sample(pvals):
    """Draw one sample from a multinomial distribution (similar to np.random.multinomial)"""

    # random float in the half-open interval [0.0, np.sum(pvals))
    rand = np.random.random() * np.sum(pvals)
    # cumulative sum is same type as pvals (float32) so that the sum of pvals is always less than rand
    cumsum = np.float32(0.0)
    output = np.zeros_like(pvals, dtype=np.int64)
    for i, p in enumerate(pvals):
        cumsum += p
        if rand < cumsum:
            output[i] = 1
            break

    return output