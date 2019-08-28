from numba import jit, float32
import numpy as np

@jit(float32[:](float32[:]), nopython=True, nogil=True)
def multinomial_sample(pvals):
    """Draw one sample from a multinomial distribution (similar to np.random.multinomial)
    pvals   - 1D array of probabilities (that should sum to one - but doesn't have to)
    return  - 1D array of same length as pvals with one value set to 1. and all others 0.
              this can then be used as an array of probabilities"""

    # random float in the half-open interval [0.0, np.sum(pvals))
    rand = np.random.random() * np.sum(pvals)

    # cumulative sum is same type as pvals (float32) so that the sum of pvals is always less than rand
    cumsum = np.float32(0.0)
    output_probs = np.zeros_like(pvals, dtype=np.float32)
    for i, prob in enumerate(pvals):
        cumsum += prob
        if rand < cumsum:
            output_probs[i] = 1
            break

    return output_probs
