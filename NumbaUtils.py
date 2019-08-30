from numba import jit, float32
import numpy as np


@jit([float32[:](float32[:]), float32[:,:](float32[:,:])], nopython=True, nogil=True)
def multinomial_sample(pvals):
    """Draw one sample from a multinomial distribution (similar to np.random.multinomial)
    pvals   - 1D or 2D array of probabilities (np.float32) (that should sum to one - but doesn't have to)
    return  - 1D or 2D array (np.float32) of same length as pvals with one value set to 1. and all others 0.
              this can then be used as an array of probabilities
    Note: the implementation will work with a pvals array of any dimension, but the function signature
    limits it to 1D or 1D"""

    # Random float in the half-open interval [0.0, np.sum(pvals))
    rand = np.random.random() * np.sum(pvals)

    # Cumulative sum is same type as pvals so that the total sum of pvals is always less than rand
    cumsum = np.float32(0.0)

    # 1D loop over pvals using pvals.ravel()
    output_probs = np.zeros(pvals.size, dtype=np.float32)
    for i, prob in enumerate(pvals.ravel()):
        cumsum += prob
        if rand < cumsum:
            output_probs[i] = 1
            break

    # Reshape output to be same as pvals shape
    return output_probs.reshape(pvals.shape)