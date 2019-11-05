from numba import jit
import numpy as np
from . import NumbaUtils


def haploidHMM(targetHaplotype, sourceHaplotypes, error, recombinationRate, threshold=0.9, n_samples=10, callingMethod="dosages"):

    nLoci = len(targetHaplotype)

    # !!!! May need to cast the source Haplotypes to a matrix. #May also want to handle the probabilistic case.
    if type(sourceHaplotypes) is list:
        sourceHaplotypes = np.array(sourceHaplotypes)
    
    if type(error) is float:
        error = np.full(nLoci, error, dtype = np.float32)

    if type(recombinationRate) is float:
        recombinationRate = np.full(nLoci, recombinationRate, dtype = np.float32)
    # !!!! May need to have marker specific error/recombination rates.


    ### Build haploid HMM. 
    ###Construct penetrance values

    pointEst = getHaploidPointEstimates(targetHaplotype, sourceHaplotypes, error)
    ### Run forward-backward algorithm on penetrance values.

    # Don't need forward backward probabilites if using the sampler method
    if callingMethod != 'sampler':
        hapEst = haploidForwardBackward(pointEst, recombinationRate)

    # for i in range(nLoci) :
    #     print(hapEst[:,i])

    # raise Exception()

    ### Could also do a sampling approach, or a viterbi approach, or get dosages...
    ### Averaging over multiple samples doesn't make sense. Dosages would be better in that case.
    ### Viterbi would be an alternative.

    # callingMethod = "callhaps"
    if callingMethod == "callhaps":
        ### Call haplotypes.
        calledHaps = haploidCallHaps(hapEst, threshold) 
        ### Call genotypes.
        calledGenotypes = getHaploidGenotypes(calledHaps, sourceHaplotypes)
        return calledGenotypes
    if callingMethod == "dosages" :
        dosages = getHaploidDosages(hapEst, sourceHaplotypes)
        return dosages
    if callingMethod == 'sampler':
        dosages = getSampledDosages(pointEst, sourceHaplotypes, recombinationRate, n_samples)
        return dosages

    if callingMethod == "viterbi" :
        raise ValueError("Viterbi not yet implimented.")


@jit(nopython=True)
def getHaploidDosages(hap_est, source_haplotypes):
    """Calculate dosages for a single haplotype"""
    n_loci, n_haps = hap_est.shape
    dosages = np.zeros(n_loci, dtype=np.float32)
    for i in range(n_loci):
        for j in range(n_haps):
            dosages[i] += source_haplotypes[j, i] * hap_est[i, j]
    return dosages


@jit(nopython=True)
def getSampledDosages(point_estimates, haplotype_library, recombination_rate, n_samples=10):
    """Calculate dosages for a single haplotype by sampling"""

    # Pre-calculate forward probabilities
    forward_probs = haploidForward(point_estimates, recombination_rate)

    # Update sampled dosages as the mean of each sample
    n_loci = point_estimates.shape[1]
    dosages = np.zeros(n_loci, dtype=np.float32)
    for i in range(n_samples):
        dosages += haploidSampler(forward_probs, haplotype_library, recombination_rate)
    return dosages / n_samples


@jit(nopython=True)
def haploidCallHaps(hapEst, threshold ):
    nHaps, nLoci = hapEst.shape
    calledHaps = np.full(nLoci, -1, dtype = np.int64) # These are haplotype ids. -1 is missing.
    for i in range(nLoci):
        maxVal = -1
        maxLoc = -1
        for j in range(nHaps):
            if hapEst[j, i] > threshold and hapEst[j,i] > maxVal:
                maxLoc = j
                maxVal = hapEst[j, i]
        calledHaps[i] = maxLoc
    return calledHaps

@jit(nopython=True)
def getHaploidGenotypes(calledHaps, sourceHaplotypes):
    nHaps, nLoci = sourceHaplotypes.shape
    calledGenotypes = np.full(nLoci, 9, dtype = np.int8) # These are haplotype ids. -1 is missing.
    for i in range(nLoci):
        if calledHaps[i] != -1 :
            calledGenotypes[i] = sourceHaplotypes[calledHaps[i],i]
    return calledGenotypes

@jit(nopython=True)
def getHaploidPointEstimates(targetHaplotype, sourceHaplotypes, error):
    nHaps, nLoci = sourceHaplotypes.shape
    pointMat = np.full((nLoci, nHaps), 1, dtype=np.float32)

    ### STUFF
    for i in range(nLoci):
        if targetHaplotype[i] != 9:
            for j in range(nHaps):
                if targetHaplotype[i] == sourceHaplotypes[j, i]:
                    pointMat[i, j] = 1 - error[i]
                else:
                    pointMat[i, j] = error[i]
    return pointMat


@jit(nopython=True)
def haploidTransformProbs(previous, new, estimate, point_estimate, recombination_rate):
    """Transforms a probability distribution (over haplotypes, at a single locus)
    to a probability distribution at the next locus by accounting for emission probabilities
    (point_estimates) and transition probabilities (recombination_rate)
    This is a core step in the forward and backward algorithms

    point_estimates     emission probabilities - (1D NumPy array)
    recombination_rate  recombination rate at this locus - (scalar)
    previous            probability distribution over haplotypes (hidden states) at the *previous* locus
    estimate            newly calculated probability distribution over haplotypes at *this* locus
    new                 intermediate probability distribution (passed in to this function for speed)

    Note: previous and estimate are updated by this function
    """
    n_haps = len(previous)

    # Get estimate at this locus and normalize
    new[:] = previous * point_estimate
    new /= np.sum(new)

    # Account for recombination rate
    e = recombination_rate
    e1 = 1-recombination_rate
    for j in range(n_haps):
        new[j] = new[j]*e1 + e/n_haps

    # Update distributions (in place)
    for j in range(n_haps):
        estimate[j] *= new[j]
        previous[j] = new[j]


@jit(nopython=True)
def haploidOneSample(forward_probs, recombination_rate):
    """Sample one haplotype (an individual) from the forward and backward probability distributions
    Returns two arrays:
      sample_indices   array of indices of haplotypes in the haplotype library at each locus
                       e.g. an individual composed of haplotypes 13 and 42 with 8 loci:
                       [42, 42, 42, 42, 42, 13, 13, 13]
    A description of the sampling process would be nice here..."""

    est = forward_probs.copy()  # copy so that forward_probs is not modified
    n_loci, n_haps = forward_probs.shape
    prev = np.ones(n_haps, dtype=np.float32)
    new = np.empty(n_haps, dtype=np.float32)

    # Sampled probability distribution at one locus
    sampled_probs = np.empty(n_haps, dtype=np.float32)
    sample_indices = np.empty(n_loci, dtype=np.int64)

    # Backwards algorithm
    for i in range(n_loci-2, -1, -1): # zero indexed then minus one since we skip the boundary
        # Sample at this locus
        j = NumbaUtils.multinomial_sample(pvals=est[i+1, :])
        sampled_probs[:] = 0
        sampled_probs[j] = 1
        sample_indices[i+1] = j

        # Get estimate at this locus using the *sampled* distribution
        # (instead of the point estimates/emission probabilities)
        haploidTransformProbs(prev, new, est[i, :], sampled_probs, recombination_rate[i+1])
        # No need to normalise at this locus as multinomial_sample()
        # handles un-normalized probabilities

    # Last sample (at the first locus)
    j = NumbaUtils.multinomial_sample(pvals=est[0, :])
    sample_indices[0] = j

    return sample_indices


@jit(nopython=True)
def haploidSampleHaplotype(forward_probs, haplotype_library, recombination_rate):
    """Sample one haplotype (an individual) from the forward and backward probability distributions
    Returns: a sampled haploytpe of length n_loci"""
    indices = haploidOneSample(forward_probs, recombination_rate)
    return haplotypeFromHaplotypeIndices(indices, haplotype_library)


@jit(nopython=True)
def haplotypeFromHaplotypeIndices(indices, haplotype_library):
    """Helper function that takes an array of indices (for each locus) that 'point' to rows
    in a haplotype library and extracts the alleles from the corresponding haplotypes 
    (in the library)
    Returns: a haplotype array of length n_loci"""

    n_loci = len(indices)
    haplotype = np.empty(n_loci, dtype=np.int8)
    for col_idx in range(n_loci):
        row_idx = indices[col_idx]
        haplotype[col_idx] = haplotype_library[row_idx, col_idx]
    return haplotype


@jit(nopython=True)
def haploidForward(point_estimate, recombination_rate):
    """Calculate (unnomalized) forward probabilities"""

    n_loci, n_haps = point_estimate.shape
    est = point_estimate.copy()
    prev = np.ones(n_haps, dtype=np.float32)
    new = np.empty(n_haps, dtype=np.float32)

    for i in range(1, n_loci):
        # Update estimates at this locus
        haploidTransformProbs(prev, new, est[i, :], point_estimate[i-1, :], recombination_rate[i])

    return est


@jit(nopython=True)
def haploidBackward(point_estimate, recombination_rate):
    """Calculate (unnomalized) backward probabilities"""

    n_loci, n_haps = point_estimate.shape
    est = np.ones_like(point_estimate, dtype=np.float32)
    prev = np.ones(n_haps, dtype=np.float32)
    new = np.empty(n_haps, dtype=np.float32)

    for i in range(n_loci-2, -1, -1):  # zero indexed then minus one since we skip the boundary
        # Update estimates at this locus
        haploidTransformProbs(prev, new, est[i, :], point_estimate[i+1, :], recombination_rate[i+1])

    return est


@jit(nopython=True)
def haploidForwardBackward(point_estimate, recombination_rate):
    """Calculate normalized state probabilities at each loci using the forward-backward algorithm"""

    est = (haploidForward(point_estimate, recombination_rate) *
           haploidBackward(point_estimate, recombination_rate))

    # Return normalized probabilities
    n_loci = point_estimate.shape[0]
    for i in range(n_loci):
        est[i, :] /= np.sum(est[i, :])

    return est
