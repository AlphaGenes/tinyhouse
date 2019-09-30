from numba import jit
import numpy as np
from . import ProbMath
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
def getHaploidDosages(hapEst, sourceHaplotypes):
    nHaps, nLoci = hapEst.shape
    dosages = np.full(nLoci, 0, dtype = np.float32)
    for i in range(nLoci):
        for j in range(nHaps):
            dosages[i] += sourceHaplotypes[j, i]*hapEst[j,i]
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
def getHaploidPointEstimates(targetHaplotype, sourceHaplotypes, error) :
    nHaps, nLoci = sourceHaplotypes.shape
    pointMat = np.full((nHaps, nLoci), 1, dtype = np.float32)

    ### STUFF
    for i in range(nLoci) :
        if targetHaplotype[i] != 9 :
            for j in range(nHaps) :
                if targetHaplotype[i] == sourceHaplotypes[j, i]:
                    pointMat[j, i] = 1 - error[i]
                else:
                    pointMat[j, i] = error[i]
    return pointMat


@jit(nopython=True)
def haploidTransformProbs(previous, estimate, point_estimate, recombination_rate):
    """Transforms a probability distribution (over haplotypes, at a single locus) to a probability distribution at the next locus by 
    accounting for emission probabilities (point_estimates) and transition probabilities (recombination_rate) 
    This is a core step in the forward and backward algorithms
    
    point_estimates     emission probabilities - (1D NumPy array)
    recombination_rate  recombination rate at this locus - (scalar)
    previous            probability distribution over haplotypes (hidden states) at the *previous* locus - (1D NumPy array)
    estimate            newly calculated probability distribution over haplotypes at *this* locus - (1D NumPy array)
    
    Note: previous and estimate are updated by this function
    """      
    n_haps = len(previous)
    new = np.empty(n_haps, 0, dtype=np.float32)
    
    # Get estimate at this locus and normalize
    for j in range(n_haps):
        new[j] = previous[j]*point_estimate[j]        
    sum_j = 0
    for j in range(n_haps):
        sum_j += new[j]
    for j in range(n_haps):
        new[j] = new[j]/sum_j

    # Account for recombination rate
    e1 = 1-recombination_rate
    e = recombination_rate    
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
                       e.g. an individual composed of haplotypes 13 and 42 with 8 loci: [42, 42, 42, 42, 42, 13, 13, 13]

      sampled_probs    sampled probabilities (either 0 or 1) with shape (n_haps, n_loci)  - used to check
                       that the average of many samples converges to the forward_backward probability distribution

    A description of the sampling process would be nice here..."""
    
    est = forward_probs.copy()  # copy so that forward_probs is not modified
    n_haps, n_loci = forward_probs.shape
    prev = np.ones(n_haps, dtype=np.float32)

    sampled_probs = np.empty(forward_probs.shape, dtype=np.float32)
    sample_indices = np.empty(n_loci, dtype=np.int64)

    # Backwards algorithm
    for i in range(n_loci-2, -1, -1): # zero indexed then minus one since we skip the boundary
        # Sample at this locus
        sampled_probs[:, i+1] = NumbaUtils.multinomial_sample(pvals=est[:, i+1])
        sample_indices[i+1] = np.argmax(sampled_probs[:, i+1])
        
        # Get estimate at this locus using the *sampled* distribution (instead of the point estimates/emission probabilities)
        haploidTransformProbs(prev, est[:, i], sampled_probs[:, i+1], recombination_rate[i+1])
        
        # Normalise at this locus (so that sampling can happen next time round the loop)
        est[:, i] = est[:, i] / np.sum(est[:, i])
    
    # Last sample (at the first locus)
    sampled_probs[:, 0] = NumbaUtils.multinomial_sample(pvals=est[:, 0])
    sample_indices[0] = np.argmax(sampled_probs[:, 0])

    return sample_indices, sampled_probs


@jit(nopython=True)
def haploidSampler(forward_probs, haplotype_library, recombination_rate):
    """Sample one haplotype (an individual) from the forward and backward probability distributions
    Returns: a sampled haploytpe of length n_loci"""
    indices, _ = haploidOneSample(forward_probs, recombination_rate)
    return haplotypeFromHaplotypeIndices(indices, haplotype_library)


@jit(nopython=True)
def haplotypeFromHaplotypeIndices(indices, haplotype_library):
    """Helper function that takes an array of indices (for each locus) that 'point' to rows in a haplotype library and
    extracts the alleles from the corresponding haplotypes in the library
    Returns: a haplotype array of length n_loci"""

    n_loci = len(indices)
    haplotype = np.empty(n_loci, dtype=np.int8)
    for col_idx in range(n_loci):
        row_idx = indices[col_idx]
        haplotype[col_idx] = haplotype_library[row_idx, col_idx]
    return haplotype


@jit(nopython=True)
def haploidForward(point_estimate, recombination_rate):
    """Calculate forward probabilities"""

    n_haps, n_loci = point_estimate.shape
    est = point_estimate.copy()
    prev = np.ones(n_haps, dtype=np.float32)

    for i in range(1, n_loci):
        # Update estimates at this locus
        haploidTransformProbs(prev, est[:, i], point_estimate[:, i-1], recombination_rate[i])
        
    # Return normalized estimates 
    return est / np.sum(est, axis=0)


@jit(nopython=True)
def haploidBackward(point_estimate, recombination_rate):
    """Calculate backward probabilities"""
   
    n_haps, n_loci = point_estimate.shape
    est = np.ones_like(point_estimate, dtype=np.float32)
    prev = np.ones(n_haps, dtype=np.float32)  
    
    for i in range(n_loci-2, -1, -1):  # zero indexed then minus one since we skip the boundary
        # Update estimates at this locus
        haploidTransformProbs(prev, est[:, i], point_estimate[:, i+1], recombination_rate[i+1])
        
    # Return normalized estimates 
    return est / np.sum(est, axis=0)


@jit(nopython=True)
def haploidForwardBackward(point_estimate, recombination_rate):
    """Calculate state probabilities at each loci using the forward-backward algorithm"""

    est = haploidForward(point_estimate, recombination_rate) * haploidBackward(point_estimate, recombination_rate)
    
    # Return normalized probabilities
    return est / np.sum(est, axis=0)    
    

@jit(nopython=True)
def haploidForwardBackwardOriginal(pointEst, recombinationRate):
    """The original implementation of the forward-backward algorithm, temporarily kept for possible performance comparisons"""
    
    #This is probably way more fancy than it needs to be -- particularly it's low memory impact, but I think it works.
    nHaps, nLoci = pointEst.shape

    est = np.full(pointEst.shape, 1, dtype = np.float32)
    for i in range(nLoci):
        for j in range(nHaps):
            est[j,i] = pointEst[j,i]

    tmp = np.full(nHaps, 0, dtype = np.float32)
    new = np.full(nHaps, 0, dtype = np.float32)
    prev = np.full(nHaps, 1, dtype = np.float32)

    for i in range(1, nLoci):
        e1 = 1-recombinationRate[i]
        e = recombinationRate[i]
        #Although annoying, the loops here are much faster for small number of haplotypes.

        #Get estimate at this loci and normalize.
        for j in range(nHaps):
            tmp[j] = prev[j]*pointEst[j,i-1]        
        sum_j = 0
        for j in range(nHaps):
            sum_j += tmp[j]
        for j in range(nHaps):
            tmp[j] = tmp[j]/sum_j

        #Account for recombination rate
        for j in range(nHaps):
            new[j] = tmp[j]*e1 + e/nHaps

        #Add to est
        for j in range(nHaps):
            est[j, i] *= new[j]
        prev = new

    prev = np.full((nHaps), 1, dtype = np.float32)
    for i in range(nLoci-2, -1, -1): #zero indexed then minus one since we skip the boundary.
        e1 = 1-recombinationRate[i+1]
        e = recombinationRate[i+1]

        #Get estimate at this loci and normalize.
        for j in range(nHaps):
            tmp[j] = prev[j]*pointEst[j, i+1]
        sum_j = 0
        for j in range(nHaps):
            sum_j += tmp[j]
        for j in range(nHaps):
            tmp[j] = tmp[j]/sum_j

        #Account for recombination rate
        for j in range(nHaps):
            new[j] = tmp[j]*e1 + e/nHaps

        #Add to est
        for j in range(nHaps):
            est[j, i] *= new[j]
        prev = new

    for i in range(nLoci):
        sum_j = 0
        for j in range(nHaps):
            sum_j += est[j,i]
        for j in range(nHaps):
            est[j,i] = est[j,i]/sum_j

    return(est)


# targetHaplotype = np.array([0, 0, 9, 0, 9, 9, 1, 1], dtype = np.int8)

# sourceHaplotypes = [np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype = np.int8),
#                     np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype = np.int8)]


# print(haploidHMM(targetHaplotype, sourceHaplotypes, 0.01, 0.1, threshold = 0.9))


def diploidHMM(ind, paternalHaplotypes, maternalHaplotypes, error, recombinationRate, callingMethod = "dosages", useCalledHaps = True, includeGenoProbs = False): 

    nLoci = len(ind.genotypes)

    # !!!! NEED TO MAKE SURE SOURCE HAPLOTYPES ARE ALL NON MISSING!!!
    if type(paternalHaplotypes) is list or type(paternalHaplotypes) is tuple:
        paternalHaplotypes = np.array(paternalHaplotypes)

    if type(maternalHaplotypes) is list or type(maternalHaplotypes) is tuple:
        maternalHaplotypes = np.array(maternalHaplotypes)
    
    if type(error) is float:
        error = np.full(nLoci, error, dtype = np.float32)

    if type(recombinationRate) is float:
        recombinationRate = np.full(nLoci, recombinationRate, dtype = np.float32)
    # !!!! May need to have marker specific error/recombination rates.


    ### Build haploid HMM. 
    ###Construct penetrance values

    if useCalledHaps:
        pointEst = getDiploidPointEstimates(ind.genotypes, ind.haplotypes[0], ind.haplotypes[1], paternalHaplotypes, maternalHaplotypes, error)
    else:
        probs = ProbMath.getGenotypeProbabilities_ind(ind)
        pointEst = getDiploidPointEstimates_probs(probs, paternalHaplotypes, maternalHaplotypes, error)

    
    # if prior is not None:
    #     addDiploidPrior(pointEst, prior)

    ### Run forward-backward algorithm on penetrance values.

    hapEst = diploidForwardBackward(pointEst, recombinationRate)
    # for i in range(nLoci) :
    #     print(hapEst[:,:,i])
    # raise Exception()


    if callingMethod == "dosages" :
        dosages = getDiploidDosages(hapEst, paternalHaplotypes, maternalHaplotypes)
        ind.dosages = dosages

    if callingMethod == "probabilities" :
        values = getDiploidProbabilities(hapEst, paternalHaplotypes, maternalHaplotypes)
        ind.info = values

    if callingMethod == "callhaps":
        raise ValueError("callhaps not yet implimented.")
    if callingMethod == "viterbi" :
        raise ValueError("Viterbi not yet implimented.")

@jit(nopython=True)
def addDiploidPrior(pointEst, prior):
    nPat, nMat, nLoci = pointEst.shape
    for i in range(nLoci):
        for j in range(nPat):
            for k in range(nMat):
                pointEst[j,k,i] *= prior[j,k]

@jit(nopython=True)
def getDiploidDosages(hapEst, paternalHaplotypes, maternalHaplotypes):
    nPat, nLoci = paternalHaplotypes.shape
    nMat, nLoci = maternalHaplotypes.shape
    dosages = np.full(nLoci, 0, dtype = np.float32)
    for i in range(nLoci):
        for j in range(nPat):
            for k in range(nMat):
                dosages[i] += hapEst[i,j,k]*(paternalHaplotypes[j,i] + maternalHaplotypes[k,i])
    return dosages

@jit(nopython=True)
def getDiploidProbabilities(hapEst, paternalHaplotypes, maternalHaplotypes):
    nPat, nLoci = paternalHaplotypes.shape
    nMat, nLoci = maternalHaplotypes.shape
    probs = np.full((4, nLoci), 0, dtype = np.float32)
    for i in range(nLoci):
        for j in range(nPat):
            for k in range(nMat):
                if paternalHaplotypes[j, i] == 0 and maternalHaplotypes[k, i] == 0:
                    probs[0, i] += hapEst[j, k, i]

                if paternalHaplotypes[j, i] == 0 and maternalHaplotypes[k, i] == 1:
                    probs[1, i] += hapEst[j, k, i]

                if paternalHaplotypes[j, i] == 1 and maternalHaplotypes[k, i] == 0:
                    probs[2, i] += hapEst[j, k, i]

                if paternalHaplotypes[j, i] == 1 and maternalHaplotypes[k, i] == 1:
                    probs[3, i] += hapEst[j, k, i]
    return probs

@jit(nopython=True)
def getDiploidPointEstimates(indGeno, indPatHap, indMatHap, paternalHaplotypes, maternalHaplotypes, error):
    nPat, nLoci = paternalHaplotypes.shape
    nMat, nLoci = maternalHaplotypes.shape

    pointEst = np.full((nLoci, nPat, nMat), 1, dtype = np.float32)
    for i in range(nLoci):
        if indGeno[i] != 9:
            for j in range(nPat):
                for k in range(nMat):
                    # Seperate Phased vs non phased loci 
                    if indPatHap[i] != 9 and indMatHap[i] != 9:
                        value = 1
                        if indPatHap[i] == paternalHaplotypes[j, i]:
                            value *= (1-error[i])
                        else:
                            value *= error[i]
                        if indMatHap[i] == maternalHaplotypes[k, i]:
                            value *= (1-error[i])
                        else:
                            value *= error[i]
                        pointEst[i,j,k] = value
                    else:
                        #I think this shouldn't be too horrible.
                        sourceGeno = paternalHaplotypes[j, i] + maternalHaplotypes[k,i]
                        if sourceGeno == indGeno[i]:
                            pointEst[i,j,k,] = 1-error[i]*error[i]
                        else:
                            pointEst[i,j,k] = error[i]*error[i]
    return pointEst

@jit(nopython=True)
def getDiploidPointEstimates_geno(indGeno, paternalHaplotypes, maternalHaplotypes, error, pointEst):
    nPat, nLoci = paternalHaplotypes.shape
    nMat, nLoci = maternalHaplotypes.shape

    for i in range(nLoci):
        if indGeno[i] != 9:
            error_2 = error[i]*error[i]
            for j in range(nPat):
                for k in range(nMat):

                    #I think this shouldn't be too horrible.
                    sourceGeno = paternalHaplotypes[j, i] + maternalHaplotypes[k,i]
                    if sourceGeno == indGeno[i]:
                        pointEst[i,j,k,] = 1-error_2
                    else:
                        pointEst[i,j,k] = error_2
    return pointEst

@jit(nopython=True)
def getDiploidPointEstimates_probs(indProbs, paternalHaplotypes, maternalHaplotypes, error):
    nPat, nLoci = paternalHaplotypes.shape
    nMat, nLoci = maternalHaplotypes.shape

    pointEst = np.full((nPat, nMat, nLoci), 1, dtype = np.float32)
    for i in range(nLoci):
        for j in range(nPat):
            for k in range(nMat):
                # I'm just going to be super explicit here. 
                p_aa = indProbs[0, i]
                p_aA = indProbs[1, i]
                p_Aa = indProbs[2, i]
                p_AA = indProbs[3, i]
                e = error[i]
                if paternalHaplotypes[j,i] == 0 and maternalHaplotypes[k, i] == 0:
                    value = p_aa*(1-e)**2 + (p_aA + p_Aa)*e*(1-e) + p_AA*e**2

                if paternalHaplotypes[j,i] == 1 and maternalHaplotypes[k, i] == 0:
                    value = p_Aa*(1-e)**2 + (p_aa + p_AA)*e*(1-e) + p_aA*e**2

                if paternalHaplotypes[j,i] == 0 and maternalHaplotypes[k, i] == 1:
                    value = p_aA*(1-e)**2 + (p_aa + p_AA)*e*(1-e) + p_Aa*e**2

                if paternalHaplotypes[j,i] == 1 and maternalHaplotypes[k, i] == 1:
                    value = p_AA*(1-e)**2  + (p_aA + p_Aa)*e*(1-e) + p_aa*e**2

                pointEst[j,k,i] = value
    return pointEst


@jit(nopython=True)
def diploid_normalize(array):
    """Normalize a 'diploid' probability array in place
    The array should have shape: (# paternal haplotypes, # maternal haplotypes, # loci)
    The function normalizes such that the values at each locus sum to 1

    It's possible the accuracy could be improved by using a compensated summation algorithm, e.g.:
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    As it stands, running np.sum(array, axis=(0,1)) on the output of this function with an array
    of shape (200,200,1000) gives values that differ from 1 by ~1e-4
    Note also that Numba doesn't support axis=(0,1)"""

    n_pat, n_mat, n_loci = array.shape
    for i in range(n_loci):
        # Use a float64 accumulator to avoid losing precision
        sum_ = np.float64(0)
        for j in range(n_pat):
            for k in range(n_mat):
                sum_ += array[j,k,i]
        for j in range(n_pat):
            for k in range(n_mat):
                array[j,k,i] = array[j,k,i]/sum_


@jit(nopython=True)
def diploidTransformProbs(previous, estimate, point_estimate, recombination_rate, new, pat, mat):
    """Transforms a probability distribution (over pairs of paternal and maternal haplotypes, at a single locus)
    to a probability distribution at the next locus by accounting for emission probabilities (point_estimates)
    and transition probabilities (recombination_rate)

    This is a core step in the forward and backward algorithms

    point_estimates     emission probabilities
                        shape: (# paternal haplotypes, # maternal haplotypes)
    recombination_rate  recombination rate at this locus - scalar
    previous            probability distribution over pairs of haplotypes (hidden states) at the *previous* locus
                        shape: (# paternal haplotypes, # maternal haplotypes)
    estimate            newly calculated probability distribution over pairs of haplotypes at *this* locus
                        shape: (# paternal haplotypes, # maternal haplotypes)

    Note: previous and estimate are updated by this function"""

    n_pat, n_mat = point_estimate.shape

    # Get estimate at this locus and normalize
    # for j in range(n_pat):
    #     for k in range(n_mat):
    #         new[j, k] = previous[j, k]*point_estimate[j, k]
    new[:,:] = previous*point_estimate
    new /= np.sum(new)
    # sum_ = np.float64(0)
    # for j in range(n_pat):
    #     for k in range(n_mat):
    #         sum_ += new[j, k]
    # for j in range(n_pat):
    #     for k in range(n_mat):
    #         new[j, k] = new[j, k]/sum_

    # Get haplotype specific recombinations
    pat[:] = 0
    mat[:] = 0
    for j in range(n_pat):
        for k in range(n_mat):
            pat[j] += new[j, k]
            mat[k] += new[j, k]
    
    e = recombination_rate
    e1e = (1-e)*e
    e2m1 = (1-e)**2

    # Adding modifications to pat and mat to take into account number of haplotypes and recombination rate.
    pat *= e1e/n_pat
    mat *= e1e/n_mat

    e2 = e*e/(n_mat*n_pat)

    # Account for recombination rate
    for j in range(n_pat):
        for k in range(n_mat):
            new[j, k] = new[j, k]*e2m1 + pat[j] + mat[k] + e2

    # Update distributions (in place)
    for j in range(n_pat):
        for k in range(n_mat):
            estimate[j, k] *= new[j, k]
            previous[j, k] = new[j, k]


@jit(nopython=True)
def diploidForward(point_estimate, recombination_rate):
    """Calculate forward probabilities"""

    n_loci, n_pat, n_mat = point_estimate.shape
    est = point_estimate.copy()  # copy so that point_estimate is not modified
    prev = np.full((n_pat, n_mat), 0.25, dtype=np.float32)

    new = np.empty((n_pat, n_mat), dtype=np.float32)
    pat = np.empty(n_pat, dtype=np.float32)
    mat = np.empty(n_mat, dtype=np.float32)


    for i in range(1, n_loci):
        # Update estimates at this locus
        diploidTransformProbs(prev, est[i, :, :], point_estimate[i-1, :, :], recombination_rate[i], new, pat, mat)

    # diploidForward_normalize(est)
    return est


@jit(nopython=True)
def diploidBackward(point_estimate, recombination_rate):
    """Calculate backward probabilities"""

    n_loci, n_pat, n_mat = point_estimate.shape
    est = np.ones_like(point_estimate, dtype=np.float32)
    prev = np.full((n_pat, n_mat), 0.25, dtype=np.float32)
  
    new = np.empty((n_pat, n_mat), dtype=np.float32)
    pat = np.empty(n_pat, dtype=np.float32)
    mat = np.empty(n_mat, dtype=np.float32)

    for i in range(n_loci-2, -1, -1): # zero indexed then minus one since we skip the boundary
        # Update estimates at this locus
        diploidTransformProbs(prev, est[i, :, :], point_estimate[i+1, :, :], recombination_rate[i+1], new, pat, mat)

    # diploid_normalize(est)
    return est


@jit(nopython=True)
def diploidForwardBackward(point_estimate, recombination_rate):
    """Calculate state probabilities at each loci using the forward-backward algorithm"""

    est = diploidForward(point_estimate, recombination_rate) * diploidBackward(point_estimate, recombination_rate)

    # Return normalized probabilities
    # diploid_normalize(est)
 
    n_loci = est.shape[0]
    for i in range(n_loci):
        est[i,:,:]/=np.sum(est[i, :, :])

    return est


@jit(nopython=True)
def diploidForwardBackwardOrig(pointEst, recombinationRate) :
    """The original implementation of the forward-backward algorithm, temporarily kept for possible performance comparisons"""

    #This is probably way more fancy than it needs to be -- particularly it's low memory impact, but I think it works.
    nPat, nMat, nLoci = pointEst.shape

    est = np.full(pointEst.shape, 1, dtype = np.float32)
    for i in range(nLoci):
        for j in range(nPat):
            for k in range(nMat):
                est[j,k,i] = pointEst[j,k,i]

    tmp = np.full((nPat, nMat), 0, dtype = np.float32)
    new = np.full((nPat, nMat), 0, dtype = np.float32)
    prev = np.full((nPat, nMat), .25, dtype = np.float32)

    pat = np.full(nPat, 0, dtype = np.float32)
    mat = np.full(nMat, 0, dtype = np.float32)

    for i in range(1, nLoci):
        e = recombinationRate[i]
        e1e = e*(1-e)
        e2 = e*e
        e2m1 = (1-e)**2
        #Although annoying, the loops here are much faster for small number of haplotypes.

        #Get estimate at this loci and normalize.
        for j in range(nPat):
            for k in range(nMat):
                tmp[j,k] = prev[j,k]*pointEst[j,k,i-1]
        sum_j = 0
        for j in range(nPat):
            for k in range(nMat):
                sum_j += tmp[j, k]
        for j in range(nPat):
            for k in range(nMat):
                tmp[j,k] = tmp[j,k]/sum_j

        #Get haplotype specific recombinations
        pat[:] = 0
        mat[:] = 0
        for j in range(nPat):
            for k in range(nMat):
                pat[j] += tmp[j,k]
                mat[k] += tmp[j,k]

        #Account for recombination rate
        for j in range(nPat):
            for k in range(nMat):
                # new[j, k] = tmp[j, k]*e2m1 + e1e*pat[j] + e1e*mat[k] + e2
                new[j,k] = tmp[j, k]*e2m1 + e1e*pat[j]/nPat + e1e*mat[k]/nMat + e2/(nMat*nPat)
                # valInbred = (1-e)*tmp[j, k]
                # if j == k: valInbred += e/nPat
                # new[j,k] = (1-I)*valOutbred + I*valInbred

        #Add to est
        for j in range(nPat):
            for k in range(nMat):
                est[j,k,i] *= new[j, k]
        prev = new

    prev = np.full((nPat, nMat), 1, dtype = np.float32)
    for i in range(nLoci-2, -1, -1): #zero indexed then minus one since we skip the boundary.
        #I need better naming comditions.
        e = recombinationRate[i+1]
        e1e = (1-e)*e
        e2 = e*e
        e2m1 = (1-e)**2

        #Although annoying, the loops here are much faster for small number of haplotypes.

        #Get estimate at this loci and normalize.
        for j in range(nPat):
            for k in range(nMat):
                tmp[j,k] = prev[j,k]*pointEst[j,k,i+1]
        sum_j = 0
        for j in range(nPat):
            for k in range(nMat):
                sum_j += tmp[j, k]
        for j in range(nPat):
            for k in range(nMat):
                tmp[j,k] = tmp[j,k]/sum_j

        #Get haplotype specific recombinations
        pat[:] = 0
        mat[:] = 0
        for j in range(nPat):
            for k in range(nMat):
                pat[j] += tmp[j,k]
                mat[k] += tmp[j,k]

        #Account for recombination rate
        for j in range(nPat):
            for k in range(nMat):
                # new[j, k] = tmp[j, k]*e2m1 + e1e*pat[j] + e1e*mat[k] + e2
                new[j,k] = tmp[j, k]*e2m1 + e1e*pat[j]/nPat + e1e*mat[k]/nMat + e2/(nMat*nPat)
                # valInbred = (1-e)*tmp[j, k]
                # if j == k: valInbred += e/nPat
                # new[j,k] = (1-I)*valOutbred + I*valInbred

        #Add to est
        for j in range(nPat):
            for k in range(nMat):
                est[j,k,i] *= new[j, k]
        prev = new

    for i in range(nLoci):
        sum_j = 0
        for j in range(nPat):
            for k in range(nMat):
                sum_j += est[j,k,i]
        for j in range(nPat):
            for k in range(nMat):
                est[j,k,i] = est[j,k,i]/sum_j

    return(est)


@jit(nopython=True)
def diploidSampleHaplotypes(forward_probs, recombination_rate, paternal_haplotypes, maternal_haplotypes):
    """Sample a pair of paternal and maternal haplotypes from the forward and backward probability distributions 
    and paternal and maternal haplotype libraries.
    Returns:
      haplotypes      Pair of haplotypes as a 2D array of shape (2, n_loci)
    """

    n_loci = forward_probs.shape[0]
    haplotypes = np.full((2, n_loci), 9, dtype=np.int8)

    sampled_probs = diploidOneSample(forward_probs, recombination_rate)
    paternal_indices, maternal_indices = diploidIndices(sampled_probs)
    
    haplotypes[0] = haplotypeFromHaplotypeIndices(paternal_indices, paternal_haplotypes)
    haplotypes[1] = haplotypeFromHaplotypeIndices(maternal_indices, maternal_haplotypes)
    
    return haplotypes


@jit(nopython=True)
def diploidOneSample(forward_probs, recombination_rate):
    """Sample a pair of paternal and maternal haplotypes from the forward and backward probability distributions
    
    Returns:
      sampled_probs    sampled probabilities (exactly 0 or 1) with shape (# paternal haplotypes, # maternal haplotypes, # loci)
      Note: this can be a large array - less wasteful to return indices

    A description of the sampling process would be nice here..."""

    est = forward_probs.copy()  # copy so that forward_probs is not modified
    n_loci , n_pat, n_mat= forward_probs.shape
    prev = np.full((n_pat, n_mat), 0.25, dtype=np.float32)
    sampled_probs = np.empty(forward_probs.shape, dtype=np.float32)

    new = np.zeros((n_pat, n_mat), dtype=np.float32)
    pat = np.zeros(n_pat, dtype=np.float32)
    mat = np.zeros(n_mat, dtype=np.float32)


    # Backwards algorithm
    for i in range(n_loci-2, -1, -1): # zero indexed then minus one since we skip the boundary
        # Sample at this locus
        sampled_probs[i+1, :, :] = NumbaUtils.multinomial_sample(pvals=est[i+1, :, :])

        # Get estimate at this locus using the *sampled* distribution (instead of the point estimates/emission probabilities)
        diploidTransformProbs(prev, est[i, :, :], sampled_probs[i+1, :, :], recombination_rate[i+1], new, pat, mat)

        # Normalise at this locus (so that sampling can happen next time round the loop)
        est[i, :, :] /= np.sum(est[i, :, :])

    # Last sample (at the first locus)
    sampled_probs[0, :, :] = NumbaUtils.multinomial_sample(pvals=est[0, :, :])

    return sampled_probs


@jit(nopython=True)
def diploidIndices(sampled_probs):
    """Get paternal and maternal indices from sampled probability distribution
    Intended to be used with sampled probabilities as returned from diploidOneSample()"""
    
    n_loci, n_pat, n_mat = sampled_probs.shape
    paternal_indices = np.empty(n_loci, dtype=np.int64)
    maternal_indices = np.empty(n_loci, dtype=np.int64)
    
    eps = np.finfo(np.float32).eps
    for i in range(n_loci):
        for j in range(n_pat):
            for k in range(n_mat):
                # If sampled_probs[j, k, i] == 1
                # set indices array to values j and k
                if sampled_probs[i, j, k] > 1-eps: 
                    paternal_indices[i] = j
                    maternal_indices[i] = k
                    
    return paternal_indices, maternal_indices


# def print3D(mat):
#     nPat, nMat, nLoci = mat.shape
#     for j in range(nPat):
#         for k in range(nMat):
#             print(mat[j, k, :])
#         print("")

# targetGenotypes = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype = np.int8)
# targetHaplotypes = [np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype = np.int8),
#                     np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype = np.int8)]

# paternalHaplotypes = [np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype = np.int8),
#                     np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype = np.int8)]

# maternalHaplotypes = [np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype = np.int8)]

# class Individual(object) :
#     def __init__(self, genotypes, haplotypes):
#         self.genotypes = genotypes
#         self.haplotypes = targetHaplotypes

# ind = Individual(targetGenotypes, targetHaplotypes)
# print(diploidHMM(ind, paternalHaplotypes, maternalHaplotypes, 0.01, 0.1))
















