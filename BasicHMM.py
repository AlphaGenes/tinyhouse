from numba import jit
import numpy as np


def haploidHMM(targetHaplotype, sourceHaplotypes, error, recombinationRate, threshold = 0.9, callingMethod = "dosages"):

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
def haploidForwardBackward(pointEst, recombinationRate) :
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


def diploidHMM(ind, paternalHaplotypes, maternalHaplotypes, error, recombinationRate, callingMethod = "dosages", I = 0): #I is inbreeding coefficient. Honestly this should go into plantimpute at some point.

    nLoci = len(ind.genotypes)

    # !!!! NEED TO MAKE SURE SOURCE HAPLOTYPES ARE ALL NON MISSING!!!
    if type(paternalHaplotypes) is list:
        paternalHaplotypes = np.array(paternalHaplotypes)

    if type(maternalHaplotypes) is list:
        maternalHaplotypes = np.array(maternalHaplotypes)
    
    if type(error) is float:
        error = np.full(nLoci, error, dtype = np.float32)

    if type(recombinationRate) is float:
        recombinationRate = np.full(nLoci, recombinationRate, dtype = np.float32)
    # !!!! May need to have marker specific error/recombination rates.


    ### Build haploid HMM. 
    ###Construct penetrance values

    pointEst = getDiploidPointEstimates(ind.genotypes, ind.haplotypes[0], ind.haplotypes[1], paternalHaplotypes, maternalHaplotypes, error)
    
    # if prior is not None:
    #     addDiploidPrior(pointEst, prior)

    ### Run forward-backward algorithm on penetrance values.

    hapEst = diploidForwardBackward(pointEst, recombinationRate, I=I)
    # for i in range(nLoci) :
    #     print(hapEst[:,:,i])
    # raise Exception()


    if callingMethod == "dosages" :
        dosages = getDiploidDosages(hapEst, paternalHaplotypes, maternalHaplotypes)
        return dosages

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
                dosages[i] += hapEst[j,k,i]*(paternalHaplotypes[j,i] + maternalHaplotypes[k,i])
    return dosages

@jit(nopython=True)
def getDiploidPointEstimates(indGeno, indPatHap, indMatHap, paternalHaplotypes, maternalHaplotypes, error):
    nPat, nLoci = paternalHaplotypes.shape
    nMat, nLoci = maternalHaplotypes.shape

    pointEst = np.full((nPat, nMat, nLoci), 1, dtype = np.float32)
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
                        pointEst[j,k,i] = value
                    else:
                        #I think this shouldn't be too horrible.
                        sourceGeno = paternalHaplotypes[j, i] + maternalHaplotypes[k,i]
                        if sourceGeno == indGeno[i]:
                            pointEst[j,k, i] = 1-error[i]*error[i]
                        else:
                            pointEst[j,k,i] = error[i]*error[i]
    return pointEst


@jit(nopython=True)
def diploidForwardBackward(pointEst, recombinationRate, I = 0) :
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
















