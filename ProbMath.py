from numba import jit
import numpy as np

def getGenotypesFromMaf(maf) :
    nLoci = len(maf)
    mafGenotypes = np.full((4, nLoci), .25, dtype = np.float32)

    mafGenotypes[0,:] = (1-maf)**2
    mafGenotypes[1,:] = maf*(1-maf)
    mafGenotypes[2,:] = (1-maf)*maf
    mafGenotypes[3,:] = maf**2

    return mafGenotypes

def getGenotypeProbabilities_ind(ind, args):

    if ind.reads is not None:
        nLoci = len(ind.reads[0])
    if ind.genotypes is not None:
        nLoci = len(ind.genotypes)
    sexChromFlag = args.sexchrom and ind.sex == 0 #This is the sex chromosome and the individual is male.
    return getGenotypeProbabilities(nLoci, ind.genotypes, ind.reads, args.error, args.seqerror, sexChromFlag)


def getGenotypeProbabilities(nLoci, genotypes, reads, error = 0.01, seqError = 0.001, useSexChrom=False):
    vals = np.full((4, nLoci), .25)
    if type(error) is float:
        error = np.full(nLoci, error)
    if type(seqError) is float:
        seqError = np.full(nLoci, seqError)

    errorMat = generateErrorMat(error)

    if genotypes is not None:
        setGenoProbsFromGenotypes(genotypes, errorMat, vals)
        
    if reads is not None:
        seqError = seqError
        log1 = np.log(1-seqError)
        log2 = np.log(.5)
        loge = np.log(seqError)
        valSeq = np.array([log1*reads[0] + loge*reads[1],
                                log2*reads[0] + log2*reads[1],
                                log2*reads[0] + log2*reads[1],
                                log1*reads[1] + loge*reads[0]])
        maxVals = np.amax(valSeq, 0)
        valSeq = valSeq - maxVals
        valSeq = np.exp(valSeq)
        vals *= valSeq

    if useSexChrom:
        #Recode so we only care about the two homozygous states, but they are coded as 0, 1.
        vals[1,:] = vals[3,:]
        vals[2,:] = 0
        vals[3,:] = 0

    return vals/np.sum(vals,0)



@jit(nopython=True)
def setGenoProbsFromGenotypes(genotypes, errorMat, vals):
    nLoci = len(genotypes)
    for i in range(nLoci) :
        if genotypes[i] != 9:
            vals[:, i] = errorMat[genotypes[i], :, i]
def generateErrorMat(error) :
    errorMat = np.array([[1-error, error/2, error/2, error/2], 
                            [error/2, 1-error, 1-error, error/2],
                            [error/2, error/2, error/2, 1-error]], dtype = np.float32)
    errorMat = errorMat/np.sum(errorMat, 1)[:,None]
    return errorMat


def generateSegregationXXChrom(partial=False, e= 1e-06) :
    paternalTransmission = np.array([ [1, 1, 0, 0],[0, 0, 1, 1]])
    maternalTransmission = np.array([ [1, 0, 1, 0],[0, 1, 0, 1]])

    fatherAlleleCoding = np.array([0, 0, 1, 1])
    motherAlleleCoding = np.array([0, 1, 0, 1])

    # !                  fm  fm  fm  fm 
    # !segregationOrder: pp, pm, mp, mm
    
    segregationTensor = np.zeros((4, 4, 4, 4))
    for segregation in range(4):
        #Change so that father always passes on the maternal allele?
        if(segregation == 0) :
            father = maternalTransmission
            mother = paternalTransmission
        if(segregation == 1) :
            father = maternalTransmission
            mother = maternalTransmission
        if(segregation == 2) :
            father = maternalTransmission
            mother = paternalTransmission
        if(segregation == 3) :
            father = maternalTransmission
            mother = maternalTransmission

        # !alleles: aa, aA, Aa, AA
        for allele in range(4) :
            segregationTensor[:, :, allele, segregation] = np.outer(father[fatherAlleleCoding[allele]], mother[motherAlleleCoding[allele]])

    segregationTensor = segregationTensor*(1-e) + e/4 #trace has 4 times as many elements as it should since it has 4 internal reps.
    segregationTensor = segregationTensor.astype(np.float32)

    return(segregationTensor)


def generateSegregationXYChrom(partial=False, e= 1e-06) :
    paternalTransmission = np.array([ [1, 1, 0, 0],[0, 0, 1, 1]])
    maternalTransmission = np.array([ [1, 0, 1, 0],[0, 1, 0, 1]])

    motherAlleleCoding = np.array([0, 1, 0, 1])

    # !                  fm  fm  fm  fm 
    # !segregationOrder: pp, pm, mp, mm
    #They don't get anything from the father -- father is always 0
    segregationTensor = np.zeros((4, 4, 4, 4))
    for segregation in range(4):
        if(segregation == 0) :
            mother = paternalTransmission
        if(segregation == 1) :
            mother = maternalTransmission
        if(segregation == 2) :
            mother = paternalTransmission
        if(segregation == 3) :
            mother = maternalTransmission

        # !alleles: aa, aA, Aa, AA
        for allele in range(4) :
            for fatherAllele in range(4):
                segregationTensor[fatherAllele, :, allele, segregation] = mother[motherAlleleCoding[allele]]

    segregationTensor = segregationTensor*(1-e) + e/4 #trace has 4 times as many elements as it should since it has 4 internal reps.
    segregationTensor = segregationTensor.astype(np.float32)

    return(segregationTensor)



def generateSegregation(partial=False, e= 1e-06) :
    paternalTransmission = np.array([ [1, 1, 0, 0],[0, 0, 1, 1]])
    maternalTransmission = np.array([ [1, 0, 1, 0],[0, 1, 0, 1]])

    fatherAlleleCoding = np.array([0, 0, 1, 1])
    motherAlleleCoding = np.array([0, 1, 0, 1])

    # !                  fm  fm  fm  fm 
    # !segregationOrder: pp, pm, mp, mm
    
    segregationTensor = np.zeros((4, 4, 4, 4))
    for segregation in range(4):
        if(segregation == 0) :
            father = paternalTransmission
            mother = paternalTransmission
        if(segregation == 1) :
            father = paternalTransmission
            mother = maternalTransmission
        if(segregation == 2) :
            father = maternalTransmission
            mother = paternalTransmission
        if(segregation == 3) :
            father = maternalTransmission
            mother = maternalTransmission

        # !alleles: aa, aA, Aa, AA
        for allele in range(4) :
            segregationTensor[:, :, allele, segregation] = np.outer(father[fatherAlleleCoding[allele]], mother[motherAlleleCoding[allele]])

    if partial : segregationTensor = np.mean(segregationTensor, 3)

    segregationTensor = segregationTensor*(1-e) + e/4 #trace has 4 times as many elements as it should since it has 4 internal reps.
    segregationTensor = segregationTensor.astype(np.float32)
    return(segregationTensor)

# def generateErrorMat(error) :
#     # errorMat = np.array([[1-error*3/4, error/4, error/4, error/4], 
#     #                         [error/4, .5-error/4, .5-error/4, error/4],
#     #                         [error/4, error/4, error/4, 1-error*3/4]], dtype = np.float32)
#     errorMat = np.array([[1-error*2/3, error/3, error/3, error/3], 
#                             [error/3, 1-error*2/3, 1-error*2/3, error/3],
#                             [error/3, error/3, error/3, 1-error*2/3]], dtype = np.float32)
#     errorMat = errorMat/np.sum(errorMat, 1)[:,None]
#     return errorMat

## Not sure if below is ever used. 
# def generateTransmission(error) :
#     paternalTransmission = np.array([ [1-error, 1.-error, error, error],
#                                       [error, error, 1-error, 1-error]])
    
#     maternalTransmission = np.array([ [1.-error, error, 1.-error, error],
#                                         [error, 1-error, error, 1-error]] )
#     segregationTransmissionMatrix = np.zeros((4,4))
#     segregationTransmissionMatrix[0,:] = paternalTransmission[0,:]
#     segregationTransmissionMatrix[1,:] = paternalTransmission[0,:]
#     segregationTransmissionMatrix[2,:] = paternalTransmission[1,:]
#     segregationTransmissionMatrix[3,:] = paternalTransmission[1,:]

#     segregationTransmissionMatrix[:,0] = segregationTransmissionMatrix[:,0] * maternalTransmission[0,:]
#     segregationTransmissionMatrix[:,1] = segregationTransmissionMatrix[:,1] * maternalTransmission[1,:]
#     segregationTransmissionMatrix[:,2] = segregationTransmissionMatrix[:,2] * maternalTransmission[0,:]
#     segregationTransmissionMatrix[:,3] = segregationTransmissionMatrix[:,3] * maternalTransmission[1,:]

#     segregationTransmissionMatrix = segregationTransmissionMatrix.astype(np.float32)
#     return(segregationTransmissionMatrix)
