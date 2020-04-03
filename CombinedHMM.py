from numba import jit
import numpy as np
from . HaplotypeLibrary import haplotype_from_indices


class HaploidMarkovModel :
    def __init__(self, n_loci, error, recombination_rate = None):

        self.n_loci = n_loci

        if type(error) is float:
            self.error = np.full(n_loci, error, dtype=np.float32)
        if type(recombination_rate) is float:
            self.recombination_rate = np.full(n_loci, recombination_rate, dtype=np.float32)

        self.apply_smoothing = self.create_apply_smoothing()

    def get_mask(self, called_haplotypes):
        return np.all(called_haplotypes != 9, axis = 0)


    def get_genotype_probabilities(self, individual, haplotype_library):
        called_haplotypes = haplotype_library.get_called_haplotypes()
        haplotype_dosages = haplotype_library.get_haplotypes()

        mask = self.get_mask(called_haplotypes)     
        
        point_estimates = self.get_point_estimates(individual.genotypes, called_haplotypes, self.error, mask)
        total_probs = self.apply_smoothing(point_estimates, self.recombination_rate)
        genotype_probabilities = self.calculate_genotype_probabilities(total_probs, haplotype_dosages)
        return genotype_probabilities


    @staticmethod
    @jit(nopython=True, nogil=True)
    def get_point_estimates(genotypes, haplotypes, error, mask):
        nHap, nLoci = haplotypes.shape
        point_estimates = np.full((nLoci, nHap), 1, dtype = np.float32)
        for i in range(nLoci):
            if genotypes[i] != 9 and mask[i]:
                for j in range(nHap):
                    sourceGeno = haplotypes[j, i]
                    if 2*sourceGeno == genotypes[i]:
                        point_estimates[i, j] = 1-error[i]
                    else:
                        point_estimates[i, j] = error[i]

        return point_estimates

    @staticmethod
    @jit(nopython=True, nogil=True)
    def transmission(cummulative_probabilities, previous_point_probability, recombination_rate, output):
        output[:] = cummulative_probabilities * previous_point_probability
        normalize(output)
        output[:] *= (1-recombination_rate)
        output[:] += recombination_rate


    def create_apply_smoothing(self):

        transmission = self.transmission

        @jit(nopython=True, nogil=True)
        def directional_smoothing(point_estimate, recombination_rate, forward = False, backward = False):
            output = np.full(point_estimate.shape, 1, dtype = np.float32)
            n_loci = point_estimate.shape[0]

            if forward:
                start = 1
                stop = n_loci
                step = 1

            if backward:
                start = n_loci - 2
                stop = -1
                step = -1

            for i in range(start, stop, step):
                transmission(output[i-step,:], point_estimate[i - step,:], recombination_rate[i], output[i,:])

            return output

        @jit(nopython=True, nogil=True)
        def apply_smoothing(point_estimate, recombination_rate):
            """Calculate normalized state probabilities at each loci using the forward-backward algorithm"""

            est = ( point_estimate * 
                    directional_smoothing(point_estimate, recombination_rate, forward = True) *
                    directional_smoothing(point_estimate, recombination_rate, backward = True) )

            # Return normalized probabilities

            normalize_along_first_axis(est)
            return est

        return apply_smoothing


    @staticmethod        
    @jit(nopython=True, nogil=True)
    def calculate_genotype_probabilities(total_probs, reference_haplotypes) :
        n_hap, n_loci = reference_haplotypes.shape
        geno_probs = np.full((4, n_loci), 1, dtype = np.float32)

        for i in range(n_loci):
            for j in range(n_hap):
                hap_value = reference_haplotypes[j, i]
                prob_value = total_probs[i,j]
                if hap_value != 9:
                    # Add in a sum of total_probs values. 
                    geno_probs[0, i] += prob_value * (1-hap_value)
                    geno_probs[1, i] += 0
                    geno_probs[2, i] += 0
                    geno_probs[3, i] += prob_value * hap_value

        geno_probs = geno_probs/np.sum(geno_probs, axis = 0)
        return geno_probs


class DiploidMarkovModel(HaploidMarkovModel) :
    def __init__(self, n_loci, error, recombination_rate = None):
        HaploidMarkovModel.__init__(self, n_loci, error, recombination_rate)


    def get_genotype_probabilities(self, individual, paternal_haplotype_library, maternal_haplotype_library = None):

        seperate_reference_panels = (maternal_haplotype_library is None)

        if maternal_haplotype_library is None:
            maternal_haplotype_library = paternal_haplotype_library

        paternal_called_haplotypes = paternal_haplotype_library.get_called_haplotypes()
        paternal_haplotype_dosages = paternal_haplotype_library.get_haplotypes()

        maternal_called_haplotypes = maternal_haplotype_library.get_called_haplotypes()
        maternal_haplotype_dosages = maternal_haplotype_library.get_haplotypes()

        mask = self.get_mask(paternal_called_haplotypes) & self.get_mask(maternal_called_haplotypes) 
        
        point_estimates = self.get_point_estimates(individual.genotypes, paternal_called_haplotypes, maternal_called_haplotypes, self.error, mask)
        total_probs = self.apply_smoothing(point_estimates, self.recombination_rate)
        genotype_probabilities = self.calculate_genotype_probabilities(total_probs, paternal_haplotype_dosages, maternal_haplotype_dosages, seperate_reference_panels)
        
        return genotype_probabilities

    
    @staticmethod
    @jit(nopython=True, nogil=True)
    def calculate_genotype_probabilities(total_probs, paternal_haplotypes, maternal_haplotypes, seperate_reference_panels) :
        n_pat, n_loci = paternal_haplotypes.shape
        n_mat, n_loci = maternal_haplotypes.shape
        geno_probs = np.full((4, n_loci), 1, dtype = np.float32)

        for i in range(n_loci):
            for j in range(n_pat):
                for k in range(n_mat):
                    # diploid case where the markers are assumed independent.
                    if not seperate_reference_panels or j != k: 
                        pat_value = paternal_haplotypes[j, i]
                        mat_value = maternal_haplotypes[k, i]
                        prob_value = total_probs[i,j,k]
                        if pat_value != 9 and mat_value != 9:
                            # Add in a sum of total_probs values. 
                            geno_probs[0, i] += prob_value * (1-pat_value)*(1-mat_value)   #aa
                            geno_probs[1, i] += prob_value * (1-pat_value)*mat_value     #aA
                            geno_probs[2, i] += prob_value * pat_value*(1-mat_value)     #Aa
                            geno_probs[3, i] += prob_value * pat_value*mat_value       #AA

                    # Haploid case where the markers are not independent
                    else:
                        hap_value = paternal_haplotypes[j, i]
                        prob_value = total_probs[i,j,k]
                        if hap_value != 9:
                            geno_probs[0, i] += prob_value * (1-hap_value)
                            geno_probs[1, i] += 0
                            geno_probs[2, i] += 0
                            geno_probs[3, i] += prob_value * hap_value

        geno_probs = geno_probs/np.sum(geno_probs, axis = 0)
        return geno_probs


    @staticmethod
    @jit(nopython=True, nogil=True)
    def get_point_estimates(indGeno, paternalHaplotypes, maternalHaplotypes, error, mask):
        nPat, nLoci = paternalHaplotypes.shape
        nMat, nLoci = maternalHaplotypes.shape

        point_estimates = np.full((nLoci, nPat, nMat), 1, dtype = np.float32)

        for i in range(nLoci):
            if indGeno[i] != 9 and mask[i]:
                for j in range(nPat):
                    for k in range(nMat):
                        sourceGeno = paternalHaplotypes[j, i] + maternalHaplotypes[k, i]
                        if sourceGeno == indGeno[i]:
                            point_estimates[i, j, k] = 1-error[i]
                        else:
                            point_estimates[i, j, k] = error[i]

        return point_estimates
    
    
    @staticmethod
    @jit(nopython=True, nogil=True)
    def transmission(cummulative_probabilities, previous_point_probability, recombination_rate, output):
        output[:] = cummulative_probabilities * previous_point_probability
        normalize(output)

        row_sums = np.sum(output, 0)
        col_sums = np.sum(output, 1)

        output[:] *= (1 - recombination_rate)**2 # No recombination on either chromosome.
        output[:] += np.expand_dims(row_sums, 0)/output.shape[0]*recombination_rate*(1-recombination_rate) # recombination on the maternal (second) chromosome)
        output[:] += np.expand_dims(col_sums, 1)/output.shape[1]*recombination_rate*(1-recombination_rate) # recombination on the paternal (first) chromosome)
        output[:] += recombination_rate**2/output.size # double recombination


@jit(nopython=True, nogil=True)
def normalize(mat):
    mat[:] /= np.sum(mat)    

@jit(nopython=True, nogil=True)
def normalize_along_first_axis(mat):
    for i in range(mat.shape[0]):
        normalize(mat[i,:])


