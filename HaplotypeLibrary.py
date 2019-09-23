
import random
import numpy as np
import numba
from numba import njit, jit, int8, int32,int64, boolean, deferred_type, optional, jitclass, float32
from collections import OrderedDict, defaultdict

def profile(x):
    return x

class HaplotypeLibrary(object):
    """A library of haplotypes
    Each haplotype can have an identifier (any Python object, but typically a str or int)
    The identifiers are used to select haplotypes for updating or masking
    Haplotypes should be NumPy arrays of dtype np.int8

    Some functions only work on frozen libraries; some only on unfrozen ones. 
    Use freeze() and unfreeze() to swap between the two states. Typically a library is 
    built with append() and then frozen to enable additional functionality (sample(), masked(), etc.)"""

    # NOTES: 
    # Change to HaplotypeLibrary <= no 2
    
    
    def __init__(self, n_loci):
        self._n_loci = n_loci
        self._frozen = False
        self._haplotypes = []
        self._identifiers = []  # index to identifier mapping

    def __repr__(self):
        return repr(self._identifiers) + '\n' + repr(self._haplotypes)

    def __len__(self):
        """Number of haplotypes in the library"""
        return len(self._haplotypes)

    def append(self, haplotype, identifier=None):
        """Append a single haplotype to the library. 
        Note: a copy of the haplotype is taken"""
        if self._frozen:
            raise RuntimeError('Cannot append to frozen library')
        self._check_haplotype(haplotype, expected_shape=(self._n_loci,))
        self._identifiers.append(identifier)
        self._haplotypes.append(haplotype.copy())
        
    def freeze(self):
        """Freeze the library: convert identifier and haplotype lists to NumPy arrays"""
        if self._frozen:
            raise RuntimeError('Cannot freeze an already frozen library')
        self._haplotypes = np.array(self._haplotypes)
        self._identifiers = np.array(self._identifiers)
        self._frozen = True

    def unfreeze(self):
        """Unfreeze the library: convert identifiers and haplotypes to lists"""
        if not self._frozen:
            raise RuntimeError('Cannot unfreeze an unfrozen library')
        self._haplotypes = list(self._haplotypes)
        self._identifiers = list(self._identifiers)
        self._frozen = False
        
    def update_pair(self, paternal_haplotype, maternal_haplotype, identifier):
        """Update a pair of haplotypes"""
        if not self._frozen:
            raise RuntimeError('Cannot update an unfrozen library')
        self._check_identifier_exists(identifier)
        self._check_haplotype(paternal_haplotype, expected_shape=(self._n_loci,))
        self._check_haplotype(maternal_haplotype, expected_shape=(self._n_loci,))
        indices = self._indices(identifier)
        if len(indices) != 2:
            raise ValueError(f"Indentifer '{identifier}' does not have exactly two haplotypes in the library")
        self._haplotypes[indices] = np.vstack([paternal_haplotype, maternal_haplotype])

    def sample(self, n_haplotypes):
        """Return a randomly sampled HaplotypeLibrary() of n_haplotypes"""
        if not self._frozen:
            raise RuntimeError('Cannot sample an unfrozen library')
        if n_haplotypes > len(self):
            n_haplotypes = len(self)
        sampled_indices = np.sort(np.random.choice(len(self), size=n_haplotypes, replace=False))
        library = HaplotypeLibrary(self._n_loci)
        library._frozen = True
        library._haplotypes = self._haplotypes[sampled_indices]
        library._identifiers = self._identifiers[sampled_indices]
        return library

    def masked(self, identifier):
        """Returns a copy of all haplotypes *not* associated with an identifier
        (The copy is due the use of fancy indexing)
        If identifier is not in the library, then all haplotypes are returned"""
        if not self._frozen:
            raise RuntimeError('Cannot mask an unfrozen library')
        mask = (self._identifiers != identifier)
        return self._haplotypes[mask]

    def asMatrix(self):
        """Return the NumPy array - kept for backwards compatibility"""
        if self._frozen:
            return self._haplotypes.copy()
        return np.array(self._haplotypes)
    
    def removeMissingValues(self):
        """Replace missing values randomly with 0 or 1 with 50 % probability - kept for backwards compatibility"""
        for hap in self._haplotypes:
            removeMissingValues(hap)
    
    def _indices(self, identifier):
        """Get rows indices associated with an identifier. These can be used for fancy indexing"""
        if not self._frozen:
            raise RuntimeError('Cannot run _indices() on an unfrozen library')
        return  np.flatnonzero(self._identifiers == identifier).tolist()

    def _check_haplotype(self, haplotype, expected_shape):
        """Check haplotype has expected shape and dtype.
        Could extend to check values in {0,1,9}"""
        if haplotype.shape != expected_shape:
            raise ValueError('haplotype(s) has unexpected shape')
        if haplotype.dtype != np.int8:
            raise TypeError('haplotype(s) not dtype np.int8')

    def _check_identifier_exists(self, identifier):
        if identifier not in self._identifiers:
            raise KeyError(f"Identifier '{identifier}' not in library")
            
            
@njit
def removeMissingValues(hap):
    for i in range(len(hap)) :
        if hap[i] == 9:
            hap[i] = random.randint(0, 1)

class ErrorLibrary(object):
    @profile
    def __init__(self, hap, haplotypes):
        self.hap = hap
        self.hapLib = haplotypes
        self.errors = jit_assessErrors(hap, self.hapLib)

    def getWindowValue(self, k):
        return jit_getWindowValue(self.errors, k, self.hap)

@jit(nopython=True)
def jit_getWindowValue(errors, k, hap) :
    window = np.full(errors.shape, 0, dtype = np.int8)
    nHaps, nLoci = errors.shape
    #Let me be silly here.
    for i in range(k+1):
        window[:,0] += errors[:,i]

    for i in range(1, nLoci):
        window[:,i] = window[:,i-1]
        if i > k:
            if hap[i-k-1] != 9:
                window[:,i] -= errors[:,i-k-1] #This is no longer in the window.
        if i < (nLoci-k):
            if hap[i+k] != 9:
                window[:,i] += errors[:,i+k] #This is now included in the window.

    return window

@jit(nopython=True)
def jit_assessErrors(hap, haps):
    errors = np.full(haps.shape, 0, dtype = np.int8)
    nHaps, nLoci = haps.shape
    for i in range(nLoci):
        if hap[i] != 9:
            if hap[i] == 0:
                errors[:, i] = haps[:,i]
            if hap[i] == 1:
                errors[:, i] = 1-haps[:,i]
    return errors

from collections import OrderedDict
class HaplotypeDict(object):
    def __init__(self):
        self.nHaps = 0
        self.haps = []
        self.tree = dict()
    # @profile
    def append(self, haplotype):
        byteVal = haplotype.tobytes()
        if byteVal in self.tree:
            return self.tree[byteVal]
        else:
            self.tree[byteVal] = self.nHaps
            self.haps.append(haplotype)
            self.nHaps += 1
            return self.nHaps -1
        return self.tree[byteVal]

    def get(self, index):
        return self.haps[index]

# hap = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# hapLib = [np.array([1, 0, 0, 0, 0, 0, 1]),
#           np.array([0, 1, 0, 0, 0, 1, 0]),
#           np.array([0, 0, 1, 0, 1, 0, 0]),
#           np.array([0, 0, 0, 1, 0, 0, 0]),
#           np.array([0, 0, 1, 0, 1, 0, 0])]

# aa = ErrorLibrary(hap, hapLib)
# print(aa.errors)
# print(aa.getWindowValue(2))


# node_type = deferred_type()
# jit_randomBinary_spec = OrderedDict()
# jit_randomBinary_spec['array'] = int64[:]
# jit_randomBinary_spec['index'] = int64
# jit_randomBinary_spec['nItems'] = int64

# @jitclass(jit_randomBinary_spec)
# class jit_RandomBinary(object):
#     def __init__(self, nItems):
#         self.index = 0
#         self.nItems = nItems
#         self.array = np.random.randint(2, size = nItems)
#     def next():
#         self.index += 1
#         if self.index == self.nItems:
#             self.array = np.random.randint(2, size = nItems)
#             self.index = 0
#         return self.array[self.index]

# I Don't think this is used any more.
# def getCores(nLoci, lengths, offsets = [0]) :
#     nCores = len(lengths)*len(offsets)
#     startStop = []

#     for length in lengths:
#         for offset in offsets:
#             finished = False
#             if offset > 0:
#                 start = 0
#                 stop = min(offset, nLoci)
#                 startStop.append((start, stop))
#                 if stop == nLoci: finished = True
#             else:
#                 stop = 0
#             while not finished:
#                 start = stop
#                 stop = min(stop + length, nLoci)
#                 startStop.append((start, stop))
#                 if stop == nLoci: finished = True
#     return startStop

# from collections import OrderedDict
# class HaplotypeCount_dict(object):
#     def __init__(self):
#         self.nHaps = 0
#         self.haps = []
#         self.tree = OrderedDict()
#     # @profile
#     @profile
#     def append(self, haplotype, score = 1):
#         byteVal = haplotype.tobytes()
#         if byteVal in self.tree:
#             self.tree[byteVal] += score
#         else:
#             self.tree[byteVal] = score
#             self.haps.append(haplotype)
#             self.nHaps += 1

#     def getLargest(self):
#         #This is sloppy
#         vals = [value for key, value in self.tree.items()]

#         index = np.argmax(vals)
#         return self.haps[index]
