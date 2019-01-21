
import random
import numpy as np
import numba
from numba import njit, jit, int8, int32,int64, boolean, deferred_type, optional, jitclass, float32
from collections import OrderedDict

def profile(x): 
    return x

class HaplotypeLibrary(object) :
    def __init__(self) :
        self.library = []
        self.nHaps = 0
        # self.randGen = jit_RandomBinary(1000) #Hard coding for now. 1000 seemed reasonable.

    def append(self, hap):
        self.library.append(hap)
        self.nHaps = len(self.library)

    def removeMissingValues(self):
        for hap in self.library:
            removeMissingValues(hap)
    def asMatrix(self):
        return np.array(self.library)

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
