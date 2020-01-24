import numpy as np
import numba
from numba import jit, int8, int64, boolean, deferred_type, optional, jitclass, float32, double
from collections import OrderedDict
from . import InputOutput
from . import ProbMath
from . import MultiThreadIO




class Family(object):
    """Family is a container for fullsib families"""
    def __init__(self, idn, sire, dam, offspring):
        self.idn = idn
        self.sire = sire
        self.dam = dam
        self.offspring = offspring
        self.generation = max(sire.generation, dam.generation) + 1

        # Add this family to both the sire and dam's family.
        self.sire.families.append(self)
        self.dam.families.append(self)

    def addChild(self, child) :
        self.offspring.append(child)

    def toJit(self):
        """Returns a just in time version of itself with Individuals replaced by id numbers"""
        offspring = np.array([child.idn for child in self.offspring])
        return jit_Family(self.idn, self.sire.idn, self.dam.idn, offspring)


spec = OrderedDict()
spec['idn'] = int64
spec['sire'] = int64
spec['dam'] = int64
spec['offspring'] = int64[:]

@jitclass(spec)
class jit_Family(object):
    def __init__(self, idn, sire, dam, offspring):
        self.idn = idn
        self.sire = sire
        self.dam = dam
        self.offspring = offspring

class Individual(object):
    
    def __init__(self, idx, idn) :

        self.genotypes = None
        self.haplotypes = None
        self.dosages = None

        self.imputed_genotypes = None
        self.imputed_haplotypes = None

        self.reads = None
        self.longReads = []

        # Do we use this?
        self.genotypeDosages = None
        self.haplotypeDosages = None
        self.hapOfOrigin = None

        # Info is here to provide other software to add in additional information to an Individual.
        self.info = None

        #For plant impute. Inbred is either DH or heavily selfed. Ancestors is historical source of the cross (may be more than 2 way so can't handle via pedigree).
        self.inbred = False
        self.imputationAncestors = [] #This is a list of lists. Either length 0, length 1 or length 2.
        self.selfingGeneration = None

        self.sire = None
        self.dam = None
        self.idx = idx # User inputed string identifier
        self.idn = idn # ID number assigned by the pedigree
        self.fileIndex = dict() # Position of an animal in each file when reading in. Used to make sure Input and Output order are the same.
        self.fileIndex["id"] = idx

        self.dummy = False

        self.offspring = []
        self.families = []

        self.sex = -1
        self.generation = None

        self.initHD = False
        # used in pythonHMM, but how should this best be coded when already have initHD? Should probably be set when data is read in,
        # but need high_density_threshold to be set in the pedigree first
        self.is_high_density = False

        self.genotypedFounderStatus = None #?
    
    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.idn)

    def subset(self, start, stop):
        # Take an individual and create an individual that just contains information on those markers.
        # It's okay if this wipes other information.

        new_ind = Individual(self.idx, self.idn)
        if self.genotypes is not None:
            new_ind.genotypes = self.genotypes[start:stop].copy() # Maybe could get away with not doing copies... doing them just to be safe.

        if self.haplotypes is not None:
            new_ind.haplotypes = (self.haplotypes[0][start:stop].copy(), self.haplotypes[1][start:stop].copy())

        if self.reads is not None:
            new_ind.reads = (self.reads[0][start:stop].copy(), self.reads[1][start:stop].copy())
        return new_ind

    def getPercentMissing(self):
        return np.mean(self.genotypes == 9)

    def getGeneration(self):

        if self.generation is not None : return self.generation

        if self.dam is None: 
            damGen = -1
        else:
            damGen = self.dam.getGeneration()
        if self.sire is None: 
            sireGen = -1
        else:
            sireGen = self.sire.getGeneration()

        self.generation = max(sireGen, damGen) + 1
        return self.generation


    def constructInfo(self, nLoci, genotypes = True,  haps = False, reads = False) :
        if genotypes and self.genotypes is None:
            self.genotypes = np.full(nLoci, 9, dtype = np.int8)
        
        if  haps and self.haplotypes is None:
            self.haplotypes = (np.full(nLoci, 9, dtype = np.int8), np.full(nLoci, 9, dtype = np.int8))

        if reads and self.reads is None:
            self.reads = (np.full(nLoci, 0, dtype = np.int64), np.full(nLoci, 0, dtype = np.int64))
    
    def isFounder(self):
        return (self.sire is None) and (self.dam is None)

    def getGenotypedFounderStatus(self):
        # options: 1: "GenotypedFounder", 0:"ChildOfNonGenotyped", 2:"ChildOfGenotyped"
        if self.genotypedFounderStatus is None:
            if self.isFounder() :
                if self.genotypes is None or np.all(self.genotypes == 9):
                    self.genotypedFounderStatus = 0 
                else:
                    self.genotypedFounderStatus = 1
            else:
                parentStatus = max(self.sire.getGenotypedFounderStatus(), self.sire.getGenotypedFounderStatus())
                if parentStatus > 0:
                    self.genotypedFounderStatus = 2
                else:
                    if self.genotypes is None or np.all(self.genotypes == 9):
                        self.genotypedFounderStatus = 0 
                    else:
                        self.genotypedFounderStatus = 1

        return self.genotypedFounderStatus
    def isGenotypedFounder(self):
        return (self.getGenotypedFounderStatus() == 1)

# Not sure of the code source: https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
# Slightly modified.
import re 
def sorted_nicely( l , key): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda k: [ convert(c) for c in re.split('([0-9]+)', str(key(k))) ] 
    return sorted(l, key = alphanum_key)


class Generation(object):

    def __init__(self, number):

        self.number = number

        self.families = []

        self.individuals = []
        self.sires = set()
        self.dams = set()
        self.parents = set()

    def add_individual(self, ind):
        self.individuals.append(ind)

    def add_family(self, fam):
        self.families.append(fam)
        self.sires.add(fam.sire)
        self.dams.add(fam.dam)
        self.parents.add(fam.sire)
        self.parents.add(fam.dam)
        # Note: Individuals are added seperately.



class Pedigree(object):
 
    def __init__(self, fileName = None, constructor = Individual):

        self.maxIdn = 0
        self.maxFam = 0

        self.individuals = dict()
        self.families = None
        self.constructor = constructor
        self.nGenerations = 0
        self.generations = None #List of lists

        self.truePed = None
        self.nLoci = 0
        
        self.startsnp = 0
        self.endsnp = self.nLoci

        self.referencePanel = [] #This should be an array of haplotypes. Or a dictionary?

        self.maf=None #Maf is the frequency of 2s.

        # Threshold that determines if an individual is high-density genotyped
        self.high_density_threshold = 0.9

        if fileName is not None:
            self.readInPedigree(fileName)

        self.args = None
        self.writeOrderList = None

    def subset(self, start, stop):
        new_pedigree = Pedigree(constructor = self.constructor)
        new_pedigree.nLoci = stop - start
        # Add all of the individuals.

        for ind in self:
            # Note: ind.subset strips away all of the family information.
            new_pedigree[ind.idx] = ind.subset(start, stop)

        for ind in self:
            if ind.sire is not None:
                new_ind = new_pedigree[ind.idx]
                new_sire = new_pedigree[ind.sire.idx]


                # Add individuals
                new_ind.sire = new_sire
                new_sire.offspring.append(new_ind)

            if ind.dam is not None:
                new_ind = new_pedigree[ind.idx]
                new_dam = new_pedigree[ind.dam.idx]

                # Add individuals
                new_ind.dam = new_dam
                new_dam.offspring.append(new_ind)
        return new_pedigree

    def merge(self, new_pedigree, start, stop):
        # This just merged genotype, haplotype, and dosage information (if availible).
        
        for ind in self:
            new_ind = new_pedigree[ind.idx]
            if new_ind.genotypes is not None:
                if ind.genotypes is None:
                    ind.genotypes = np.full(self.nLoci, 9, dtype = np.int8)
                ind.genotypes[start:stop] = new_ind.genotypes
            
            if new_ind.dosages is not None:
                if ind.dosages is None:
                    ind.dosages = np.full(self.nLoci, -1, dtype = np.float32)

                ind.dosages[start:stop] = new_ind.dosages
            
            if new_ind.haplotypes is not None:
                if ind.haplotypes is None:
                    ind.haplotypes = (np.full(self.nLoci, 9, dtype = np.int8), np.full(self.nLoci, 9, dtype = np.int8))

                ind.haplotypes[0][start:stop] = new_ind.haplotypes[0]
                ind.haplotypes[1][start:stop] = new_ind.haplotypes[1]



    def __len__(self):
        return len(self.individuals)


    def writeOrder(self):
        if self.writeOrderList is None:
            inds = [ind for ind in self if (not ind.dummy) and (self.args.writekey in ind.fileIndex)]
            self.writeOrderList = sorted_nicely(inds, key = lambda ind: ind.fileIndex[self.args.writekey])
            
            if not self.args.onlykeyed:
                indsNoDummyNoFileIndex = [ind for ind in self if (not ind.dummy) and (not self.args.writekey in ind.fileIndex)]
                self.writeOrderList.extend(sorted_nicely(indsNoDummyNoFileIndex, key = lambda ind: ind.idx))
                
                dummys = [ind for ind in self if ind.dummy]
                self.writeOrderList.extend(sorted_nicely(dummys, key = lambda ind: ind.idx))

        for ind in self.writeOrderList :
            yield (ind.idx, ind)

    def setMaf(self):
        """Calculate minor allele frequencies at each locus"""

        # The default values of 1 (maf) and 2 (counts) provide a sensible prior
        # For example, a locus where all individuals are missing, the MAF will be 0.5
        maf = np.full(self.nLoci, 1, dtype = np.float32)
        counts = np.full(self.nLoci, 2, dtype = np.float32)
        for ind in self.individuals.values():
            if ind.genotypes is not None:
                addIfNotMissing(maf, counts, ind.genotypes)
        
        self.maf = maf/counts

    def getMissingness(self):
        missingness = np.full(self.nLoci, 1, dtype = np.float32)

        counts = 0
        for ind in self.individuals.values():
            if ind.genotypes is not None:
                counts += 1
                addIfMissing(missingness, ind.genotypes)
        return missingness/counts


    def set_high_density(self):
        """Set whether each individual is high-density"""
        for individual in self:#.individuals.values():
            is_high_density = np.mean(individual.genotypes != 9) >= self.high_density_threshold
            if is_high_density:
                individual.is_high_density = True


    def fillIn(self, genotypes = True, haps = False, reads = False):

        for individual in self:
            individual.constructInfo(self.nLoci, genotypes = True, haps = haps, reads = reads)


    def __getitem__(self, key) :
        return self.individuals[key]

    def __setitem__(self, key, value):
        self.individuals[key] = value

    def __iter__(self) :
        if self.generations is None:
            self.setUpGenerations()
        for gen in self.generations:
            for ind in gen.individuals:
                yield ind

    def __reversed__(self) :
        if self.generations is None:
            self.setUpGenerations()
        for gen in reversed(self.generations):
            for ind in gen.individuals:
                yield ind

    # Generation code

    def setUpGenerations(self) :
        self.nGenerations = 0
        #We can't use a simple iterator over self here, becuase __iter__ calls this function.
        for idx, ind in self.individuals.items():
            gen = ind.getGeneration()
            self.nGenerations = max(gen, self.nGenerations)
        self.nGenerations += 1 #To account for generation 0. 

        self.generations = [Generation(i) for i in range(self.nGenerations)]

        for idx, ind in self.individuals.items():
            gen = ind.getGeneration()
            self.generations[gen].add_individual(ind)

    #This is really sloppy, but probably not important.
    def setupFamilies(self) :

        if self.generations is None:
            self.setUpGenerations()

        self.families = dict()
        for ind in self:
            if not ind.isFounder():
                parents = (ind.sire.idx, ind.dam.idx)
                if parents in self.families :
                    self.families[parents].addChild(ind)
                else:
                    self.families[parents] = Family(self.maxFam, ind.sire, ind.dam, [ind])
                    self.maxFam += 1

        
        for family in self.families.values():
            self.generations[family.generation].add_family(family)
    

    def getFamilies(self, rev = False) :
        if self.generations is None:
            self.setUpGenerations()
        if self.families is None:
            self.setupFamilies()

        gens = range(0, len(self.generations))
        if rev: gens = reversed(gens)

        for i in gens:
            for family in self.generations[i].families:
                yield family
 


    def getIndividual(self, idx) :
        if idx not in self.individuals:
            self.individuals[idx] = self.constructor(idx, self.maxIdn)
            self.maxIdn += 1
            self.generations = None
        return self.individuals[idx]

    def readInPedigree(self, fileName):
        with open(fileName) as f:
            lines = f.readlines()
        pedList = [line.split() for line in lines]
        self.readInPedigreeFromList(pedList)

    def readInPlantInfo(self, fileName):
        with open(fileName) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.split()
            idx = parts[0]; 

            if idx not in self.individuals:
                self.individuals[idx] = self.constructor(idx, self.maxIdn)
                self.maxIdn += 1

            ind = self.individuals[idx]
            if len(parts) > 1:
                if parts[1] == "DH" or parts[1] == "INBRED":
                    ind.inbred = True
                elif parts[1][0] == "S" :
                    ind.inbred = False
                    ind.selfingGeneration = int(parts[1][1:])
                else:
                    ind.inbred = False

            if len(parts) > 2:
                if "|" in line:
                    first, second = line.split("|")
                    self.addAncestors(ind, first.split()[2:])
                    self.addAncestors(ind, second.split())
                else:
                    self.addAncestors(ind, parts[2:])


    def addAncestors(self, ind, parts):
        ancestors = []
        for idx in parts:
            if idx not in self.individuals:
                self.individuals[idx] = self.constructor(idx, self.maxIdn)
                self.maxIdn += 1
            ancestor = self.individuals[idx]
            ancestors.append(ancestor)
        ind.imputationAncestors.append(ancestors)


    def readInPedigreeFromList(self, pedList):
        index = 0
        for parts in pedList :
            idx = parts[0]
            self.individuals[idx] = self.constructor(idx, self.maxIdn)
            self.maxIdn += 1
            self.individuals[idx].fileIndex['pedigree'] = index; index += 1

        for parts in pedList :
            idx = parts[0]
            if parts[1] == "0": parts[1] = None
            if parts[2] == "0": parts[2] = None
            
            if parts[1] is not None and parts[2] is None:
                parts[2] = "MotherOf"+parts[0]
            if parts[2] is not None and parts[1] is None:
                parts[1] = "FatherOf"+parts[0] 

            ind = self.individuals[parts[0]]
            
            if parts[1] is not None:
                if parts[1] not in self.individuals:
                    self.individuals[parts[1]] = self.constructor(parts[1], self.maxIdn)
                    self.maxIdn += 1
                    self.individuals[parts[1]].fileIndex['pedigree'] = index; index += 1
                    self.individuals[parts[1]].dummy=True

                sire = self.individuals[parts[1]]
                ind.sire = sire
                sire.offspring.append(ind)
                sire.sex = 0

            if parts[2] is not None:
                if parts[2] not in self.individuals:
                    self.individuals[parts[2]] = self.constructor(parts[2], self.maxIdn)
                    self.maxIdn += 1
                    self.individuals[parts[2]].fileIndex['pedigree'] = index; index += 1
                    self.individuals[parts[1]].dummy=True

                dam = self.individuals[parts[2]]
                ind.dam = dam
                dam.offspring.append(ind)
                dam.sex = 1

            # Optional fourth column contains sex OR inbred/outbred status
            if len(parts) > 3:
                male, female = {'m', '0', 'xy'}, {'f', '1', 'xx'}
                inbred, outbred = {'dh', 'inbred'}, {'outbred'}
                expected_entries = male | female | inbred | outbred
                if parts[3].lower() not in expected_entries:
                    raise ValueError(f"Unexpected entry in pedigree file, fourth field: '{parts[3]}'")
                # Sex
                if parts[3].lower() in male:
                    ind.sex = 0
                elif parts[3].lower() in female:
                    ind.sex = 1
                # Inbred/DH
                if parts[3].lower() in inbred:
                    ind.inbred = True
                elif parts[3].lower() in outbred:
                    ind.inbred = False


    def readInFromPlink(self, idList, pedList, bed, externalPedigree = False):
        index = 0

        if not externalPedigree:
            self.readInPedigreeFromList(pedList)
    
        for i, idx in enumerate(idList):
            genotypes=bed[:, i].copy() ##I think this is the right order. Doing the copy to be safe.
            nLoci = len(genotypes)
            if self.nLoci == 0:
                self.nLoci = nLoci
            if self.nLoci != nLoci:
                raise ValueError(f"Incorrect number of loci when reading in plink file. Expected {self.nLoci} got {nLoci}.")
            if idx not in self.individuals:
                self.individuals[idx] = self.constructor(idx, self.maxIdn)
                self.maxIdn += 1

            ind = self.individuals[idx]
            ind.constructInfo(nLoci, genotypes=True)
            ind.genotypes = genotypes

            ind.fileIndex['plink'] = index; index += 1

            if np.mean(genotypes == 9) < .1 :
                ind.initHD = True


    
    def check_line(self, id_data, idxExpected = None, ncol = None, getInd = True):
        idx, data = id_data

        if idxExpected is not None and idx != idxExpected:
            raise ValueError(f"Expected individual {idxExpected} but got individual {idx}")

        if ncol is None:
            ncol = len(data)
        if ncol != len(data):
            raise ValueError(f"Incorrect number of columns in {fileName}. Expected {ncol} values but got {len(data)} for individual {idx}.")

        
        nLoci = len(data)
        if self.nLoci == 0:
            self.nLoci = nLoci
        if self.nLoci != nLoci:
            raise ValueError(f"Incorrect number of values from {fileName}. Expected {self.nLoci} got {nLoci}.")

        ind = None
        if getInd :
            ind = self.getIndividual(idx)

        return ind, data, ncol 

    def readInGenotypes(self, fileName, startsnp=None, stopsnp = None):

        print("Reading in AlphaImpute Format:", fileName)
        index = 0
        ncol = None

        data_list = MultiThreadIO.readLines(fileName, startsnp = startsnp, stopsnp = stopsnp, dtype = np.int8)

        for value in data_list:
            ind, genotypes, ncol = self.check_line(value, idxExpected = None, ncol = ncol)

            ind.constructInfo(self.nLoci, genotypes=True)
            ind.genotypes = genotypes

            ind.fileIndex['genotypes'] = index; index += 1

            if np.mean(genotypes == 9) < .1 :
                ind.initHD = True

    def readInReferencePanel(self, fileName, startsnp=None, stopsnp = None):

        print("Reading in reference panel:", fileName)
        index = 0
        ncol = None

        data_list = MultiThreadIO.readLines(fileName, startsnp = startsnp, stopsnp = stopsnp, dtype = np.int8)

        for value in data_list:
            ind, haplotype, ncol = self.check_line(value, idxExpected = None, ncol = ncol, getInd=False)
            self.referencePanel.append(haplotype)

    def readInPhase(self, fileName, startsnp=None, stopsnp = None):
        print("Reading in phase data:", fileName)
        index = 0
        ncol = None

        data_list = MultiThreadIO.readLines(fileName, startsnp = startsnp, stopsnp = stopsnp, dtype = np.int8)

        e = 0
        currentInd = None
        for value in data_list:

            if e == 0: 
                idxExpected = None
            else:
                idxExpected = currentInd.idx

            ind, haplotype, ncol = self.check_line(value, idxExpected = idxExpected, ncol = ncol)
            currentInd = ind

            ind.constructInfo(self.nLoci, haps=True)
            ind.haplotypes[e][:] = haplotype
            e = 1-e

            ind.fileIndex['phase'] = index; index += 1

        
    def readInSequence(self, fileName, startsnp=None, stopsnp = None):
        index = 0
        ncol = None

        print("Reading in sequence data :", fileName)
        
        data_list = MultiThreadIO.readLines(fileName, startsnp = startsnp, stopsnp = stopsnp, dtype = np.int64)
        e = 0
        currentInd = None

        for value in data_list:
            if e == 0: 
                idxExpected = None
            else:
                idxExpected = currentInd.idx

            ind, reads, ncol = self.check_line(value, idxExpected = idxExpected, ncol = ncol)
            currentInd = ind

            ind.constructInfo(self.nLoci, reads=True)
            ind.fileIndex['sequence'] = index; index += 1

            ind.reads[e][:] = reads
            e = 1-e


    def callGenotypes(self, threshold):
        for idx, ind in self.writeOrder():
            matrix = ProbMath.getGenotypeProbabilities_ind(ind, InputOutput.args)
            
            matrixCollapsedHets = np.array([matrix[0,:], matrix[1,:] + matrix[2,:], matrix[3,:]], dtype=np.float32)
            calledGenotypes = np.argmax(matrixCollapsedHets, axis = 0)
            setMissing(calledGenotypes, matrixCollapsedHets, threshold)
            if InputOutput.args.sexchrom and ind.sex == 0:
                doubleIfNotMissing(calledGenotypes)
            ind.genotypes = calledGenotypes


    def writePedigree(self, outputFile):
        with open(outputFile, 'w+') as f:
            for ind in self:
                sire = "0"
                if ind.sire is not None:
                    sire = ind.sire.idx
                dam = "0"
                if ind.dam is not None:
                    dam = ind.dam.idx
                f.write(ind.idx + ' ' + sire + ' ' + dam + '\n')



    def writeGenotypes(self, outputFile):

        data_list = []
        for ind in self :
            data_list.append( (ind.idx, ind.genotypes) )

        MultiThreadIO.writeLines(outputFile, data_list, str)
        # with open(outputFile, 'w+') as f:
        #     for idx, ind in self.individuals.items():
        #         self.writeLine(f, ind.idx, ind.genotypes, str)

    def writePhase(self, outputFile):
        data_list = []
        for ind in self :
            data_list.append( (ind.idx, ind.haplotypes[0]) )
            data_list.append( (ind.idx, ind.haplotypes[1]) )

        MultiThreadIO.writeLines(outputFile, data_list, str)

        # with open(outputFile, 'w+') as f:
        #     for idx, ind in self.individuals.items():

        #         self.writeLine(f, ind.idx, ind.haplotypes[0], str)
        #         self.writeLine(f, ind.idx, ind.haplotypes[1], str)



    def writeDosages(self, outputFile):
        data_list = []
        for ind in self :
            if ind.dosages is not None:
                data_list.append( (ind.idx, ind.dosages) )
            else:
                dosages = ind.genotypes.copy()
                dosages[dosages == 9] = 1
                data_list.append( (ind.idx, dosages) )


        MultiThreadIO.writeLines(outputFile, data_list, "{:.4f}".format)

        # with open(outputFile, 'w+') as f:
        #     for idx, ind in self.individuals.items():
        #         if ind.dosages is not None:
        #             dosages = ind.dosages
        #         else: 
        #             dosages = ind.genotypes.copy()
        #             dosages[dosages == 9] = 1
        #         self.writeLine(f, ind.idx, dosages, "{:.4f}".format)


    def writeGenotypes_prefil(self, outputFile):
        # print("Output is using filled genotypes. Filling missing with a value of 1")
        # fillValues = np.full(1, self.nLoci)

        print("Output is using filled genotypes. Filling missing with rounded allele frequency")
        self.setMaf()
        fillValues = np.round(self.maf)

        with open(outputFile, 'w+') as f:
            for idx, ind in self.individuals.items():
                fill(ind.genotypes, fillValues)
                self.writeLine(f, ind.idx, ind.genotypes, str)

    def writeLine(self, f, idx, data, func) :
        f.write(idx + ' ' + ' '.join(map(func, data)) + '\n')



@jit(nopython=True)
def fill(genotypes, fillValue):
    for i in range(len(genotypes)):
        if genotypes[i] == 9:
            genotypes[i] = fillValue[i]

@jit(nopython=True)
def addIfNotMissing(array1, counts, array2):
    for i in range(len(array1)):
        if array2[i] != 9:
            array1[i] += array2[i]
            counts[i] += 2



@jit(nopython=True)
def addIfMissing(array1, array2):
    for i in range(len(array1)):
        if array2[i] == 9:
            array1[i] += 1

@jit(nopython=True)
def doubleIfNotMissing(calledGenotypes):
    nLoci = len(calledGenotypes)
    for i in range(nLoci):
        if calledGenotypes[i] == 1:
            calledGenotypes[i] = 2

@jit(nopython=True)
def setMissing(calledGenotypes, matrix, thresh) :
    nLoci = len(calledGenotypes)
    for i in range(nLoci):
        if matrix[calledGenotypes[i],i] < thresh:
            calledGenotypes[i] = 9















