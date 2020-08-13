import concurrent.futures
import itertools
import numpy as np
import math
from . import InputOutput

def convert_data_to_line(data_tuple, fmt) :
    idx, data = data_tuple
    return idx + ' ' + ' '.join(map(fmt, data)) + '\n'


def writeLines(fileName, data_list, fmt):
    print(f"Writing results to: {fileName}")
    try:
        iothreads = InputOutput.args.iothreads
    except AttributeError as error:
        iothreads = 1

    with open(fileName, 'w+') as f:

        if iothreads > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers = iothreads) as executor: # The minus one is to account for the main thread.
                # Break up into small-ish chunks to reduce overall memory cost.
                # Hard code: splits into 1k individuals.
                # These get then split up into one chunk per thread.
                subsets = split_by(data_list, 1000)
                for subset in subsets:
                    for result in executor.map(convert_data_to_line, subset, itertools.repeat(fmt), chunksize=math.ceil(1000/iothreads)):
                        f.write(result)

        if iothreads <= 1:
            for data_tuple in data_list:
                result = convert_data_to_line(data_tuple, fmt)
                f.write(result)




def split_by(array, step):

    output = []
    i = 0
    while i*step < len(array):
        start = i*step
        stop = (i+1)*step
        output.append(array[start:stop])
        i += 1
    return output



def process_input_line(line, startsnp, stopsnp, dtype):
    parts = line.split(); 
    idx = parts[0]
    parts = parts[1:]

    if startsnp is not None :
        parts = parts[startsnp : stopsnp + 1] #Offset 1 for id and 2 for id + include stopsnp

    data=np.array([int(val) for val in parts], dtype = dtype)

    return (idx, data)


def process_input_line_plink(line, startsnp, stopsnp, dtype):
    """Proces a line from PLINK .ped file
    Fields:
    Family ID ('FID')
    Within-family ID ('IID'; cannot be '0')
    Within-family ID of father ('0' if father isn't in dataset)
    Within-family ID of mother ('0' if mother isn't in dataset)
    Sex code ('1' = male, '2' = female, '0' = unknown)
    Phenotype value ('1' = control, '2' = case, '-9'/'0'/non-numeric = missing data if case/control)
    ...genotypes

    Assume each genotype is space separated, e.g.: G C or A A, need to recode this to 0, 1, 2

    At present this extracts individual's identifier and genotypes
    TO DO:
    * handle missing (genotype 0)
    * check all loci are biallelic
    * possibly handle numeric (1, 2)
    """

    parts = line.split()
    idx = parts[1]  # Use within-family ID
    genotypes = parts[6:]  # Seventh field onwards contains the genotypes

    if startsnp is not None:
        genotypes = genotypes[startsnp*2: stopsnp*2 + 2]  # Each locus is represented by two alleles

    data = np.array(genotypes, dtype=np.bytes_)

    return (idx, data)


def readLines(fileName, startsnp, stopsnp, dtype):
    # print(f"Reading in file: {fileName}")

    try:
        iothreads = InputOutput.args.iothreads
    except AttributeError as error:
        iothreads = 1

    output = []
    with open(fileName) as f:

        if iothreads > 1:
            # This could be more efficient, but it's dwarfed by some of the other stuff in the program.
            # i.e. line is roughly the same size as the haplotypes (2 bytes per genotype value, i.e. (space)(value); and two values per haplotype.

            all_outputs = []
            lines = list(itertools.islice(f, 1000))
            while len(lines) > 0:
                with concurrent.futures.ProcessPoolExecutor(max_workers = iothreads) as executor:
                    chunk_output = executor.map(process_input_line, lines, itertools.repeat(startsnp), itertools.repeat(stopsnp), itertools.repeat(dtype), chunksize=math.ceil(1000/iothreads))
                all_outputs.append(chunk_output)
                lines = list(itertools.islice(f, 1000))
            output = itertools.chain.from_iterable(all_outputs)

        if iothreads <= 1:
            for line in f:
                output.append(process_input_line(line, startsnp = startsnp, stopsnp = stopsnp, dtype = dtype))

    return output

def readLinesPlinkPlainTxt(fileName, startsnp, stopsnp, dtype):
    # This is a complete duplicate of readLines, except it calls
    # Should remove the duplicated code somehow
    # process_input_line_plink() instead of readLines()

    try:
        iothreads = InputOutput.args.iothreads
    except AttributeError as error:
        iothreads = 1

    output = []
    with open(fileName) as f:

        if iothreads > 1:
            # This could be more efficient, but it's dwarfed by some of the other stuff in the program.
            # i.e. line is roughly the same size as the haplotypes (2 bytes per genotype value, i.e. (space)(value); and two values per haplotype.

            all_outputs = []
            lines = list(itertools.islice(f, 1000))
            while len(lines) > 0:
                with concurrent.futures.ProcessPoolExecutor(max_workers = iothreads) as executor:
                    chunk_output = executor.map(process_input_line_plink, lines, itertools.repeat(startsnp), itertools.repeat(stopsnp), itertools.repeat(dtype), chunksize=math.ceil(1000/iothreads))
                all_outputs.append(chunk_output)
                lines = list(itertools.islice(f, 1000))
            output = itertools.chain.from_iterable(all_outputs)

        if iothreads <= 1:
            for line in f:
                output.append(process_input_line_plink(line, startsnp = startsnp, stopsnp = stopsnp, dtype = dtype))

    return output


