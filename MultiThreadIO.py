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
            lines = [line for line in f]
            with concurrent.futures.ProcessPoolExecutor(max_workers = iothreads) as executor:
                output = executor.map(process_input_line, lines, itertools.repeat(startsnp), itertools.repeat(stopsnp), itertools.repeat(dtype), chunksize=1000)

        if iothreads <= 1:
            for line in f:
                output.append(process_input_line(line, startsnp = startsnp, stopsnp = stopsnp, dtype = dtype))

    return output



