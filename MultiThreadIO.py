import concurrent.futures
from . import InputOutput
import itertools


def convert_data_to_line(data_tuple, fmt) :
    idx, data = data_tuple
    return idx + ' ' + ' '.join(map(fmt, data)) + '\n'


def writeLines(fileName, data_list, fmt):
    print(f"Writing results to: {fileName}")

    iothreads = InputOutput.args.iothreads

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

