import concurrent.futures
from . import InputOutput
import itertools


def convert_data_to_line(data_tuple, fmt) :
    idx, data = data_tuple
    return idx + ' ' + ' '.join(map(fmt, data)) + '\n'


def writeLines(fileName, data_list, fmt):
    print(f"Writing results to: {fileName}")

    with open(fileName, 'w+') as f:

        if InputOutput.args.iothreads > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers = InputOutput.args.iothreads) as executor: # The minus one is to account for the main thread.
                for result in executor.map(convert_data_to_line, data_list, itertools.repeat(fmt), chunksize=1000):
                    f.write(result)

        if InputOutput.args.iothreads <= 1:
            for data_tuple in data_list:
                result = convert_data_to_line(data_tuple, fmt)
                f.write(result)


