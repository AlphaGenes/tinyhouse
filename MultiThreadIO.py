import concurrent.futures
from . import InputOutput



def writeMultiThreadLines(fileName, indList, formatFunction):
    print(f"Writing results to: {fileName}")

    with open(fileName, 'w+') as f:

        if InputOutput.args.maxthreads > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers = InputOutput.args.maxthreads - 1) as executor: # The minus one is to account for the main thread.
                for result in executor.map(formatFunction, indList, chunksize=1000):
                    f.write(result)

        if InputOutput.args.maxthreads > 1:
            for ind in indList:
                result = formatFunction(ind)
                f.write(result)
