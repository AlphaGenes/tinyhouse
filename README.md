Hello,

Welcome to tinyhouse, an increasingly non light-weight python module for processing genotypes and pedigrees, and for performing common operations for livestock breeding settings. This document is designed to give a brief overview of this software package, including some over-arching design philosophies, and some specifics about programs that use our software.

This document covers three main topics.

1. Some general philosophy about writing code in python, and using tinyhouse as a library.
2. a bit about what each of the files in tinyhouse contains. This is not a full exhaustive list though. It will also include areas where improvement is needed (and hopefully will take place in the near future!).
3. It will include a brief overview of the software that depends on tinyhouse. The goal to give some idea of how it is currently being used, along with example programs that show how it might be used in the future.

Some overarching ideas.
======

Python
------

When I started writing this library, most of the group (myself included) was using Fortran as their primary coding language. This worked well because most of our legacy code in AlphaImpute and AlphaHouse was written in Fortran, and John knew and understood the language. As time went on, working with Fortran was creating more problems then it was solving, and so we started looking for other languages. Python was an obvious choice -- it is widely used both within and outside of academia. There are a large number of mature third party modules and libraries that can be taken advantage of. It is easy to learn, a joy to code in, and simple to deploy and maintain. It also provides (through `numpy` and `numba`) fast computational libraries that give C or Fortran like speed increases, with minimal overhead.

Many of these features stand in stark contrast to Fortran, which is no longer widely used, hard to code in, and lacks easy interfaces to common operations. Some simple data structures like linked lists and dictionaries are missing in Fortran, along with modern object-oriented programing principles like inheritance, and common operations like searching and sorting. Although it is possible (and AlphaImpute, AlphaPhase, and AlphaHouse impliment) these features, it takes a sizable number of developer-hours to do this, and porting them to different data structures is not always straightforward. These hurdles can make it expensive (in terms of developer-hours) to try out new ideas, or play around with new approaches and algorithms which hampered my ability to do research.

Computational speed has remained one of the main concerns about moving away from Fortran. This group uses a large amount of CPU time every month, and any computational savings can increase the throughput we have getting jobs through Eddie. In addition, some of the commercial partners we work with, need to be able to run our software in a short time frame (generally 24 hours on their data), and so an order of magnitude speed difference will matter.

To allow python to obtain Fortran level speeds for performing large scale matrix operations, much of the code is now parallelizable, and uses both `numpy` and `numba`. `numpy` is a commonly used matrix library in python. `numba` is a just in time (jit) compiler that is able to compile a subset of the core python language, along with some numpy features into faster code. In many cases, using `numba` and numpy requires little extra coding and provides competitive speed gains compared to Fortran. In some cases, working with `numba` can require some creative engineering, but the speed gains and convenience of using python more generally are worth it.

Most importantly, I believe that concerns over choosing a language for speed are generally misplaced. In an academic environment the limiting factor for developing fast code tends to be developer time. More developer time means that the developer can spend more time profiling the code, and refining the algorithms used. This will generally give larger speed increases then re-implementing the same algorithm in a (slightly) faster language. I believe that this is particularly the case for non-speed critical operations, where the convenience of having a pre-existing data structure, like a non-fixed length list, or a dictionary, greatly outweighs the marginal speed losses for using one.  

-Andrew Whalen
 
Numpy
---

`numpy` is a commonly used matrix library for python. It provides sensible matrix algebra, and because of the C/Fortran/MKL framework that underlies it, allows these operations to be very high-performance particularly for large matracies. It also interfaces well with `numba`. We use `numpy` arrays to store most of our genotypic data across our programs. 

Because `numba` is type sensitive, and much of tinyhouse was developed to be `numba` aware, we tend to be explicit about the matrix type when declaring a numpy array. Failing to be explicit about types can lead to downstream compilation errors (which are generally not very informative, and can be tricky to debug). For example, if you want to allocate an empty array with 10 elements, use:
```
newArray = np.full(10, 0, dtype = np.float32)
```
or 
```
newArray = np.full(10, 0, dtype = np.int8)
```
over 
```
newArray = np.full(10, 0)
```

As a general rule, use the following types for the following data structures:

* `float32`: General purpose floating point number. Most of the computations we make are not influenced by floating point errors (particularly in AlphaImpute and AlphaPeel), where the error rates or margins of uncertainty are much larger than the floating point precision. Because of this, it makes sense to store information like phenotypes, genotype probabilities, or genotype dosages as a `float32` over a `float64`.
* `int64`: general purpose integer. Also used for sequence data read counts. There is a concern that an `int32` is not large  enough to handle future datasets (e.g. for very large pedigrees), but an `int64` should be more than enough.
* `int8`: We store genotype values and haplotype values as an `int8`. Under a traditional coding scheme, Genotypes can take a value 0, 1, or 2. Haplotypes will take a value of 0 or 1. For historical reasons, we use 9 to denote a missing value. Potentially a more-informative missing value, like `NA` might be usable, but we would need to ensure `numba` and `numpy` compatibility. 

Some other notes:

* When indexing matrices that have `nLoci` elements, I tend to have the last column be the loci, e.g. if we have a matrix of genotypes for three individuals, Use 
```
np.full((3, nLoci), 0, dtype = np.int8)
``` 
over 
```
np.full((nLoci, 3), 0, dtype = np.int8).
```


numba
---

`numba` is a just in time (jit) compiler for python. The idea is that you take a normal python function, and add on a decorator[^1]. `numba` will then compile the function and give potentially massive speed gains. The existence and usability of `numba` is one of the key features that makes python a feasible language for us in the long term.

As an example, here is a function that adds two numbers together:

```
def add(a, b) :
    return a + b
add(1, 2)
```

Here is a jit compiled version of the same function:

```
from numba import jit 
@jit(nopython=True)
def add(a, b) :
    return a + b
add(1, 2)
```

In the code we use the decorator `@jit(nopython=True)` or `@njit` to compile a function. If the `@jit` flag is given without arguments, then the code may not be compiled, particularly if it contains non-`numba` compatible elements. Using the `nopython=True` arguments, or `njit` forces the function to be compiled. An error will be thrown if the function cannot be compiled. 

The types of arguments that `numba` can accept is growing. Generally it can take most base python objects including numpy arrays. You can also write just-in-time classes. See the `jit_family` class in `pedigree.py` for an example.

Some notes:

* Generally in base python it is faster to use a numpy vectorized operation (e.g., for matrix addition). In `numba` this is not always the case. If the vector or matrix is small (under 100 elements) it is usually faster to write the explicit for loop. The speed increase for the for loop can be an order of magnitude greater than the vectorized operation. This occurs a lot in AlphaPeel since we are working with genotype probabilities that are of length 4. 
* Auxiliary data structures: Although many objects can be passed to `numba`, our `individual` and `pedigree` objects cannot. Although it may be possible to make them `numba` compatible, I think this would take a lot of work and would decrease their usability in the long term. One way around this is to create wrappers around a `jit` function which take an individual (or set of individuals) and performs a common operation on e.g. their genotypes. Alternatively we can use `jit` versions of a class, where e.g., individuals are replaced with integer id numbers.
* For individual programs, it can often make sense to create "information" objects which are holders for a large number of matrices. These matrices are generally indexed by an individuals id number, `idn`. Unlike traditional pedigree objects, these information objects are usually much easier to turn into a `jit` class. It may seem a little bit weird that information on an individual is not directly linked to the individual object, but so far this has been an effective work around.

[^1]: Decorators are functions that get applied to a function and return a function. They are denoted with the `@` symbol. We use two common decorators, `numba`'s `@jit` and `kernprof`'s `@profile`. See, e.g., https://realpython.com/primer-on-python-decorators/

Parallelization 
---

There are a number of ways to parallelize python code. The most common are to use `numpy`'s internal parallelized functions, or to use `concurrent.futures.ThreadPoolExecutor` or `concurrent.futures.ProcessPoolExecutor`. `numpy`'s default parallelization is a good low-overhead option if most of the computational bottlenecks are large matrix operations. However it doesn't not always provide substantial improvements for complex matrix operations, and may not scale to a large number of processes. In contrast, both the `ThreadPoolExecutor` and `ProcessPoolExecutor` can parallelize arbitrary functions. The primary difference between them is:

**ThreadPoolExecutor** This executes the function across several **threads**. By default these threads share memory. However Python is still bound by the global interpreter lock, which prevents multiple threads executing their commands simultaneously. This means that `ThreadPoolExecutor` will generally not lead to any performance gains since only a single thread is active at a given time, but may allow for some asynchronous tasks to be performed with python. It is possible to get around the global interpreter lock with `numba`.

**ProcessPoolExecutor** This executes the functions across several python **processes**. These processes do not share memory, but have separate global interpreters. This means that you can get sizable speed gains by executing function calls across multiple processes, but may occur overhead for transferring data between processes. 

In our code we use both `ThreadPoolExecutor` and `ProcessPoolExecutor`, depending on what situation we are in. If we need to parallelize a large section of `jit` compiled `numba` code, we use a `ThreadPoolExecutor` and flag the function with `@jit(nopython=True, nogil=True)`. For example, see `Peeling.py` in `TinyPeel`. If we need to parallelize non-numba functions, we use a `ProcessPoolExecutor` instead. 

For more information about Python's global interpreter lock, see e.g., https://wiki.python.org/moin/GlobalInterpreterLock

Profiling
----

For profiling, Kernprof seems to do a pretty good job. It requires adding the `@profile` decorator to functions. This will cause errors when not running the profiler. Because of this, many of the packages have the following lines in the header:
```
try:
    profile
except:
    def profile(x):
        return x

```
These lines look to see if profile is a valid function (i.e. if currently running with kernprof), otherwise it turns `profile` into the identity function.

Style
===

We vaugley follow PEP8: https://www.python.org/dev/peps/pep-0008/

Some exceptions:

* camelCase is generally used for variable names.
* underscores (\_) are inconsistently used to denote `@jit` functions and classes. 

Folder layouts
==============

All of the python programs should follow a similar layout.  For example, here is the layout for the program "AlphaCall".

```
.
├── build_pipeline.sh
├── docs
├── example
│   ├── out.called.0.98
│   ├── reads.txt
│   └── runScript.sh
├── setup.py
├── src
│   └── alphacall
│       ├── AlphaCall-Convert.py
│       ├── AlphaCall.py
│       └── tinyhouse
│           ├── BasicHMM.py
│           ├── BurrowsWheelerLibrary.py
│           ├── HaplotypeLibrary.py
│           ├── HaplotypeOperations.py
│           ├── InputOutput.py
│           ├── Pedigree.py
│           ├── ProbMath.py
└── tests
    ├── basedata
    │   ├── reads.txt
    │   └── target.txt
    ├── checkResults.r
    ├── outputs
    │   └── out.called.0.98
    └── runTests.sh

```
In this folder, there are a number of mandatory files, and some optional files.

**build_pipeline.sh** This bash script should contain all of the commands required to build and install the python module. It may be as simple as `python setup.py -bdist_wheel`, or may contain some additional code to clean and maintain the code base. Making sure this is up to date is highly recommended, since it is easy to forget exactly what arguments need to be passed to `setup.py`.

**setup.py** This is a python file that contains instructions for how to compile the python program. For an example, see the `setup.py` in `AlphaCall`. There may be a broader explanation for how `setup.py` files work in the future.

**src** This folder contains all of the python source code for the project. A version of `tinyhouse` is likely also included here. In this folder there should be a sub-folder with the name of the package, e.g. `alphacall`. The main scripts should be included in this sub-folder. Due to the way python handles relative imports, there may be some scripts in `src` to enable running the code directly, without having to first install the program. These are there to help with debugging and testing functionality.

**tests** This folder should contain a set of tests to test the functionality of the program, including datasets required to run those tests (and potentially scripts to generate the data). The tests should be run-able with `./runTests.sh`, and should output success or failure. In the future there may be a more general way of running an entire suite of tests.

**example:** This folder should contain a simple example of the program. The example should run by calling `./runScript.sh`.

**docs:** This folder should contain all of the documentation for the project. If the documentation needs to be compiled, all of the files to compile it should be included. If there is a relevant paper, it may be a good idea to place it in here as well. 



Some files
==========

InputOutput.py
----

Need a lot of words here.

BasicHMM.py 
----

This module contains code to run a simple Li and Stephens style HMM based on a set of reference haplotypes. This was originally generated for TinyPlantImpute. There is a haploid and a diploid version. Both needs work. The primary functions are `haploidHMM` and `diploidHMM`. The haploid HMM takes in a single haplotype and a reference panel. The diploid HMM takes in an individual and a set of haplotype panels (for each sire/dam). There is an option to output either the called genotypes, dosages, or maximum likelihood (Viterbi) path, although not all of these algorithms are currently implemented. 

Pedigree.py 
----
This module contains functions for storing data for individuals in a structured way. There are three main classes, `Individual`, `Pedigree`, and `Family`. The `Indivdiual` class provides a space for storing data related to an individual. This includes common data structures like genotypes, haplotypes, or read counts, and relationships with other individuals like sires, dams, or offspring. A `Family` object is a container for a full sib family (a single sire and dam, and their offspring). A `Pedigree` object is the default class for holding individuals. The primary container is a dictionary that contains all of the individuals indexed by ID number. It also includes an iterator to iterate over individuals in "generation" order (which places offspring after their parents). A large portion of the input/output is handled here and should be separated out. 


ProbMath.py 
----
This module contains some probability math that is shared between programs. The key function is `getGenotypeProbabilities_ind` which takes in an individual and returns a `4 x nLoci` matrix of genotype probabilities based on the individuals genotype and read count data. This is also where the transmission matrix lives for AlphaPeel, AlphaAssign, and AlphaFamImpute.

HaplotypeOperations.py
----
This module contains some simple operations that can be performed on haplotypes.


BurrowsWheelerLibrary.py and HaplotypeLibrary.py
-----
These two modules include two different ways of constructing haplotype
libraries. These are solely used by AlphaImpute2, and will be better documented when AlphaImpute2 gets a bit more mature.

Some programs 
===
AlphaImpute2.0
AlphaPeel
AlphaCall
TinyPlantImpute
AlphaFamImpute


