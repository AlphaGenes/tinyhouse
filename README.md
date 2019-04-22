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

Computational speed has remained one of the main concerns about moving away from Fortran. This group burns a disturbingly large amount of CPU time every month, and any computational savings can increase the throughput we have getting jobs through Eddie. In addition, some of the commercial partners we work with, need to be able to run our software in a short time frame (generally 24 hours on their data), and so an order of magnitude speed difference will matter.

To allow python to obtain Fortran level speeds for performing large scale matrix operations, much of the code is now parallelizable, and heavily utilizes both numpy and `numba`. Numpy is a commonly used matrix library in python. `numba` is a just in time (jit) compiler that is able to compile a subset of the core python language, along with some numpy features into faster code. In many cases, using `numba` and numpy requires little extra coding and provides competitive speed gains compared to Fortran. In some cases, working with `numba` can require some creative engineering, but the speed gains and convenience of using python more generally are worth it.

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

As an example, here is a function for addition

```
def add(a, b) :
    return a + b
add(1, 2)
```

and a just in time version of the function

```
from numba import njit 
@njit
def add(a, b) :
    return a + b
add(1, 2)
```

In the code we use the flag “jit(nopython=True)” or “njit” to mark code for compilation. If just the “jit” flag is given, then the code may not be compiled (if it contains non-`numba` compatible elements). Using “nopython=True” or “njit” means that an error will be thrown instead.

The types of arguments that `numba` can accept is growing. Generally it can take most “common” items including numpy arrays. You can also write just-in-time classes. See the jit_family class in pedigree.py for an example.

Some quick notes
* Sometimes it is faster to loop than it is to use numpy. Numpy has some overhead when being called from `numba`. If the vector or matrix is small (under 100 elements) the explicit loop can be an order of magnitude faster. We do this a lot in AlphaPeel since we are working with a lot of vectors of length 4.
* Auxiliary data structures: Although many objects can be passed to `numba`, individuals and pedigrees cannot be. It may be possible to make them just in time compatible, but this would take a lot of work and would increase our ability to use and develop on them in the future. The speed gains tend not to be worth it either. To get around this issue, I’ve tended to either create “jit” versions of a class, which replaces things like integers with id numbers.  
* Alternatively I’ve created “information” objects which basically just create a large number of matricies which contain the data we need. In most cases these are “idn” indexed matrices, where each individual gets their own row. It’s a little bit weird that the information for an individual doesn’t live inside the individual object, but so far it has been an effective work around for getting things working.

[^1]: Decorators are functions that get applied to a function and return a function. They are denoted with the `@` symbol. We use two common decorators, `numba`'s `@jit` and `kernprof`'s `@profile`. See, e.g., https://realpython.com/primer-on-python-decorators/

Style
===

We generally follow some mix of PEP8: https://www.python.org/dev/peps/pep-0008/

Folder layouts
==============

All of the python programs have the following folder layout.

Folder Layout.


Some files
==========

BasicHMM.py 
BurrowsWheelerLibrary.py
HaplotypeLibrary.py 
HaplotypeOperations.py  3.28 KB 
InputOutput.py
Pedigree.py
ProbMath.py

Some programs 
===
AlphaImpute2.0
AlphaPeel
AlphaCall
TinyPlantImpute
AlphaFamImpute


