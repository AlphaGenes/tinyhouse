Dear John,

If you are reading this, I probably got hit by a bus. I wanted to leave some notes for whatever poor post-doc has been given the task of understanding the software I wrote. This might also serve as an introductory document for new hires to our python code base.

There’s a lot in here. I’m going to go through three main topics. First: some general philosophy and gatchas about writing code in python, and using tinyhouse as a library. Second: a bit about what each of the files in tinyhouse contains. This is not a full exauhstive list though. It will also include areas where I hope we’ll improve in the (near) future. Last it will include a brief overview of the software that depends on tinyhouse. The goal to give some idea of how it is currently being used.

Some overarching ideas
======

Python
------

When I started writing python code, most of the group was using a mix of Fortran and R for their programming needs. This had some issues, but seemed to work at the time. We still have a fairly large codebase written in Fortran, and it’s likely that most of those programs will stay that way. 

In particular, I wrote first version of AlphaPeel was written in Fortran, and learned a lot through that experience. One of the key things though was that we spent an awful lot of time writing code for some fairly simple things. Unlike most modern programming langauges, we had to code up our own version of common datastructures, liked linked lists, or dictionaries. We also had to write custom code for common operations like searching and sorting. This would be fine, but given how Fortran is structured as a language, it can take considerable development time to adapt existing code to new problems or new types.

I was finding that it was limiting the type of development I was able to do, and restricting the types of algorithms we could create and try. In short, our choice of programming language was actively hampering our ability to develop software and do research. So we took a look at python instead.

There are a lot of advantages for python. The language is rapidly developing, has a massive user base, there are a lot of third party modules available. It’s also easy to learn, easy to develop, and easy to maintain. The availibility of common libraries, means that we don’t need to worry about creating linked lists (or lists) for common objects, we have them build in. We also don’t need to re-write common functions like search functions, since they were already implemented. 

The main disasdvantage was that we already had a lot of code written in Fortran, and there was a worry that speed was going to be an issue. I think that in a lot of cases, speed has been fairly overlooked or minimized as something that people care about. Computers have gotten faster, but algorithms and common problems haven’t gotten substantially more complex. That is not the case for our group. We burn a disturbingly large amount of CPU time every month, and while Eddie is somewhat free, as a group we’ve invested quite a lot of money to obtain the resources we have, so it makes sense to use them wisely.

To get around issues regarding speed, we’ve taken advantage of using numpy and numba, two high performance computing libraries for python. Numpy is a matrix library, numba is a just in time compiler that allows compiled code using a subset of the core python language, along with some numpy features. In most cases, utilizing numba/numpy doesn’t require changing any code. In numba’s case a just in time decorator is added to functions, but sometimes it requires a little it of code refactoring.

To put it bluntly, I think concerns over speed for most of our use cases are somewhat over-rated. In many cases the types of things that in python are slowest (and can’t be sped up using numba), are also the types of things we don’t spend a whole lot of run time. I think sorting a pedigree is a good example of this. To sort a pedigree, we take a list of individuals, calculate the generation of each individual (the generation is defined as 0 if the individual doesn’t have any parents, and the max(sire_generation, dam_generation) + 1, otherwise). We then create a list of generations and assign an individual to the list containing other individuals of their same generation. This creates a “sorted” pedigree, and let’s us go from the earliest individual to the most recent, and guarantees that we will always process an individual’s parents before the individual. In python, this takes 10-15 lines of code to do. In Fortran it took hundreds, and was a big enough deal that we have an entire program devoted to doing this on our website (AlphaRecode). In terms of runtime though – sorting a pedigree with 200,000 individuals takes under a second and so any speed gain will be negligible compared to the cost of what ever post-processing happens on that pedigree. 

Numpy
=====

Most of the data we use in our python programs are stored as numpy arrays. Because we are also using numba, it is good to be explicit about the data types you use. E.g. if you want to allocate an empty array of size ten, use:

newArray = np.full(10, 0, dtype = np.float32)
newArray = np.full(10, 0, dtype = np.int8)

over 

newArray = np.full(10, 0)

In general we use the following types for the following data structures:

* float32: General purpose double. Most of the computations we make are not hugely influenced by floating point errors (particularly in AlphaImpute and AlphaPeel). It makes sense to store information like phenotypes and dosages as float32 over float64s.
* int64: general purpose integer. Also used for read counts. Int32 is potentially not big enough to handle future datasets (on things like size of the pedigree), but an int64 should be more than enough. The space shavings should be negligible, since most of our data is not stored this way.
* int8: We store genotype values and haplotype values as int8s. Under a traditional coding scheme, Genotypes can take a value 0, 1, or 2. Haplotypes will take a value of 0 or 1. Even for polyploid individuals, an int8 can still capture most common genotype values. For historical reasons, we use 9 as a missing value.

Some other notes:

* When indexing matracies that have nLoci elements, I tend to have the last column be the loci, e.g. if we have a matrix of genotypes for three individuals, I would use np.full((3, nLoci), 0, dtype = np.int8) over np.full((nLoci, 3), 0, dtype = np.int8).


Numba
==

As far as I can tell, Numba is pretty much just magic. The idea with numba is that if your function is mostly written in simple base python, you add on a decorator to the function and numba will compile it using a just in time (jit) compiler. This has the potential to give massive speed gains for commonly used functions, and is most of the reason why I think we can get away with using python long term.

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

In the code we use the flag “jit(nopython=True)” or “njit” to mark code for compilation. If just the “jit” flag is given, then the code may not be compiled (if it contains non-numba compatible elements). Using “nopython=True” or “njit” means that an error will be thrown instead.

The types of arguments that numba can accept is growing. Generally it can take most “common” items including numpy arrays. You can also write just-in-time classes. See the jit_family class in pedigree.py for an example.

Some quick notes
* Sometimes it is faster to loop than it is to use numpy. Numpy has some overhead when being called from numba. If the vector or matrix is small (under 100 elements) the explicit loop can be an order of magnitude faster. We do this a lot in AlphaPeel since we are working with a lot of vectors of length 4.
* Auxiliary data structures: Although many objects can be passed to numba, individuals and pedigrees cannot be. It may be possible to make them just in time compatible, but this would take a lot of work and would increase our ability to use and develop on them in the future. The speed gains tend not to be worth it either. To get around this issue, I’ve tended to either create “jit” versions of a class, which replaces things like integers with id numbers.  
* Alternatively I’ve created “information” objects which basically just create a large number of matricies which contain the data we need. In most cases these are “idn” indexed matrices, where each individual gets their own row. It’s a little bit weird that the information for an individual doesn’t live inside the individual object, but so far it has been an effective work around for getting things working.

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


