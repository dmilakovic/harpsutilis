#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:47:19 2018

@author: dmilakov
"""

from pathos.pools import ParallelPool as Pool
pool = Pool()


if __name__ == '__main__':
    def add(x, y, z):
        """Add three values"""
        return x + y + z

    def busybeaver(x):
        """This can take a while"""
        for num in range(1000000):
            x = x + num
        return x

    # Immediate evaluation example
    import time
    start = time.time()
    results = pool.map(busybeaver, range(10))
    print('Time to queue the jobs: %s' % (time.time() - start))
    start = time.time()
    # Casting the ppmap generator to a list forces each result to be
    # evaluated.  When done immediately after the jobs are submitted,
    # our program twiddles its thumbs while the work is finished.
    print(list(results))
    print('Time to get the results: %s' % (time.time() - start))

    # Delayed evaluation example
    start = time.time()
    results = pool.imap(busybeaver, range(10))
    print('Time to queue the jobs: %s' % (time.time() - start))
    # In contrast with the above example, this time we're submitting a
    # batch of jobs then going off to do more work while they're
    # processing.  Maybe "time.sleep" isn't the most exciting example,
    # but it illustrates the point that our main program can do work
    # before ppmap() is finished.  Imagine that you're submitting some
    # heavyweight image processing jobs at the beginning of your
    # program, going on to do other stuff like fetching more work to
    # do from a remote server, then coming back later to handle the
    # results.
    time.sleep(5)
    start = time.time()
    print(list(results))
    print('Time to get the first results: %s' % (time.time() - start))

    # Built-in map example
    print(list(map(add, [1, 2, 3], [4, 5, 6], [7, 8, 9])))

    # Trivial ppmap tests
    for i in range(10):
        print('-' * 30)
        start = time.time()
        print(pool.map(add, [1, 2, 3], [4, 5, 6], [7, 8, 9]))
        print('Iteration time: %s' % (time.time() - start))

    # Heavier ppmap tests
    for i in range(10):
        print('-' * 30)
        start = time.time()
        print(pool.map(busybeaver, range(10)))
        print('Iteration time: %s' % (time.time() - start))
        #%%
import multiprocessing as mp
from pathos.multiprocessing import Pool as Pool1
from pathos.pools import ParallelPool as Pool2
from pathos.parallel import ParallelPool as Pool3
import time

def square(x):  
    # calculate the square of the value of x
    return x*x

if __name__ == '__main__':

    dataset = range(0,10000)

    start_time = time.time()
    for d in dataset:
        square(d)
    print('test with no cores: %s seconds' %(time.time() - start_time))

    nCores = 6
    print('number of cores used: %s' %(nCores))  


    start_time = time.time()

    p = mp.Pool(nCores)
    p.map(square, dataset)

    # Close
    p.close()
    p.join()

    print('test with multiprocessing: %s seconds' %(time.time() - start_time))


    start_time = time.time()

    p = Pool1(nCores)
    p.map(square, dataset)

    # Close
    p.close()
    p.join()

    print('test with pathos multiprocessing: %s seconds' %(time.time() - start_time))


    start_time = time.time()

    p = Pool2(nCores)
    p.map(square, dataset)

    # Close
    p.close()
    p.join()

    print('test with pathos pools: %s seconds' %(time.time() - start_time))


    start_time = time.time()

    p = Pool3()
    p.ncpus = nCores
    p.map(square, dataset)

    # Close
    p.close()
    p.join()

    print('test with pathos parallel: %s seconds' %(time.time() - start_time))