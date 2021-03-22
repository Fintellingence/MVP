# distutils: language = c++
# distutils: sources = _bootstrap.cpp

import cython
import numpy as np

cimport numpy as np
from libcpp.vector cimport vector


cdef extern from "_bootstrap.h":
    vector[double] sampled_event_uniqueness(
            int n,
            int num_of_events,
            int num_of_timestamps,
            int num_of_threads,
            np.int8_t* indicator,
            vector[int] sampled_events
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def sequencial_bootstrap(
        np.ndarray[np.int8_t, ndim=2, mode="fortran"] indicator,
        int num_of_threads,
        int num_of_data=-1,
        object rng=None
):
    cdef int num_of_events = indicator.shape[1]
    cdef vector[int] sampled_events
    cdef np.ndarray[np.int8_t, ndim=2, mode="c"] C_indicator = indicator.T

    if num_of_data <= 0:
        num_of_data = num_of_events
    if rng is None:
        rng = np.random.default_rng(12345)
    for _ in range(num_of_data):
        sampled_events.push_back(0)

    for n in range(num_of_data):
        prob = sampled_event_uniqueness(
                n,
                C_indicator.shape[0],
                C_indicator.shape[1],
                num_of_threads,
                &C_indicator[0, 0],
                sampled_events
        )
        sampled_events[n] = rng.choice(range(num_of_events), p=prob)
    return sampled_events

