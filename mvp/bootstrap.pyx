# distutils: language = c++
# distutils: sources = _bootstrap.cpp

import cython
import numpy as np

cimport numpy as np
from libcpp.vector cimport vector


cdef extern from "_bootstrap.h":
    vector[double] probabilities_from_sampled_events(
            int n,
            int num_of_events,
            int num_of_threads,
            np.int32_t* indicator,
            vector[int] sampled_events
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def sequencial_bootstrap(
        np.ndarray[np.int32_t, ndim=2, mode="c"] horizon,
        int num_of_threads,
        int num_of_data=-1,
        object rng=None
):
    cdef int num_of_events = horizon.shape[0]
    cdef vector[int] sampled_events

    if num_of_data <= 0:
        num_of_data = num_of_events
    if rng is None:
        rng = np.random.default_rng(12345)
    for _ in range(num_of_data):
        sampled_events.push_back(0)

    for n in range(num_of_data):
        prob = probabilities_from_sampled_events(
                n,
                horizon.shape[0],
                num_of_threads,
                &horizon[0, 0],
                sampled_events
        )
        sampled_events[n] = rng.choice(range(num_of_events), p=prob)
    return sampled_events

