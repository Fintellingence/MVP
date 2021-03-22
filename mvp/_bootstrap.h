#ifndef _BOOTSTRAP_H
#define _BOOTSTRAP_H

#include <cinttypes>
#include <vector>


void point_wise_sum(std::vector<int>& u, const std::vector<std::int8_t>& v);
double event_uniqueness(const std::vector<int>& u, const std::vector<std::int8_t>& v);
std::vector<double> probabilities(const std::vector<double>& avg_uniqueness);
std::vector<double> sampled_event_uniqueness(
        int n,
        int num_of_events,
        int num_of_timestamps,
        int num_of_threads,
        std::int8_t* indicator,
        const std::vector<int>& sampled_events
);

#endif
