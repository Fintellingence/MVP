#ifndef _BOOTSTRAP_H
#define _BOOTSTRAP_H

#include <cinttypes>
#include <vector>


std::vector<int> overlapped_idx(
        const std::vector<int>& event,
        const int start,
        const int end
);
std::vector<double> probabilities(const std::vector<double>& avg_uniqueness);
std::vector<double> probabilities_from_sampled_events(
        const int n,
        const int num_of_events,
        const int num_of_threads,
        const std::int32_t* horizon,
        const std::vector<int>& sampled_events
);
double event_avg_uniqueness(const std::vector<int>& sum);
void increment(std::vector<int>& sum, const std::vector<int>& range);

#endif
