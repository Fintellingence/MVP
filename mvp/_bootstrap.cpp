#include <vector>
#include <iostream>
#include <cinttypes>
#include <omp.h>
#include "_bootstrap.h"


void increment(std::vector<int>& sum, const std::vector<int>& range)
{
    for(int i = range[0]; i < range[1]; i++) {
        sum[i] += 1;
    }
}


double event_avg_uniqueness(const std::vector<int>& sum)
{
    int size = sum.size();
    double uniqueness = 0;
    std::vector<double> point_uniqueness (size, 1.0);
    for(int i = 0; i < size; i++) {
        point_uniqueness[i] /= double(sum[i]);
    }
    for(int i = 0; i < size; i++) {
        uniqueness += point_uniqueness[i];
    }
    return uniqueness / size;
}


std::vector<double> probabilities(const std::vector<double>& avg_uniqueness)
{
    double sum = 0;
    std::vector<double> probs;
    
    for(auto avg: avg_uniqueness) {
        sum += avg;
    }
    for(auto avg: avg_uniqueness) {
        probs.push_back(avg / sum);
    }
    return probs;
}


std::vector<int> overlapped_idx(
        const std::vector<int>& event,
        const int start,
        const int end
)
{
    std::vector<int> overlap (2, 0);
    if(start == event[0] && end == event[1]) {
        overlap[0] = 0;
        overlap[1] = event[1] - event[0] + 1; 
        return overlap;
    }
    if(start <= event[0]) {
        overlap[0] = 0;
    }
    else {
        overlap[0] = start - event[0];
    }
    if(end >= event[1]) {
        overlap[1] = event[1] - event[0] + 1;
    }
    else {
        overlap[1] = end - event[0] + 1;
    }
    return overlap;
}


std::vector<double> probabilities_from_sampled_events(
        const int n,
        const int num_of_events,
        const int num_of_threads,
        const std::int32_t* horizon,
        const std::vector<int>& sampled_events
)
{
    std::vector<double> avg_uniqueness (num_of_events, 0);
#pragma omp parallel for num_threads(num_of_threads)
    for(int i = 0; i < num_of_events; i++) {
        const std::vector<std::int32_t> event_of_interest (
                horizon + i * 2, horizon + (i + 1) * 2
        );
        std::vector<int> sum_of_occurrences (
                event_of_interest[1] - event_of_interest[0] + 1, 1
        );
        for(int j = 0; j < n; j++) {
            const int event = sampled_events[j];
            const int start = *(horizon + event * 2);
            const int end = *(horizon + event * 2 + 1);
            if(start <= event_of_interest[1] && end >= event_of_interest[0]) {
                const std::vector<int> overlap = overlapped_idx(event_of_interest, start, end);
                increment(sum_of_occurrences, overlap);
            }
        }
        avg_uniqueness[i] = event_avg_uniqueness(sum_of_occurrences);
    }
    return probabilities(avg_uniqueness);
}
