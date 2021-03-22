#include <vector>
#include <iostream>
#include <cinttypes>
#include <omp.h>
#include "_bootstrap.h"


void point_wise_sum(std::vector<int>& u, const std::vector<std::int8_t>& v)
{
    int size = u.size();
    for(int i = 0; i < size; i++) {
        u[i] += v[i];
    }
}


double event_uniqueness(const std::vector<int>& u, const std::vector<std::int8_t>& v)
{
    int size = u.size();
    std::vector<double> point_uniqueness;
    for(int i = 0; i < size; i++) {
        if(u[i] > 0 && v[i] > 0) {
            point_uniqueness.push_back(double(v[i]) / double(u[i]));
        }
    }

    double uniqueness = 0;
    size = point_uniqueness.size();
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


std::vector<double> sampled_event_uniqueness(
        int n,
        int num_of_events,
        int num_of_timestamps,
        int num_of_threads,
        std::int8_t* indicator,
        const std::vector<int>& sampled_events
)
{
    int n_sampled_events = n + 1;
    std::vector<double> avg_uniqueness (num_of_events, 0);
#pragma omp parallel for num_threads(num_of_threads)
    for(int i = 0; i < num_of_events; i++) {
        std::vector <int> _sampled_events (sampled_events.begin(), sampled_events.end());
        _sampled_events[n] = i;
        std::vector<int> sum_of_occurrences (num_of_timestamps, 0);
        std::vector<std::int8_t> event_occurrences (num_of_timestamps, 0);
        for(int j = 0; j < n_sampled_events; j++) {
            event_occurrences.assign(
                    indicator + _sampled_events[j] * num_of_timestamps,
                    indicator + (_sampled_events[j] + 1) * num_of_timestamps
            );
            point_wise_sum(sum_of_occurrences, event_occurrences);
        }
        avg_uniqueness[i] = event_uniqueness(sum_of_occurrences, event_occurrences);
    }
    return probabilities(avg_uniqueness);
}
