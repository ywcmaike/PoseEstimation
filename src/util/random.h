//
// Created by yuhailin.
//

#ifndef VLOCTIO_RANDOM_H
#define VLOCTIO_RANDOM_H

#include <chrono>
#include <random>
#include <thread>
#include <unordered_set>

#include "util/logging.h"
#include "util/threading.h"

namespace PE {

extern thread_local std::mt19937* PRNG;

static const unsigned kRandomPRNGSeed = std::numeric_limits<unsigned>::max();

// Initialize the PRNG with the given seed.
//
// @param seed   The seed for the PRNG. If the seed is -1, the current time
//               is used as the seed.
void SetPRNGSeed(unsigned seed = kRandomPRNGSeed);

// Generate uniformly distributed random integer number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomInteger(const T min, const T max);

// Generate uniformly distributed random real number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomReal(const T min, const T max);

// Generate Gaussian distributed random real number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomGaussian(const T mean, const T stddev);

// Fisher-Yates shuffling.
//
// Note that the vector may not contain more values than UINT32_MAX. This
// restriction comes from the fact that the 32-bit version of the
// Mersenne Twister PRNG is significantly faster.
//
// @param elems            Vector of elements to shuffle.
// @param num_to_shuffle   Optional parameter, specifying the number of first
//                         N elements in the vector to shuffle.
template <typename T>
void Shuffle(const uint32_t num_to_shuffle, std::vector<T>* elems);

// Hailin Yu, Weigthed Random Sample
//
// This implementation is unbiased
template <typename T>
std::vector<size_t> WeightedRandomSample(const std::vector<T> &prob, const size_t size);

// Hailin Yu, Progressive Random Sample
//
// This implementation is unbiased
template <typename T>
void ProgressiveShuffle(const uint32_t num_to_shuffle, std::vector<T>* elems, const size_t inc_idx);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
T RandomInteger(const T min, const T max) {
    if (PRNG == nullptr) {
        SetPRNGSeed();
    }

    std::uniform_int_distribution<T> distribution(min, max);

    return distribution(*PRNG);
}

template <typename T>
T RandomReal(const T min, const T max) {
    if (PRNG == nullptr) {
        SetPRNGSeed();
    }

    std::uniform_real_distribution<T> distribution(min, max);

    return distribution(*PRNG);
}

template <typename T>
T RandomGaussian(const T mean, const T stddev) {
    if (PRNG == nullptr) {
        SetPRNGSeed();
    }

    std::normal_distribution<T> distribution(mean, stddev);
    return distribution(*PRNG);
}

template <typename T>
void Shuffle(const uint32_t num_to_shuffle, std::vector<T>* elems) {
    CHECK_LE(num_to_shuffle, elems->size());
    const uint32_t last_idx = static_cast<uint32_t>(elems->size() - 1);
    //std::cout << "Num to shuffle:" << num_to_shuffle << std::endl;
    for (uint32_t i = 0; i < num_to_shuffle; ++i) {
        const auto j = RandomInteger<uint32_t>(i, last_idx);
        std::swap((*elems)[i], (*elems)[j]);
    }
}

template <typename T>
std::vector<size_t> WeightedRandomSample(const std::vector<T> &prob, const size_t size) {
    CHECK_LE(size, prob.size());
    if (PRNG == nullptr) {
        SetPRNGSeed();
    }
    std::discrete_distribution<int> distribution(prob.begin(), prob.end());

    std::unordered_set<size_t> udset;
    while (udset.size() < size) {
        size_t number = static_cast<size_t>(distribution(*PRNG));
        udset.insert(number);
    }
    return std::vector<size_t>(udset.begin(), udset.end());
}

template <typename T>
void ProgressiveShuffle(const uint32_t num_to_shuffle, std::vector<T>* elems, const size_t inc_idx) {
    CHECK_LE(num_to_shuffle, inc_idx);
    const uint32_t last_idx = static_cast<uint32_t>(inc_idx - 1);
    //std::cout << "Num to shuffle:" << num_to_shuffle << std::endl;
    for (uint32_t i = 0; i < num_to_shuffle; ++i) {
        const auto j = RandomInteger<uint32_t>(i, last_idx);
        std::swap((*elems)[i], (*elems)[j]);
    }
}

}

#endif //VLOCTIO_RANDOM_H
