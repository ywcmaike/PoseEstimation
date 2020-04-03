//
// Created by yuhailin.
//

#ifndef VLOCTIO_RANDOM_SAMPLER_H
#define VLOCTIO_RANDOM_SAMPLER_H

#include <vector>

#include "estimators/sampler.h"

namespace PE {

// Random sampler for RANSAC-based methods.
//
// Note that a separate sampler should be instantiated per thread.
class RandomSampler : public Sampler {
public:
    explicit RandomSampler(const size_t num_samples);

    void Initialize(const size_t total_num_samples) override;

    size_t MaxNumSamples() override;

    std::vector<size_t> Sample() override;

private:
    const size_t num_samples_;
    std::vector<size_t> sample_idxs_;
};

// Weighted random sampler for Weighted RANSAC-based methods
//
// Note that a discrete probability distribution should be given
class WeigthedRandomSampler : public Sampler {
public:
    explicit WeigthedRandomSampler(const size_t num_sample);

    void Initialize(const size_t total_num_samples) override;

    void SetProbability(const std::vector<double> &prob);

    size_t MaxNumSamples() override;

    std::vector<size_t> Sample() override;

private:
    std::vector<double> prob_;
    const size_t num_samples_;
};


// Random sampler for PROSAC (Progressive Sample Consensus), as described in:
//
//    "Matching with PROSAC - Progressive Sample Consensus".
//        Ondrej Chum and Matas, CVPR 2005.
//
// Note that a separate sampler should be instantiated per thread and that the
// data to be sampled from is assumed to be sorted according to the quality
// function in descending order, i.e., higher quality data is closer to the
// front of the list.
class ProgressiveSampler : public Sampler {
 public:
  explicit ProgressiveSampler(const size_t num_samples);

  void Initialize(const size_t total_num_samples) override;

  size_t MaxNumSamples() override;

  std::vector<size_t> Sample() override;

 private:
  const size_t num_samples_;
  size_t total_num_samples_;

  // The number of generated samples, i.e. the number of calls to `Sample`.
  size_t t_;
  size_t n_;

  // Variables defined in equation 3.
  double T_n_;
  double T_n_p_;
};

}

#endif //VLOCTIO_RANDOM_SAMPLER_H
