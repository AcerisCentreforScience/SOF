#pragma once

#include "baselines.hpp"

#include <cstddef>
#include <cmath>
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace mab {

class SlidingWindowUCB : public Policy {
public:
    SlidingWindowUCB(std::size_t num_arms, std::size_t window_size, std::uint64_t seed = 321)
        : num_arms_(num_arms), window_size_(window_size) {
        if (num_arms == 0) throw std::invalid_argument("num_arms must be > 0");
        if (window_size == 0) throw std::invalid_argument("window_size must be > 0");
        windows_.resize(num_arms);
        sums_.assign(num_arms, 0.0);
    }

    std::size_t select() override {
        for (std::size_t i = 0; i < num_arms_; ++i) {
            if (windows_[i].empty()) return i;
        }
        std::size_t best = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        const double log_term = std::log(static_cast<double>(std::min(t_, window_size_)) + 1.0);
        for (std::size_t i = 0; i < num_arms_; ++i) {
            const double n = static_cast<double>(windows_[i].size());
            const double mean = sums_[i] / n;
            const double score = mean + std::sqrt(2.0 * log_term / n);
            if (score > best_score) {
                best_score = score;
                best = i;
            }
        }
        return best;
    }

    void update(std::size_t arm, double reward) override {
        ++t_;
        windows_[arm].push_back(reward);
        sums_[arm] += reward;
        if (windows_[arm].size() > window_size_) {
            sums_[arm] -= windows_[arm].front();
            windows_[arm].pop_front();
        }
    }

    std::size_t num_arms() const override { return num_arms_; }

private:
    std::size_t num_arms_;
    std::size_t window_size_;
    std::vector<std::deque<double>> windows_;
    std::vector<double> sums_;
    std::size_t t_{0};
};

class RestartUCB : public Policy {
public:
    RestartUCB(std::size_t num_arms, std::size_t restart_interval, std::uint64_t seed = 654)
        : num_arms_(num_arms), restart_interval_(restart_interval), rng_(seed) {
        if (num_arms == 0) throw std::invalid_argument("num_arms must be > 0");
        if (restart_interval == 0) throw std::invalid_argument("restart_interval must be > 0");
        reset();
    }

    std::size_t select() override {
        if (t_ > 0 && t_ % restart_interval_ == 0) reset();
        for (std::size_t i = 0; i < num_arms_; ++i) {
            if (counts_[i] == 0) return i;
        }
        std::size_t best = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        const double log_term = std::log(static_cast<double>(t_in_epoch_ + 1));
        for (std::size_t i = 0; i < num_arms_; ++i) {
            const double n = static_cast<double>(counts_[i]);
            const double score = means_[i] + std::sqrt(2.0 * log_term / n);
            if (score > best_score) {
                best_score = score;
                best = i;
            }
        }
        return best;
    }

    void update(std::size_t arm, double reward) override {
        ++t_;
        ++t_in_epoch_;
        ++counts_[arm];
        const double delta = reward - means_[arm];
        means_[arm] += delta / static_cast<double>(counts_[arm]);
    }

    std::size_t num_arms() const override { return num_arms_; }

private:
    void reset() {
        counts_.assign(num_arms_, 0);
        means_.assign(num_arms_, 0.0);
        t_in_epoch_ = 0;
    }

    std::size_t num_arms_;
    std::size_t restart_interval_;
    std::vector<std::size_t> counts_;
    std::vector<double> means_;
    std::size_t t_{0};
    std::size_t t_in_epoch_{0};
    std::mt19937_64 rng_;
};

}  // namespace mab
