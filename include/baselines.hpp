#pragma once

#include <cstddef>
#include <random>
#include <vector>
#include <limits>
#include <stdexcept>

namespace mab {

class Policy {
public:
    virtual ~Policy() = default;
    virtual std::size_t select() = 0;
    virtual void update(std::size_t arm, double reward) = 0;
    virtual std::size_t num_arms() const = 0;
};

class RandomPolicy : public Policy {
public:
    RandomPolicy(std::size_t num_arms, std::uint64_t seed = 123)
        : num_arms_(num_arms), rng_(seed) {
        if (num_arms == 0) throw std::invalid_argument("num_arms must be > 0");
    }

    std::size_t select() override {
        std::uniform_int_distribution<std::size_t> dist(0, num_arms_ - 1);
        return dist(rng_);
    }

    void update(std::size_t, double) override {}

    std::size_t num_arms() const override { return num_arms_; }

private:
    std::size_t num_arms_;
    std::mt19937_64 rng_;
};

class EpsilonGreedyPolicy : public Policy {
public:
    EpsilonGreedyPolicy(std::size_t num_arms, double epsilon = 0.1, std::uint64_t seed = 456)
        : num_arms_(num_arms), epsilon_(epsilon), rng_(seed) {
        if (num_arms == 0) throw std::invalid_argument("num_arms must be > 0");
        if (epsilon < 0.0 || epsilon > 1.0) throw std::invalid_argument("epsilon must be in [0,1]");
        counts_.assign(num_arms, 0);
        means_.assign(num_arms, 0.0);
    }

    std::size_t select() override {
        for (std::size_t i = 0; i < num_arms_; ++i) {
            if (counts_[i] == 0) return i;
        }
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        if (uni(rng_) < epsilon_) {
            std::uniform_int_distribution<std::size_t> dist(0, num_arms_ - 1);
            return dist(rng_);
        }
        std::size_t best = 0;
        double best_mean = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < num_arms_; ++i) {
            if (means_[i] > best_mean) {
                best_mean = means_[i];
                best = i;
            }
        }
        return best;
    }

    void update(std::size_t arm, double reward) override {
        ++counts_[arm];
        const double delta = reward - means_[arm];
        means_[arm] += delta / static_cast<double>(counts_[arm]);
    }

    std::size_t num_arms() const override { return num_arms_; }

private:
    std::size_t num_arms_;
    double epsilon_;
    std::vector<std::size_t> counts_;
    std::vector<double> means_;
    std::mt19937_64 rng_;
};

}  // namespace mab
