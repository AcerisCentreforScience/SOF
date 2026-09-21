#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace mab {

class BanditEnvironment {
public:
    virtual ~BanditEnvironment() = default;

    virtual std::size_t num_arms() const = 0;
    virtual double pull(std::size_t arm) = 0;
    virtual double true_mean(std::size_t arm) const = 0;

    std::size_t best_arm() const {
        std::size_t best = 0;
        double best_mean = true_mean(0);
        for (std::size_t i = 1; i < num_arms(); ++i) {
            const double m = true_mean(i);
            if (m > best_mean) {
                best_mean = m;
                best = i;
            }
        }
        return best;
    }

    double best_mean() const { return true_mean(best_arm()); }
};

class BernoulliBandit : public BanditEnvironment {
public:
    BernoulliBandit(std::vector<double> probs, std::uint64_t seed = 7)
        : probs_(std::move(probs)), rng_(seed) {
        if (probs_.empty()) throw std::invalid_argument("probs must not be empty");
        for (double p : probs_) {
            if (p < 0.0 || p > 1.0) throw std::invalid_argument("probs must be in [0,1]");
        }
    }

    std::size_t num_arms() const override { return probs_.size(); }

    double pull(std::size_t arm) override {
        check(arm);
        std::bernoulli_distribution dist(probs_[arm]);
        return dist(rng_) ? 1.0 : 0.0;
    }

    double true_mean(std::size_t arm) const override {
        check(arm);
        return probs_[arm];
    }

private:
    void check(std::size_t arm) const {
        if (arm >= probs_.size()) throw std::out_of_range("arm index out of range");
    }

    std::vector<double> probs_;
    std::mt19937_64 rng_;
};

class GaussianBandit : public BanditEnvironment {
public:
    GaussianBandit(std::vector<double> means, double sigma = 1.0, std::uint64_t seed = 7)
        : means_(std::move(means)), sigma_(sigma), rng_(seed) {
        if (means_.empty()) throw std::invalid_argument("means must not be empty");
        if (sigma_ <= 0.0) throw std::invalid_argument("sigma must be > 0");
    }

    std::size_t num_arms() const override { return means_.size(); }

    double pull(std::size_t arm) override {
        check(arm);
        std::normal_distribution<double> dist(means_[arm], sigma_);
        return dist(rng_);
    }

    double true_mean(std::size_t arm) const override {
        check(arm);
        return means_[arm];
    }

private:
    void check(std::size_t arm) const {
        if (arm >= means_.size()) throw std::out_of_range("arm index out of range");
    }

    std::vector<double> means_;
    double sigma_;
    std::mt19937_64 rng_;
};

inline std::vector<double> make_spaced_means(std::size_t num_arms,
                                             double lo = 0.2,
                                             double hi = 0.8) {
    std::vector<double> means(num_arms);
    for (std::size_t i = 0; i < num_arms; ++i) {
        const double t = num_arms == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(num_arms - 1);
        means[i] = lo + t * (hi - lo);
    }
    std::random_shuffle(means.begin(), means.end());
    return means;
}

}  // namespace mab
