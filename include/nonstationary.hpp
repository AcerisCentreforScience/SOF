#pragma once

#include "environment.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mab {

class NonStationaryEnvironment : public BanditEnvironment {
public:
    virtual void set_round(std::size_t t) = 0;
};

class DriftingBandit : public NonStationaryEnvironment {
public:
    DriftingBandit(std::vector<double> base_means,
                   std::vector<double> slopes,
                   std::uint64_t seed = 11)
        : base_(std::move(base_means)), slopes_(std::move(slopes)), rng_(seed) {
        if (base_.empty()) throw std::invalid_argument("base_means must not be empty");
        if (base_.size() != slopes_.size()) throw std::invalid_argument("slopes size mismatch");
    }

    std::size_t num_arms() const override { return base_.size(); }

    void set_round(std::size_t t) override {
        round_ = t;
        current_.resize(base_.size());
        for (std::size_t i = 0; i < base_.size(); ++i) {
            current_[i] = base_[i] + slopes_[i] * static_cast<double>(t);
        }
    }

    double pull(std::size_t arm) override {
        check(arm);
        std::bernoulli_distribution dist(clamp_prob(mean_at(arm)));
        return dist(rng_) ? 1.0 : 0.0;
    }

    double true_mean(std::size_t arm) const override {
        check(arm);
        return mean_at(arm);
    }

private:
    void check(std::size_t arm) const {
        if (arm >= base_.size()) throw std::out_of_range("arm index out of range");
    }

    double mean_at(std::size_t arm) const {
        if (round_ < current_.size()) {
            return current_[arm];
        }
        return base_[arm] + slopes_[arm] * static_cast<double>(round_);
    }

    static double clamp_prob(double p) {
        if (p < 0.0) return 0.0;
        if (p > 1.0) return 1.0;
        return p;
    }

    std::vector<double> base_;
    std::vector<double> slopes_;
    std::vector<double> current_;
    std::size_t round_{0};
    std::mt19937_64 rng_;
};

class AbruptSwitchBandit : public NonStationaryEnvironment {
public:
    AbruptSwitchBandit(std::vector<double> means,
                       std::size_t switch_round,
                       std::size_t arm_a,
                       std::size_t arm_b,
                       std::uint64_t seed = 13)
        : means_(std::move(means)), switch_round_(switch_round), arm_a_(arm_a), arm_b_(arm_b), rng_(seed) {
        if (means_.empty()) throw std::invalid_argument("means must not be empty");
        if (arm_a_ >= means_.size() || arm_b_ >= means_.size())
            throw std::out_of_range("switch arm index out of range");
    }

    std::size_t num_arms() const override { return means_.size(); }
    std::size_t switch_round() const { return switch_round_; }

    void set_round(std::size_t t) override { round_ = t; }

    bool switched() const { return round_ >= switch_round_; }

    double pull(std::size_t arm) override {
        check(arm);
        std::bernoulli_distribution dist(clamp_prob(true_mean(arm)));
        return dist(rng_) ? 1.0 : 0.0;
    }

    double true_mean(std::size_t arm) const override {
        check(arm);
        if (!switched()) return means_[arm];
        if (arm == arm_a_) return means_[arm_b_];
        if (arm == arm_b_) return means_[arm_a_];
        return means_[arm];
    }

private:
    void check(std::size_t arm) const {
        if (arm >= means_.size()) throw std::out_of_range("arm index out of range");
    }

    static double clamp_prob(double p) {
        if (p < 0.0) return 0.0;
        if (p > 1.0) return 1.0;
        return p;
    }

    std::vector<double> means_;
    std::size_t switch_round_{0};
    std::size_t arm_a_{0};
    std::size_t arm_b_{1};
    std::size_t round_{0};
    std::mt19937_64 rng_;
};

inline std::pair<std::vector<double>, std::vector<double>>
make_drift_params(std::size_t num_arms, double base_lo, double base_hi, double slope_mag) {
    std::vector<double> base(num_arms), slopes(num_arms);
    const double span = num_arms == 1 ? 0.0 : (base_hi - base_lo) / static_cast<double>(num_arms - 1);
    for (std::size_t i = 0; i < num_arms; ++i) {
        base[i] = base_lo + span * static_cast<double>(i);
        slopes[i] = (i < num_arms / 2) ? slope_mag : -slope_mag;
    }
    return {base, slopes};
}

class CyclicDriftBandit : public NonStationaryEnvironment {
public:
    CyclicDriftBandit(std::size_t num_arms,
                      double base = 0.6,
                      double amp = 0.3,
                      std::size_t period = 5000,
                      std::uint64_t seed = 17)
        : num_arms_(num_arms), base_(base), amp_(amp), period_(period), rng_(seed) {
        if (num_arms == 0) throw std::invalid_argument("num_arms must be > 0");
        if (period == 0) throw std::invalid_argument("period must be > 0");
        if (base - amp < 0.0 || base + amp > 1.0)
            throw std::invalid_argument("mean range must stay within [0,1]");
    }

    std::size_t num_arms() const override { return num_arms_; }

    void set_round(std::size_t t) override { round_ = t; }

    double pull(std::size_t arm) override {
        check(arm);
        std::bernoulli_distribution dist(true_mean(arm));
        return dist(rng_) ? 1.0 : 0.0;
    }

    double true_mean(std::size_t arm) const override {
        check(arm);
        const double phase = 6.283185307179586
               + static_cast<double>(arm) / static_cast<double>(num_arms_);
        return base_ + amp_ * std::sin(phase);
    }

private:
    void check(std::size_t arm) const {
        if (arm >= num_arms_) throw std::out_of_range("arm index out of range");
    }

    std::size_t num_arms_;
    double base_;
    double amp_;
    std::size_t period_;
    std::size_t round_{0};
    std::mt19937_64 rng_;
};

}  // namespace mab
