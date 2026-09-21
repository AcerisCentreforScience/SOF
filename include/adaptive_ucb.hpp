#pragma once

#include "baselines.hpp"
#include "sparse_vector.hpp"

#include <cstddef>
#include <cmath>
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mab {

enum class StatsMode { Full, SlidingWindow, Discounted };

struct AdaptiveConfig {
    StatsMode mode{StatsMode::Full};
    std::size_t window_size{200};
    double gamma{0.99};
    double v1{1.0};
    double v2{1.0};
    double xi{0.0};
    std::size_t exploration_dim{0};
    bool cap_variance{true};
    bool sparse_correction{true};
};

class AdaptiveArmStats {
public:
    AdaptiveArmStats() = default;

    explicit AdaptiveArmStats(const AdaptiveConfig& cfg) : cfg_(cfg) {}

    void update(double reward) {
        ++total_counts_;
        switch (cfg_.mode) {
            case StatsMode::Full: {
                const double delta = reward - full_mean_;
                full_mean_ += delta / static_cast<double>(total_counts_);
                full_m2_ += delta * (reward - full_mean_);
                break;
            }
            case StatsMode::SlidingWindow: {
                window_.push_back(reward);
                win_sum_ += reward;
                win_sumsq_ += reward * reward;
                while (window_.size() > cfg_.window_size) {
                    const double old = window_.front();
                    window_.pop_front();
                    win_sum_ -= old;
                    win_sumsq_ -= old * old;
                }
                break;
            }
            case StatsMode::Discounted: {
                d_weight_ = d_weight_ * cfg_.gamma + 1.0;
                d_sum_ = d_sum_ * cfg_.gamma + reward;
                d_sumsq_ = d_sumsq_ * cfg_.gamma + reward * reward;
                break;
            }
        }
    }

    std::size_t effective_counts() const {
        switch (cfg_.mode) {
            case StatsMode::Full: return total_counts_;
            case StatsMode::SlidingWindow: return window_.size();
            case StatsMode::Discounted: return static_cast<std::size_t>(std::ceil(d_weight_));
        }
        return total_counts_;
    }

    std::size_t total_counts() const { return total_counts_; }

    double mean() const {
        switch (cfg_.mode) {
            case StatsMode::Full:
                return total_counts_ == 0 ? 0.0 : full_mean_;
            case StatsMode::SlidingWindow:
                return window_.empty() ? 0.0 : win_sum_ / static_cast<double>(window_.size());
            case StatsMode::Discounted:
                return d_weight_ <= 0.0 ? 0.0 : d_sum_ / d_weight_;
        }
        return 0.0;
    }

    double variance() const {
        switch (cfg_.mode) {
            case StatsMode::Full:
                return total_counts_ < 2 ? 0.0 : full_m2_ / static_cast<double>(total_counts_ - 1);
            case StatsMode::SlidingWindow: {
                const std::size_t n = window_.size();
                if (n < 2) return 0.0;
                const double m = win_sum_ / static_cast<double>(n);
                const double var = win_sumsq_ / static_cast<double>(n) - m * m;
                return var < 0.0 ? 0.0 : var * static_cast<double>(n) / static_cast<double>(n - 1);
            }
            case StatsMode::Discounted: {
                if (d_weight_ <= 0.0) return 0.0;
                const double m = d_sum_ / d_weight_;
                const double var = d_sumsq_ / d_weight_ - m * m;
                return var < 0.0 ? 0.0 : var;
            }
        }
        return 0.0;
    }

    SparseVector feature;

private:
    AdaptiveConfig cfg_{};
    std::size_t total_counts_{0};
    double full_mean_{0.0};
    double full_m2_{0.0};
    std::deque<double> window_;
    double win_sum_{0.0};
    double win_sumsq_{0.0};
    double d_sum_{0.0};
    double d_sumsq_{0.0};
    double d_weight_{0.0};
};

class AdaptiveHCB3Policy : public Policy {
public:
    AdaptiveHCB3Policy(std::size_t num_arms, AdaptiveConfig cfg, std::uint64_t seed = 42)
        : cfg_(cfg) {
        if (num_arms == 0) throw std::invalid_argument("num_arms must be > 0");
        arms_.resize(num_arms, AdaptiveArmStats(cfg_));
    }

    std::size_t num_arms() const { return arms_.size(); }

    void set_feature(std::size_t arm, SparseVector feature) {
        if (arm >= arms_.size()) throw std::out_of_range("arm index out of range");
        arms_[arm].feature = std::move(feature);
    }

    double upper_bound(std::size_t arm) const {
        const AdaptiveArmStats& s = arms_[arm];
        const std::size_t n_eff = s.effective_counts();
        if (n_eff == 0) return std::numeric_limits<double>::infinity();

        const double n = static_cast<double>(n_eff);
        const double N = static_cast<double>(std::max<std::size_t>(total_pulls_, 1));
        const double log_term = std::log(N + 1.0);

        const double var_term = cfg_.v1 * s.variance()
                              + cfg_.v2 * std::sqrt(2.0 * log_term / n)
                              + cfg_.xi;
        const double capped_var = cfg_.cap_variance
            ? std::min(0.25, var_term)
            : var_term;
        double bonus = std::sqrt(log_term / n * capped_var);

        if (cfg_.sparse_correction && cfg_.exploration_dim > 0 && s.feature.nnz() > 0) {
            const double overlap = feature_support_overlap(s.feature);
            bonus *= 1.0 + 0.5 * overlap;
        }
        return s.mean() + bonus;
    }

    std::size_t select() {
        for (std::size_t i = 0; i < arms_.size(); ++i) {
            if (arms_[i].effective_counts() == 0) return i;
        }
        std::size_t best = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < arms_.size(); ++i) {
            const double score = upper_bound(i);
            if (score > best_score) {
                best_score = score;
                best = i;
            }
        }
        return best;
    }

    void update(std::size_t arm, double reward) {
        if (arm >= arms_.size()) throw std::out_of_range("arm index out of range");
        arms_[arm].update(reward);
        ++total_pulls_;
        for (const auto& e : arms_[arm].feature.entries()) support_[e.index] = true;
    }

    const AdaptiveArmStats& stats(std::size_t arm) const { return arms_[arm]; }

private:
    double feature_support_overlap(const SparseVector& v) const {
        if (v.nnz() == 0) return 0.0;
        std::size_t hit = 0;
        for (const auto& e : v.entries()) {
            if (support_.count(e.index)) ++hit;
        }
        return static_cast<double>(hit) / static_cast<double>(v.nnz());
    }

    AdaptiveConfig cfg_;
    std::vector<AdaptiveArmStats> arms_;
    std::unordered_map<std::size_t, bool> support_;
    std::size_t total_pulls_{0};
};

}  // namespace mab
