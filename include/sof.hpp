#pragma once

#include "sparse_vector.hpp"

#include <cstddef>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mab {

struct ArmStats {
    std::size_t counts{0};
    double mean{0.0};
    double m2{0.0};
    SparseVector feature;

    void update(double reward) {
        ++counts;
        const double delta = reward - mean;
        mean += delta / static_cast<double>(counts);
        m2 += delta * (reward - mean);
    }

    double variance() const {
        if (counts < 2) return 0.0;
        return m2 / static_cast<double>(counts - 1);
    }
};

struct SOFConfig {
    std::size_t exploration_dim{0};
    double v1{1.0};
    double v2{1.0};
    double xi{0.0};
    std::size_t total_rounds{1};
};

class SOFPolicy {
public:
    SOFPolicy(std::size_t num_arms, SOFConfig config, std::uint64_t seed = 42)
        : config_(config), rng_(seed) {
        if (num_arms == 0) throw std::invalid_argument("num_arms must be > 0");
        arms_.resize(num_arms);
        feature_sum_ = SparseVector();
    }

    std::size_t num_arms() const { return arms_.size(); }
    std::size_t total_pulls() const { return total_pulls_; }

    void set_feature(std::size_t arm, SparseVector feature) {
        if (arm >= arms_.size()) throw std::out_of_range("arm index out of range");
        arms_[arm].feature = std::move(feature);
    }

    double upper_bound(std::size_t arm) const {
        const ArmStats& s = arms_[arm];
        if (s.counts == 0) return std::numeric_limits<double>::infinity();

        const double n = static_cast<double>(s.counts);
        const double N = static_cast<double>(std::max<std::size_t>(total_pulls_, 1));
        const double log_term = std::log(N + 1.0);

        const double var_term = config_.v1 * s.variance()
                              + config_.v2 * std::sqrt(2.0 * log_term / n)
                              + config_.xi;
        const double capped_var = std::min(0.25, var_term);

        double bonus = std::sqrt(log_term / n * capped_var);

        if (config_.exploration_dim > 0 && s.feature.nnz() > 0) {
            const double feature_norm = s.feature.norm();
            const double overlap_ratio = feature_support_overlap(s.feature);
            bonus *= (1.0 + config_.v2 * overlap_ratio) / (1.0 + feature_norm > 0 ? (1.0 + feature_norm) : 1.0);
        }

        return s.mean + bonus;
    }

    std::size_t select() {
        for (std::size_t i = 0; i < arms_.size(); ++i) {
            if (arms_[i].counts == 0) return i;
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
        accumulate_support(arms_[arm].feature);
    }

    const ArmStats& stats(std::size_t arm) const { return arms_[arm]; }

    std::size_t best_empirical_arm() const {
        std::size_t best = 0;
        double best_mean = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < arms_.size(); ++i) {
            if (arms_[i].counts == 0) continue;
            if (arms_[i].mean > best_mean) {
                best_mean = arms_[i].mean;
                best = i;
            }
        }
        return best;
    }

private:
    double feature_support_overlap(const SparseVector& v) const {
        std::size_t hit = 0;
        for (const auto& e : v.entries()) {
            if (support_.count(e.index)) ++hit;
        }
        return v.nnz() == 0 ? 0.0 : static_cast<double>(hit) / static_cast<double>(v.nnz());
    }

    void accumulate_support(const SparseVector& v) {
        for (const auto& e : v.entries()) {
            support_[e.index] = true;
        }
    }

    SOFConfig config_;
    std::vector<ArmStats> arms_;
    SparseVector feature_sum_;
    std::unordered_map<std::size_t, bool> support_;
    std::size_t total_pulls_{0};
    std::mt19937_64 rng_;
};

}  // namespace mab
