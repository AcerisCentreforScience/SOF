#pragma once

#include "baselines.hpp"
#include "nonstationary_policies.hpp"
#include "sparse_vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace mab {


inline double sample_beta(std::mt19937_64& rng, double a, double b) {
    std::gamma_distribution<double> ga(a, 1.0);
    std::gamma_distribution<double> gb(b, 1.0);
    const double x = ga(rng);
    const double y = gb(rng);
    const double s = x + y;
    return s <= 0.0 ? 0.5 : x / s;
}

class PageHinkley {
public:
    PageHinkley(double z_threshold = 4.0, std::size_t window = 200,
                std::size_t min_obs = 30, std::size_t check_interval = 50)
        : z_threshold_(z_threshold), window_(window),
          min_obs_(min_obs), check_interval_(check_interval) {}

    bool update(double x) {
        buf_.push_back(x);
        sum_ += x;
        if (buf_.size() > window_) {
            sum_ -= buf_.front();
            buf_.pop_front();
        }
        ++since_check_;

        if (buf_.size() < min_obs_) return false;
        if (since_check_ < check_interval_) return false;
        since_check_ = 0;

        const double n = static_cast<double>(buf_.size());
        const double mean = sum_ / n;
        const double sigma = 0.5;
        const double z = std::fabs(mean) * std::sqrt(n) / sigma;
        if (z > z_threshold_) {
            ++detections_;
            reset();
            return true;
        }
        return false;
    }

    void reset() {
        buf_.clear();
        sum_ = 0.0;
        since_check_ = 0;
    }

    std::size_t detections() const { return detections_; }

private:
    double z_threshold_{4.0};
    std::size_t window_{200};
    std::size_t min_obs_{30};
    std::size_t check_interval_{50};
    std::deque<double> buf_;
    double sum_{0.0};
    std::size_t since_check_{0};
    std::size_t detections_{0};
};

class DAL_HCB3 : public Policy {
public:
    DAL_HCB3(std::size_t num_arms,
             double v1 = 1.0,
             double v2 = 1.0,
             double ph_delta = 0.05,
             double ph_threshold = 0.4,
             std::uint64_t seed = 606,
             double z_threshold = 4.0)
        : num_arms_(num_arms), v1_(v1), v2_(v2), detector_(z_threshold, 200, 30, 50) {
        if (num_arms == 0) throw std::invalid_argument("num_arms must be > 0");
        counts_.assign(num_arms, 0);
        means_.assign(num_arms, 0.0);
        m2_.assign(num_arms, 0.0);
        features_.resize(num_arms);
    }

    void set_feature(std::size_t arm, SparseVector feature) {
        if (arm >= num_arms_) throw std::out_of_range("arm index out of range");
        features_[arm] = std::move(feature);
    }

    std::size_t select() override {
        for (std::size_t i = 0; i < num_arms_; ++i) {
            if (counts_[i] == 0) return i;
        }
        std::size_t best = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < num_arms_; ++i) {
            const double score = upper_bound(i);
            if (score > best_score) { best_score = score; best = i; }
        }
        return best;
    }

    void update(std::size_t arm, double reward) override {
        if (arm >= num_arms_) throw std::out_of_range("arm index out of range");
        const double pred_err = reward - means_[arm];
        ++counts_[arm];
        const double delta = pred_err;
        means_[arm] += delta / static_cast<double>(counts_[arm]);
        m2_[arm] += delta * (reward - means_[arm]);
        ++total_;
        for (const auto& e : features_[arm].entries()) support_[e.index] = true;
        if (detector_.update(pred_err)) {
            ++detections_;
            decay();
        }
    }

    std::size_t num_arms() const override { return num_arms_; }
    std::size_t detections() const { return detections_; }

private:
    double upper_bound(std::size_t arm) const {
        const std::size_t n = counts_[arm];
        if (n == 0) return std::numeric_limits<double>::infinity();
        const double nd = static_cast<double>(n);
        const double N = static_cast<double>(std::max<std::size_t>(total_, 1));
        const double log_term = std::log(N + 1.0);
        const double var = n < 2 ? 0.0 : m2_[arm] / static_cast<double>(n - 1);
        const double var_term = v1_ * var + v2_ * std::sqrt(2.0 * log_term / nd);
        const double capped = std::min(0.25, var_term);
        double bonus = std::sqrt(log_term / nd * capped);
        if (features_[arm].nnz() > 0) bonus *= 1.0 + 0.5 * overlap(features_[arm]);
        return means_[arm] + bonus;
    }

    double overlap(const SparseVector& v) const {
        if (v.nnz() == 0) return 0.0;
        std::size_t hit = 0;
        for (const auto& e : v.entries()) {
            if (support_.count(e.index)) ++hit;
        }
        return static_cast<double>(hit) / static_cast<double>(v.nnz());
    }

    void decay() {
        for (std::size_t i = 0; i < num_arms_; ++i) {
            counts_[i] /= 2;
            m2_[i] /= 2.0;
        }
        total_ /= 2;
    }

    std::size_t num_arms_;
    double v1_, v2_;
    PageHinkley detector_;
    std::vector<std::size_t> counts_;
    std::vector<double> means_;
    std::vector<double> m2_;
    std::vector<SparseVector> features_;
    std::unordered_map<std::size_t, bool> support_;
    std::size_t total_{0};
    std::size_t detections_{0};
};

class TSCD : public Policy {
public:
    TSCD(std::size_t num_arms,
         double ph_delta = 0.05,
         double ph_threshold = 0.4,
         std::uint64_t seed = 111)
        : num_arms_(num_arms), detector_(4.0, 200, 30, 50), rng_(seed) {
        if (num_arms == 0) throw std::invalid_argument("num_arms must be > 0");
        alpha_.assign(num_arms, 1.0);
        beta_.assign(num_arms, 1.0);
    }

    std::size_t select() override {
        std::size_t best = 0;
        double best_sample = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < num_arms_; ++i) {
            const double s = sample_beta(rng_, alpha_[i], beta_[i]);
            if (s > best_sample) { best_sample = s; best = i; }
        }
        return best;
    }

    void update(std::size_t arm, double reward) override {
        if (arm >= num_arms_) throw std::out_of_range("arm index out of range");
        const double pred = alpha_[arm] / (alpha_[arm] + beta_[arm]);
        const double pred_err = reward - pred;
        alpha_[arm] += reward;
        beta_[arm] += 1.0 - reward;
        if (detector_.update(pred_err)) {
            ++detections_;
            alpha_.assign(num_arms_, 1.0);
            beta_.assign(num_arms_, 1.0);
        }
    }

    std::size_t num_arms() const override { return num_arms_; }
    std::size_t detections() const { return detections_; }

private:
    std::size_t num_arms_;
    PageHinkley detector_;
    std::vector<double> alpha_;
    std::vector<double> beta_;
    std::size_t detections_{0};
    std::mt19937_64 rng_;
};

class BOBSWUCB : public Policy {
public:
    BOBSWUCB(std::size_t num_arms,
             std::vector<std::size_t> window_grid,
             std::size_t block_len = 1000,
             std::uint64_t seed = 707)
        : num_arms_(num_arms), grid_(std::move(window_grid)), block_len_(block_len) {
        if (num_arms == 0 || grid_.empty() || block_len == 0)
            throw std::invalid_argument("invalid arguments");
        for (std::size_t w : grid_) {
            learners_.push_back(std::unique_ptr<SlidingWindowUCB>(new SlidingWindowUCB(num_arms, w, seed)));
        }
        master_counts_.assign(grid_.size(), 0);
        master_means_.assign(grid_.size(), 0.0);
    }

    std::size_t select() override {
        if (block_pos_ == 0) chosen_ = master_select();
        return learners_[chosen_]->select();
    }

    void update(std::size_t arm, double reward) override {
        for (auto& l : learners_) l->update(arm, reward);
        block_reward_ += reward;
        ++block_pos_;
        if (block_pos_ >= block_len_) {
            const double avg = block_reward_ / static_cast<double>(block_len_);
            ++master_counts_[chosen_];
            const double delta = avg - master_means_[chosen_];
            master_means_[chosen_] += delta / static_cast<double>(master_counts_[chosen_]);
            ++master_total_;
            block_pos_ = 0;
            block_reward_ = 0.0;
        }
    }

    std::size_t num_arms() const override { return num_arms_; }

    std::size_t chosen_window() const { return grid_[chosen_]; }

private:
    std::size_t master_select() {
        for (std::size_t i = 0; i < grid_.size(); ++i) {
            if (master_counts_[i] == 0) return i;
        }
        const double log_term = std::log(static_cast<double>(master_total_) + 1.0);
        std::size_t best = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < grid_.size(); ++i) {
            const double score = master_means_[i]
                + std::sqrt(2.0 * log_term / static_cast<double>(master_counts_[i]));
            if (score > best_score) { best_score = score; best = i; }
        }
        return best;
    }

    std::size_t num_arms_;
    std::vector<std::size_t> grid_;
    std::size_t block_len_;
    std::vector<std::unique_ptr<SlidingWindowUCB>> learners_;
    std::vector<std::size_t> master_counts_;
    std::vector<double> master_means_;
    std::size_t master_total_{0};
    std::size_t chosen_{0};
    std::size_t block_pos_{0};
    double block_reward_{0.0};
};

class SWMOSS : public Policy {
public:
    SWMOSS(std::size_t num_arms, std::size_t window, std::uint64_t seed = 808)
        : num_arms_(num_arms), window_(window) {
        if (num_arms == 0 || window == 0) throw std::invalid_argument("invalid arguments");
        windows_.resize(num_arms);
        sums_.assign(num_arms, 0.0);
    }

    std::size_t select() override {
        for (std::size_t i = 0; i < num_arms_; ++i) {
            if (windows_[i].empty()) return i;
        }
        std::size_t best = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < num_arms_; ++i) {
            const double n = static_cast<double>(windows_[i].size());
            const double mean = sums_[i] / n;
            const double arg = static_cast<double>(t_) / (static_cast<double>(num_arms_) * n);
            const double bonus = std::sqrt(std::max(0.0, (4.0 / n) * std::log(std::max(1.0, arg))));
            const double score = mean + bonus;
            if (score > best_score) { best_score = score; best = i; }
        }
        return best;
    }

    void update(std::size_t arm, double reward) override {
        ++t_;
        windows_[arm].push_back(reward);
        sums_[arm] += reward;
        if (windows_[arm].size() > window_) {
            sums_[arm] -= windows_[arm].front();
            windows_[arm].pop_front();
        }
    }

    std::size_t num_arms() const override { return num_arms_; }

private:
    std::size_t num_arms_;
    std::size_t window_;
    std::vector<std::deque<double>> windows_;
    std::vector<double> sums_;
    std::size_t t_{0};
};

class DiscountedTS : public Policy {
public:
    DiscountedTS(std::size_t num_arms, double gamma = 0.995, std::uint64_t seed = 909)
        : num_arms_(num_arms), gamma_(gamma), rng_(seed) {
        if (num_arms == 0) throw std::invalid_argument("num_arms must be > 0");
        if (gamma <= 0.0 || gamma > 1.0) throw std::invalid_argument("gamma must be in (0,1]");
        alpha_.assign(num_arms, 1.0);
        beta_.assign(num_arms, 1.0);
    }

    std::size_t select() override {
        std::size_t best = 0;
        double best_sample = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < num_arms_; ++i) {
            const double s = sample_beta(rng_, alpha_[i], beta_[i]);
            if (s > best_sample) { best_sample = s; best = i; }
        }
        return best;
    }

    void update(std::size_t arm, double reward) override {
        if (arm >= num_arms_) throw std::out_of_range("arm index out of range");
        alpha_[arm] = gamma_ * alpha_[arm] + reward;
        beta_[arm] = gamma_ * beta_[arm] + (1.0 - reward);
    }

    std::size_t num_arms() const override { return num_arms_; }

private:
    std::size_t num_arms_;
    double gamma_;
    std::vector<double> alpha_;
    std::vector<double> beta_;
    std::mt19937_64 rng_;
};

class ADRBandit : public Policy {
public:
    ADRBandit(std::size_t num_arms,
              std::size_t min_interval = 200,
              std::size_t max_interval = 5000,
              double ph_delta = 0.05,
              double ph_threshold = 0.4,
              std::uint64_t seed = 505)
        : num_arms_(num_arms), min_interval_(min_interval), max_interval_(max_interval),
          initial_max_(max_interval), detector_(4.0, 200, 30, 50) {
        if (num_arms == 0 || min_interval == 0 || max_interval < min_interval)
            throw std::invalid_argument("invalid arguments");
        counts_.assign(num_arms, 0);
        means_.assign(num_arms, 0.0);
    }

    std::size_t select() override {
        if (t_ > 0 && ((alarm_ && t_since_ >= min_interval_) || t_since_ >= max_interval_)) {
            restart();
        }
        for (std::size_t i = 0; i < num_arms_; ++i) {
            if (counts_[i] == 0) return i;
        }
        const double log_term = std::log(static_cast<double>(t_since_) + 1.0);
        std::size_t best = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < num_arms_; ++i) {
            const double score = means_[i]
                + std::sqrt(2.0 * log_term / static_cast<double>(counts_[i]));
            if (score > best_score) { best_score = score; best = i; }
        }
        return best;
    }

    void update(std::size_t arm, double reward) override {
        if (arm >= num_arms_) throw std::out_of_range("arm index out of range");
        ++t_;
        ++t_since_;
        const double pred_err = reward - means_[arm];
        ++counts_[arm];
        const double delta = pred_err;
        means_[arm] += delta / static_cast<double>(counts_[arm]);
        if (detector_.update(pred_err)) alarm_ = true;
    }

    std::size_t num_arms() const override { return num_arms_; }
    std::size_t restarts() const { return restarts_; }

private:
    void restart() {
        const bool triggered_by_alarm = alarm_;
        counts_.assign(num_arms_, 0);
        means_.assign(num_arms_, 0.0);
        t_since_ = 0;
        alarm_ = false;
        detector_.reset();
        ++restarts_;
        if (triggered_by_alarm) {
            max_interval_ = std::max(min_interval_ * 2, max_interval_ / 2);
        } else {
            max_interval_ = std::min(initial_max_ * 2, max_interval_ + max_interval_ / 4);
        }
    }

    std::size_t num_arms_;
    std::size_t min_interval_;
    std::size_t max_interval_;
    std::size_t initial_max_;
    PageHinkley detector_;
    std::vector<std::size_t> counts_;
    std::vector<double> means_;
    std::size_t t_{0};
    std::size_t t_since_{0};
    std::size_t restarts_{0};
    bool alarm_{false};
};

class SWLinUCB : public Policy {
public:
    SWLinUCB(std::size_t num_arms,
             std::size_t window = 500,
             double lambda = 1.0,
             double alpha = 1.0,
             std::uint64_t seed = 404)
        : num_arms_(num_arms), window_(window), lambda_(lambda), alpha_(alpha) {
        if (num_arms == 0 || window == 0 || lambda <= 0.0) throw std::invalid_argument("invalid arguments");
        windows_.resize(num_arms);
        sum_r_.assign(num_arms, 0.0);
        features_.resize(num_arms);
    }

    void set_feature(std::size_t arm, SparseVector feature) {
        if (arm >= num_arms_) throw std::out_of_range("arm index out of range");
        features_[arm] = std::move(feature);
    }

    std::size_t select() override {
        for (std::size_t i = 0; i < num_arms_; ++i) {
            if (windows_[i].empty()) return i;
        }
        std::size_t best = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < num_arms_; ++i) {
            const double score = score_of(i);
            if (score > best_score) { best_score = score; best = i; }
        }
        return best;
    }

    void update(std::size_t arm, double reward) override {
        if (arm >= num_arms_) throw std::out_of_range("arm index out of range");
        windows_[arm].push_back(reward);
        sum_r_[arm] += reward;
        if (windows_[arm].size() > window_) {
            sum_r_[arm] -= windows_[arm].front();
            windows_[arm].pop_front();
        }
    }

    std::size_t num_arms() const override { return num_arms_; }

protected:
    virtual double weight_of(double raw) const { return raw; }

    double score_of(std::size_t arm) const {
        const SparseVector& f = features_[arm];
        const std::size_t n = windows_[arm].size();
        if (n == 0) return std::numeric_limits<double>::infinity();
        const double nd = static_cast<double>(n);
        const double sum_r = sum_r_[arm];
        double linear = 0.0;
        double conf = 0.0;
        for (const auto& e : f.entries()) {
            const double x = e.value;
            if (x == 0.0) continue;
            const double denom = nd * x * x + lambda_;
            const double raw_w = x * sum_r / denom;
            const double w = weight_of(raw_w);
            linear += x * w;
            conf += (x * x) / denom;
        }
        return linear + alpha_ * std::sqrt(std::max(0.0, conf));
    }

    std::size_t num_arms_;
    std::size_t window_;
    double lambda_;
    double alpha_;
    std::vector<std::deque<double>> windows_;
    std::vector<double> sum_r_;
    std::vector<SparseVector> features_;
};

class ADLasso : public Policy {
public:
    ADLasso(std::size_t num_arms,
            std::size_t window = 500,
            double lambda = 1.0,
            double l1 = 0.02,
            std::uint64_t seed = 303)
        : num_arms_(num_arms), window_(window), min_window_(50), lambda_(lambda),
          l1_(l1), detector_(4.0, 200, 30, 50) {
        if (num_arms == 0 || window == 0 || lambda <= 0.0) throw std::invalid_argument("invalid arguments");
        windows_.resize(num_arms);
        sum_r_.assign(num_arms, 0.0);
        features_.resize(num_arms);
    }

    void set_feature(std::size_t arm, SparseVector feature) {
        if (arm >= num_arms_) throw std::out_of_range("arm index out of range");
        features_[arm] = std::move(feature);
    }

    std::size_t select() override {
        for (std::size_t i = 0; i < num_arms_; ++i) {
            if (windows_[i].empty()) return i;
        }
        std::size_t best = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < num_arms_; ++i) {
            const double score = score_of(i);
            if (score > best_score) { best_score = score; best = i; }
        }
        return best;
    }

    void update(std::size_t arm, double reward) override {
        if (arm >= num_arms_) throw std::out_of_range("arm index out of range");
        const double pred = windows_[arm].empty() ? 0.0 : sum_r_[arm] / static_cast<double>(windows_[arm].size());
        const double pred_err = reward - pred;
        windows_[arm].push_back(reward);
        sum_r_[arm] += reward;
        while (windows_[arm].size() > window_) {
            sum_r_[arm] -= windows_[arm].front();
            windows_[arm].pop_front();
        }
        if (detector_.update(pred_err)) {
            ++detections_;
            window_ = std::max(min_window_, window_ / 2);
            l1_ = std::min(0.5, l1_ + 0.02);
        }
    }

    std::size_t num_arms() const override { return num_arms_; }
    std::size_t detections() const { return detections_; }
    std::size_t current_window() const { return window_; }
    double current_l1() const { return l1_; }

private:
    double score_of(std::size_t arm) const {
        const SparseVector& f = features_[arm];
        const std::size_t n = windows_[arm].size();
        if (n == 0) return std::numeric_limits<double>::infinity();
        const double nd = static_cast<double>(n);
        const double sum_r = sum_r_[arm];
        double linear = 0.0;
        double conf = 0.0;
        for (const auto& e : f.entries()) {
            const double x = e.value;
            if (x == 0.0) continue;
            const double denom = nd * x * x + lambda_;
            const double raw_w = x * sum_r / denom;
            const double sign = raw_w >= 0.0 ? 1.0 : -1.0;
            const double w = sign * std::max(std::fabs(raw_w) - l1_, 0.0);
            linear += x * w;
            conf += (x * x) / denom;
        }
        return linear + std::sqrt(std::max(0.0, conf));
    }

    std::size_t num_arms_;
    std::size_t window_;
    std::size_t min_window_;
    double lambda_;
    double l1_;
    PageHinkley detector_;
    std::vector<std::deque<double>> windows_;
    std::vector<double> sum_r_;
    std::vector<SparseVector> features_;
    std::size_t detections_{0};
};

}  // namespace mab
