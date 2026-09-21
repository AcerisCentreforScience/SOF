#pragma once

#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mab {

class LinUCBPolicy {
public:
    LinUCBPolicy(std::size_t num_arms, std::size_t feature_dim,
                 double alpha = 1.0, double lambda = 1.0)
        : num_arms_(num_arms), dim_(feature_dim), alpha_(alpha) {
        if (num_arms_ == 0) throw std::invalid_argument("num_arms must be > 0");
        if (dim_ == 0) throw std::invalid_argument("feature_dim must be > 0");
        A_.assign(num_arms_, std::vector<double>(dim_ * dim_, 0.0));
        b_.assign(num_arms_, std::vector<double>(dim_, 0.0));
        Ainv_.assign(num_arms_, std::vector<double>());
        for (std::size_t a = 0; a < num_arms_; ++a) {
            for (std::size_t i = 0; i < dim_; ++i) {
                A_[a][i * dim_ + i] = lambda;
            }
        }
    }

    std::size_t num_arms() const { return num_arms_; }

    std::size_t select(const std::vector<double>& x) const {
        if (x.size() != dim_) throw std::invalid_argument("context dim mismatch");
        std::size_t best = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t a = 0; a < num_arms_; ++a) {
            const double score = score_of(a, x);
            if (score > best_score) {
                best_score = score;
                best = a;
            }
        }
        return best;
    }

    void update(std::size_t arm, const std::vector<double>& x, double reward) {
        if (arm >= num_arms_) throw std::out_of_range("arm index out of range");
        std::vector<double>& Ainv = Ainv_[arm];
        if (Ainv.empty()) {
            Ainv.assign(dim_ * dim_, 0.0);
            for (std::size_t i = 0; i < dim_; ++i) Ainv[i * dim_ + i] = 1.0 / lambda_;
        }
        std::vector<double> Ax(dim_, 0.0);
        for (std::size_t i = 0; i < dim_; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < dim_; ++j) s += Ainv[i * dim_ + j] * x[j];
            Ax[i] = s;
        }
        double xAx = 0.0;
        for (std::size_t i = 0; i < dim_; ++i) xAx += x[i] * Ax[i];
        const double denom = 1.0 + xAx;
        for (std::size_t i = 0; i < dim_; ++i) {
            for (std::size_t j = 0; j < dim_; ++j) {
                Ainv[i * dim_ + j] -= Ax[i] * Ax[j] / denom;
            }
        }
        for (std::size_t i = 0; i < dim_; ++i) b_[arm][i] += reward * x[i];
        for (std::size_t i = 0; i < dim_; ++i) {
            for (std::size_t j = 0; j < dim_; ++j) {
                A_[arm][i * dim_ + j] += x[i] * x[j];
            }
        }
    }

private:
    double score_of(std::size_t arm, const std::vector<double>& x) const {
        std::vector<double> theta(dim_, 0.0);
        if (!Ainv_[arm].empty()) {
            for (std::size_t i = 0; i < dim_; ++i) {
                double s = 0.0;
                for (std::size_t j = 0; j < dim_; ++j) s += Ainv_[arm][i * dim_ + j] * b_[arm][j];
                theta[i] = s;
            }
        }
        double mean = 0.0;
        for (std::size_t i = 0; i < dim_; ++i) mean += x[i] * theta[i];
        double var = 0.0;
        if (!Ainv_[arm].empty()) {
            std::vector<double> Ax(dim_, 0.0);
            for (std::size_t i = 0; i < dim_; ++i) {
                double s = 0.0;
                for (std::size_t j = 0; j < dim_; ++j) s += Ainv_[arm][i * dim_ + j] * x[j];
                Ax[i] = s;
            }
            for (std::size_t i = 0; i < dim_; ++i) var += x[i] * Ax[i];
        }
        return mean + alpha_ * std::sqrt(std::max(var, 0.0));
    }

    std::size_t num_arms_;
    std::size_t dim_;
    double alpha_;
    double lambda_{1.0};
    std::vector<std::vector<double>> A_;
    std::vector<std::vector<double>> b_;
    mutable std::vector<std::vector<double>> Ainv_{};
};

}  // namespace mab
