#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <random>

namespace mab {

struct SparseEntry {
    std::size_t index{0};
    double value{0.0};
};

class SparseVector {
public:
    SparseVector() = default;

    explicit SparseVector(std::vector<SparseEntry> entries) : entries_(std::move(entries)) {
        sort_and_merge();
    }

    void add(std::size_t index, double value) {
        if (value == 0.0) return;
        entries_.push_back(SparseEntry{index, value});
        sorted_ = false;
    }

    void finalize() { sort_and_merge(); }

    std::size_t nnz() const { return entries_.size(); }
    const std::vector<SparseEntry>& entries() const { return entries_; }

    double dot(const SparseVector& other) const {
        double sum = 0.0;
        std::size_t i = 0, j = 0;
        while (i < entries_.size() && j < other.entries_.size()) {
            const auto& a = entries_[i];
            const auto& b = other.entries_[j];
            if (a.index == b.index) {
                sum += a.value * b.value;
                ++i; ++j;
            } else if (a.index < b.index) {
                ++i;
            } else {
                ++j;
            }
        }
        return sum;
    }

    double squaredNorm() const {
        double sum = 0.0;
        for (const auto& e : entries_) sum += e.value * e.value;
        return sum;
    }

    double norm() const { return std::sqrt(squaredNorm()); }

private:
    void sort_and_merge() {
        if (sorted_) return;
        std::sort(entries_.begin(), entries_.end(),
                  [](const SparseEntry& a, const SparseEntry& b) { return a.index < b.index; });
        std::vector<SparseEntry> merged;
        merged.reserve(entries_.size());
        for (const auto& e : entries_) {
            if (!merged.empty() && merged.back().index == e.index) {
                merged.back().value += e.value;
            } else {
                merged.push_back(e);
            }
        }
        merged.erase(std::remove_if(merged.begin(), merged.end(),
                                    [](const SparseEntry& e) { return e.value == 0.0; }),
                     merged.end());
        entries_.swap(merged);
        sorted_ = true;
    }

    std::vector<SparseEntry> entries_;
    bool sorted_{true};
};

class SparseAccumulator {
public:
    void add(const SparseVector& vec, double scale) {
        for (const auto& e : vec.entries()) {
            sums_[e.index] += scale * e.value;
        }
    }

    double get(std::size_t index) const {
        auto it = sums_.find(index);
        return it == sums_.end() ? 0.0 : it->second;
    }

    std::size_t support_size() const { return sums_.size(); }

    void clear() { sums_.clear(); }

private:
    std::unordered_map<std::size_t, double> sums_;
};

inline SparseVector make_sparse_random(std::size_t dim,
                                       std::size_t nnz,
                                       std::mt19937_64& rng,
                                       double min_val = 0.5,
                                       double max_val = 1.5) {
    std::uniform_int_distribution<std::size_t> idx_dist(0, dim - 1);
    std::uniform_real_distribution<double> val_dist(min_val, max_val);
    std::vector<SparseEntry> entries;
    entries.reserve(nnz);
    std::unordered_map<std::size_t, bool> used;
    while (entries.size() < nnz) {
        std::size_t idx = idx_dist(rng);
        if (used.count(idx)) continue;
        used[idx] = true;
        entries.push_back(SparseEntry{idx, val_dist(rng)});
    }
    return SparseVector(std::move(entries));
}

inline std::vector<SparseVector> make_sparse_features_with_overlap(
    std::size_t num_arms, std::size_t dim, std::size_t nnz,
    double shared_ratio, std::mt19937_64& rng, double min_val = 0.5, double max_val = 1.5) {
    std::uniform_int_distribution<std::size_t> dim_dist(0, dim - 1);
    std::uniform_real_distribution<double> val_dist(min_val, max_val);

    const std::size_t shared_needed = static_cast<std::size_t>(std::ceil(shared_ratio * nnz));
    std::vector<std::size_t> shared_pool;
    shared_pool.reserve(std::max<std::size_t>(shared_needed, 1) * 2);
    {
        std::unordered_map<std::size_t, bool> used;
        while (shared_pool.size() < std::max<std::size_t>(shared_needed, 1) * 2) {
            const std::size_t d = dim_dist(rng);
            if (used.count(d)) continue;
            used[d] = true;
            shared_pool.push_back(d);
        }
    }

    std::vector<SparseVector> feats(num_arms);
    for (std::size_t a = 0; a < num_arms; ++a) {
        std::vector<SparseEntry> entries;
        entries.reserve(nnz);
        std::unordered_map<std::size_t, bool> taken;
        for (std::size_t i = 0; i < shared_needed && entries.size() < nnz; ++i) {
            const std::size_t d = shared_pool[(a * 7 + i) % shared_pool.size()];
            if (taken.count(d)) continue;
            taken[d] = true;
            entries.push_back(SparseEntry{d, val_dist(rng)});
        }
        while (entries.size() < nnz) {
            const std::size_t d = dim_dist(rng);
            if (taken.count(d)) continue;
            taken[d] = true;
            entries.push_back(SparseEntry{d, val_dist(rng)});
        }
        feats[a] = SparseVector(std::move(entries));
    }
    return feats;
}

}  // namespace mab
