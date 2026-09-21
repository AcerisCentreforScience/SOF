#include "adaptive_ucb.hpp"
#include "nonstationary.hpp"
#include "nonstationary_policies.hpp"
#include "sparse_vector.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace mab;

double mean_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}
double std_of(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean_of(v);
    double s = 0.0;
    for (double x : v) { const double d = x - m; s += d * d; }
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}
std::string f2(double v) { std::ostringstream o; o << std::fixed << std::setprecision(2) << v; return o.str(); }

struct RunOut { double total{0.0}; std::vector<double> curve; };

RunOut run_drift(NonStationaryEnvironment& env, Policy& policy, std::size_t rounds, std::size_t window) {
    RunOut r;
    std::size_t win_count = 0;
    double win_reg = 0.0;
    for (std::size_t t = 0; t < rounds; ++t) {
        env.set_round(t);
        const std::size_t arm = policy.select();
        const double reward = env.pull(arm);
        policy.update(arm, reward);
        const double regret = env.best_mean() - env.true_mean(arm);
        r.total += regret;
        ++win_count; win_reg += regret;
        if (win_count == window) {
            r.curve.push_back(win_reg / static_cast<double>(window));
            win_count = 0; win_reg = 0.0;
        }
    }
    if (win_count > 0) r.curve.push_back(win_reg / static_cast<double>(win_count));
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t rounds = 20000;
    std::size_t num_arms = 10;
    std::size_t feature_dim = 1024;
    std::size_t feature_nnz = 8;
    std::size_t window = 500;
    double gamma = 0.995;
    int num_seeds = 10;
    if (argc > 1) rounds = static_cast<std::size_t>(std::stoull(argv[1]));
    if (argc > 2) num_seeds = std::stoi(argv[2]);

    auto make_alg = [&](int which, std::uint64_t seed) -> std::shared_ptr<Policy> {
        if (which == 0) {
            AdaptiveConfig c; c.mode = StatsMode::Discounted; c.gamma = gamma;
            c.v1 = 1.0; c.v2 = 1.0; c.exploration_dim = feature_dim;
            auto p = std::make_shared<AdaptiveHCB3Policy>(num_arms, c, static_cast<std::uint64_t>(7000 + seed));
            std::mt19937_64 r(static_cast<std::uint64_t>(9000 + seed));
            for (std::size_t i = 0; i < num_arms; ++i) p->set_feature(i, make_sparse_random(feature_dim, feature_nnz, r));
            return p;
        } else if (which == 1) {
            AdaptiveConfig c; c.mode = StatsMode::SlidingWindow; c.window_size = window;
            c.v1 = 1.0; c.v2 = 1.0; c.exploration_dim = feature_dim;
            auto p = std::make_shared<AdaptiveHCB3Policy>(num_arms, c, static_cast<std::uint64_t>(7000 + seed));
            std::mt19937_64 r(static_cast<std::uint64_t>(9000 + seed));
            for (std::size_t i = 0; i < num_arms; ++i) p->set_feature(i, make_sparse_random(feature_dim, feature_nnz, r));
            return p;
        } else if (which == 2) {
            AdaptiveConfig c; c.mode = StatsMode::Full;
            c.v1 = 1.0; c.v2 = 1.0; c.exploration_dim = feature_dim;
            auto p = std::make_shared<AdaptiveHCB3Policy>(num_arms, c, static_cast<std::uint64_t>(7000 + seed));
            std::mt19937_64 r(static_cast<std::uint64_t>(9000 + seed));
            for (std::size_t i = 0; i < num_arms; ++i) p->set_feature(i, make_sparse_random(feature_dim, feature_nnz, r));
            return p;
        } else {
            return std::make_shared<SlidingWindowUCB>(num_arms, window, static_cast<std::uint64_t>(1000 + seed));
        }
    };

    std::cout << "===== Drift period sweep (non-stationary, 10 seeds)=====\n";
    std::cout << "rounds=" << rounds << " arms=" << num_arms << " amp=0.3(0.3~0.9) window=" << window << "\n";
    std::cout << std::left << std::setw(10) << "period"
              << std::setw(24) << "HCB3-Disc"
              << std::setw(24) << "HCB3-SW"
              << std::setw(24) << "HCB3-Full"
              << std::setw(24) << "SW-UCB"
              << std::setw(14) << "Disc-SW diff" << "\n";
    std::cout << std::string(120, '-') << "\n";

    const std::size_t periods[] = {500, 1000, 2000, 5000, 10000, 20000};
    for (std::size_t period : periods) {
        double res[4][2] = {};
        for (int w = 0; w < 4; ++w) {
            std::vector<double> totals;
            for (int s = 0; s < num_seeds; ++s) {
                const std::uint64_t seed = static_cast<std::uint64_t>(s + 1);
                CyclicDriftBandit env(num_arms, 0.6, 0.3, period, static_cast<std::uint64_t>(500 + seed));
                auto p = make_alg(w, seed);
                RunOut r = run_drift(env, *p, rounds, window);
                totals.push_back(r.total);
            }
            res[w][0] = mean_of(totals); res[w][1] = std_of(totals);
        }
        std::cout << std::left << std::setw(10) << period
                  << std::setw(24) << (f2(res[0][0]) + "±" + f2(res[0][1]))
                  << std::setw(24) << (f2(res[1][0]) + "±" + f2(res[1][1]))
                  << std::setw(24) << (f2(res[2][0]) + "±" + f2(res[2][1]))
                  << std::setw(24) << (f2(res[3][0]) + "±" + f2(res[3][1]))
                  << std::setw(14) << f2(res[3][0] - res[0][0]) << "\n";
    }

    std::cout << "\n===== Drift amplitude sweep (period 5000, 10 seeds)=====\n";
    std::cout << std::left << std::setw(10) << "amp"
              << std::setw(24) << "HCB3-Disc"
              << std::setw(24) << "HCB3-SW"
              << std::setw(24) << "HCB3-Full"
              << std::setw(24) << "SW-UCB"
              << std::setw(14) << "Disc-SW diff" << "\n";
    std::cout << std::string(120, '-') << "\n";

    const double amps[] = {0.1, 0.3, 0.5, 0.8};
    for (double amp : amps) {
        double env_base = 0.5;
        double env_amp = amp;
        if (amp > 0.5) { env_amp = 0.4; env_base = 0.5; }
        else if (amp == 0.5) { env_base = 0.5; env_amp = 0.5; }
        else if (amp == 0.3) { env_base = 0.6; env_amp = 0.3; }
        double res[4][2] = {};
        for (int w = 0; w < 4; ++w) {
            std::vector<double> totals;
            for (int s = 0; s < num_seeds; ++s) {
                const std::uint64_t seed = static_cast<std::uint64_t>(s + 1);
                CyclicDriftBandit env(num_arms, env_base, env_amp, 5000, static_cast<std::uint64_t>(500 + seed));
                auto p = make_alg(w, seed);
                RunOut r = run_drift(env, *p, rounds, window);
                totals.push_back(r.total);
            }
            res[w][0] = mean_of(totals); res[w][1] = std_of(totals);
        }
        std::cout << std::left << std::setw(10) << f2(amp)
                  << std::setw(24) << (f2(res[0][0]) + "±" + f2(res[0][1]))
                  << std::setw(24) << (f2(res[1][0]) + "±" + f2(res[1][1]))
                  << std::setw(24) << (f2(res[2][0]) + "±" + f2(res[2][1]))
                  << std::setw(24) << (f2(res[3][0]) + "±" + f2(res[3][1]))
                  << std::setw(14) << f2(res[3][0] - res[0][0]) << "\n";
    }

    std::cout << "\nNote: Disc-SW>0 means discounted beats SW-UCB; shrinking diff marks agility boundary\n";
    std::cout << "Drift sweep done.\n";
    return 0;
}
