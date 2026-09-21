#include "adaptive_ucb.hpp"
#include "advanced_policies.hpp"
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
std::string f1(double v) { std::ostringstream o; o << std::fixed << std::setprecision(1) << v; return o.str(); }

struct RunOut { double total{0.0}; double post{0.0}; };

template <typename PolicyT>
RunOut run_once(NonStationaryEnvironment& env, PolicyT& policy, std::size_t rounds, std::size_t sw_round) {
    RunOut r;
    for (std::size_t t = 0; t < rounds; ++t) {
        env.set_round(t);
        const std::size_t arm = policy.select();
        const double reward = env.pull(arm);
        policy.update(arm, reward);
        const double regret = env.best_mean() - env.true_mean(arm);
        r.total += regret;
        if (t >= sw_round) r.post += regret;
    }
    return r;
}

struct AlgAgg { double tm{0.0}, ts{0.0}, pm{0.0}; };

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

    const std::size_t sw = rounds / 2;
    const double deltas[] = {0.05, 0.10, 0.20, 0.30, 0.50, 0.70};

    std::cout << "===== Abrupt-switch Delta sweep (non-stationary, 10 seeds)=====\n";
    std::cout << "rounds=" << rounds << " arms=" << num_arms << " feat_dim=" << feature_dim
              << " nnz/arm=" << feature_nnz << " switch_pt=" << sw << " seeds=" << num_seeds << "\n";
    std::cout << "Env: pre-switch best=0.5+D/2, rest=0.5-D/2; swap at t=" << sw << " swap best/2nd\n\n";

    std::cout << std::left
              << std::setw(7) << "Δ"
              << std::setw(20) << "Disc cum"
              << std::setw(20) << "Disc post"
              << std::setw(20) << "SW-UCB cum"
              << std::setw(20) << "SW-UCB post"
              << std::setw(12) << "alarms"
              << std::setw(12) << "trigger"
              << std::setw(20) << "DAL cum"
              << std::setw(20) << "Full cum"
              << std::setw(16) << "Full-DAL diff" << "\n";
    std::cout << std::string(170, '-') << "\n";

    for (double delta : deltas) {
        auto mk_env = [&, delta](std::uint64_t s) -> std::shared_ptr<NonStationaryEnvironment> {
            std::vector<double> m(num_arms, 0.5 - delta / 2.0);
            m[0] = 0.5 + delta / 2.0;
            return std::make_shared<AbruptSwitchBandit>(m, sw, 0, 1, static_cast<std::uint64_t>(400 + s));
        };

        std::vector<double> disc_t, swucb_t, dal_t, full_t, dal_p, disc_p, full_p, swucb_p;
        double det_sum = 0.0; int det_fired = 0;

        for (int s = 0; s < num_seeds; ++s) {
            const std::uint64_t seed = static_cast<std::uint64_t>(s + 1);
            std::mt19937_64 frng(static_cast<std::uint64_t>(9000 + s));

            {
                auto env = mk_env(seed);
                AdaptiveConfig c; c.mode = StatsMode::Discounted; c.gamma = gamma;
                c.v1 = 1.0; c.v2 = 1.0; c.exploration_dim = feature_dim;
                AdaptiveHCB3Policy p(num_arms, c, static_cast<std::uint64_t>(7000 + s));
                std::mt19937_64 r2(static_cast<std::uint64_t>(9000 + s));
                for (std::size_t i = 0; i < num_arms; ++i) p.set_feature(i, make_sparse_random(feature_dim, feature_nnz, r2));
                RunOut r = run_once(*env, p, rounds, sw);
                disc_t.push_back(r.total); disc_p.push_back(r.post);
            }
            {
                auto env = mk_env(seed);
                SlidingWindowUCB p(num_arms, window, static_cast<std::uint64_t>(1000 + s));
                RunOut r = run_once(*env, p, rounds, sw);
                swucb_t.push_back(r.total); swucb_p.push_back(r.post);
            }
            {
                auto env = mk_env(seed);
                DAL_HCB3 p(num_arms, 1.0, 1.0, 0.005, 0.6, static_cast<std::uint64_t>(2000 + s));
                std::mt19937_64 r2(static_cast<std::uint64_t>(9100 + s));
                for (std::size_t i = 0; i < num_arms; ++i) p.set_feature(i, make_sparse_random(feature_dim, feature_nnz, r2));
                RunOut r = run_once(*env, p, rounds, sw);
                dal_t.push_back(r.total); dal_p.push_back(r.post);
                const std::size_t d = p.detections();
                det_sum += static_cast<double>(d);
                if (d > 0) ++det_fired;
            }
            {
                auto env = mk_env(seed);
                AdaptiveConfig c; c.mode = StatsMode::Full;
                c.v1 = 1.0; c.v2 = 1.0; c.exploration_dim = feature_dim;
                AdaptiveHCB3Policy p(num_arms, c, static_cast<std::uint64_t>(7000 + s));
                std::mt19937_64 r2(static_cast<std::uint64_t>(9000 + s));
                for (std::size_t i = 0; i < num_arms; ++i) p.set_feature(i, make_sparse_random(feature_dim, feature_nnz, r2));
                RunOut r = run_once(*env, p, rounds, sw);
                full_t.push_back(r.total); full_p.push_back(r.post);
            }
        }

        const double disc_m = mean_of(disc_t), sw_m = mean_of(swucb_t);
        const double dal_m = mean_of(dal_t), full_m = mean_of(full_t);
        const double disc_pm = mean_of(disc_p), sw_pm = mean_of(swucb_p);
        std::cout << std::left
                  << std::setw(7) << f2(delta)
                  << std::setw(20) << (f2(disc_m) + "±" + f2(std_of(disc_t)))
                  << std::setw(20) << (f2(disc_pm) + "±" + f2(std_of(disc_p)))
                  << std::setw(20) << (f2(sw_m) + "±" + f2(std_of(swucb_t)))
                  << std::setw(20) << (f2(sw_pm) + "±" + f2(std_of(swucb_p)))
                  << std::setw(12) << f1(det_sum / num_seeds)
                  << std::setw(12) << (std::to_string(det_fired) + "/" + std::to_string(num_seeds))
                  << std::setw(20) << (f2(dal_m) + "±" + f2(std_of(dal_t)))
                  << std::setw(20) << (f2(full_m) + "±" + f2(std_of(full_t)))
                  << std::setw(16) << f2(full_m - dal_m)
                  << "\n";
    }

    std::cout << "\nNote: Disc-SW>0 means HCB3 beats SW-UCB; Full-DAL>0 means active detector beats Full\n";
    std::cout << "Delta sweep done.\n";
    return 0;
}
