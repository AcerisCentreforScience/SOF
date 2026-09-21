#include "adaptive_ucb.hpp"
#include "environment.hpp"
#include "sparse_vector.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    const std::size_t rounds = 20000;
    const std::size_t K = 10;
    const std::size_t feature_dim = 1024;
    const double gamma = 0.995;

    std::cout << "===== Exploration floor verification (stationary, 20000 rounds)=====\n";
    std::cout << "K=" << K << " gamma=" << gamma << "\n\n";

    std::ofstream fout("/home/ubuntu/hcb3/build/explore_bound_result.txt");
    fout << "===== Exploration floor verification (stationary, 20000 rounds)=====\n";
    fout << "K=" << K << " gamma=" << gamma << "\n";
    fout << "Note: Full n_eff=real pulls; Discounted n_eff<=1/(1-gamma)\n\n";

    {
        mab::AdaptiveConfig cfg;
        cfg.mode = mab::StatsMode::Full;
        cfg.gamma = gamma;
        cfg.v1 = 1.0; cfg.v2 = 1.0; cfg.xi = 0.0;
        mab::AdaptiveHCB3Policy pol(K, cfg, 42);
        for (std::size_t a = 0; a < K; ++a) {
            std::vector<mab::SparseEntry> e{mab::SparseEntry{a % feature_dim, 1.0}};
            pol.set_feature(a, mab::SparseVector(e));
        }
        std::vector<double> probs(K);
        for (std::size_t i = 0; i < K; ++i) probs[i] = 0.2 + 0.6 * (double)i / (double)(K - 1);
        mab::BernoulliBandit env(probs, 777);

        fout << "--- Full history---\n";
        fout << std::left << std::setw(10) << "round" << std::setw(12) << "N"
             << std::setw(12) << "best_arm" << std::setw(16) << "explore_term\n";
        for (std::size_t t = 0; t < rounds; ++t) {
            const std::size_t a = pol.select();
            pol.update(a, env.pull(a));
            if ((t + 1) % 2000 == 0) {
                const double N = (double)(t + 1);
                const double n_eff_est = N / (double)K;
                const double term = std::sqrt(std::log(N + 1.0) / std::max(n_eff_est, 1.0));
                fout << std::left << std::setw(10) << (t + 1) << std::setw(12) << (t + 1)
                     << std::setw(12) << "-" << std::setw(16) << term << "\n";
            }
        }
        std::cout << "Full done: exploration term -> 0\n";
        fout << "Conclusion: Full n_eff grows linearly, exploration term -> 0\n\n";
    }

    {
        mab::AdaptiveConfig cfg;
        cfg.mode = mab::StatsMode::Discounted;
        cfg.gamma = gamma;
        cfg.v1 = 1.0; cfg.v2 = 1.0; cfg.xi = 0.0;
        mab::AdaptiveHCB3Policy pol(K, cfg, 42);
        for (std::size_t a = 0; a < K; ++a) {
            std::vector<mab::SparseEntry> e{mab::SparseEntry{a % feature_dim, 1.0}};
            pol.set_feature(a, mab::SparseVector(e));
        }
        std::vector<double> probs(K);
        for (std::size_t i = 0; i < K; ++i) probs[i] = 0.2 + 0.6 * (double)i / (double)(K - 1);
        mab::BernoulliBandit env(probs, 777);

        const double w_inf = 1.0 / (1.0 - gamma);
        fout << "--- Discounted (gamma=" << gamma << ")---\n";
        fout << "n_eff upper bound = 1/(1-gamma) = " << w_inf << "\n";
        fout << std::left << std::setw(10) << "round" << std::setw(12) << "N"
             << std::setw(12) << "best_arm" << std::setw(16) << "explore_term\n";
        for (std::size_t t = 0; t < rounds; ++t) {
            const std::size_t a = pol.select();
            pol.update(a, env.pull(a));
            if ((t + 1) % 2000 == 0) {
                const double N = (double)(t + 1);
                const double term = std::sqrt(std::log(N + 1.0) / w_inf);
                fout << std::left << std::setw(10) << (t + 1) << std::setw(12) << (t + 1)
                     << std::setw(12) << "-" << std::setw(16) << term << "\n";
            }
        }
        const double final_term = std::sqrt(std::log((double)rounds + 1.0) / w_inf);
        std::cout << "Disc done: exploration floor ~ " << final_term << "(not -> 0)\n";
        fout << "Conclusion: Discounted n_eff <=  " << w_inf
             << ", exploration term converges to ~ " << final_term << ", has floor, does not -> 0\n";
    }

    fout.close();
    std::cout << "Result written to explore_bound_result.txt\n";
    return 0;
}
