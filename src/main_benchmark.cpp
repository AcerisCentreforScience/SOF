#include "adaptive_ucb.hpp"
#include "advanced_policies.hpp"
#include "baselines.hpp"
#include "nonstationary.hpp"
#include "nonstationary_policies.hpp"
#include "sparse_vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace mab;

template <typename T>
double mean_of(const std::vector<T>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (auto x : v) s += static_cast<double>(x);
    return s / static_cast<double>(v.size());
}

template <typename T>
double std_of(const std::vector<T>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean_of(v);
    double s = 0.0;
    for (auto x : v) {
        const double d = static_cast<double>(x) - m;
        s += d * d;
    }
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

std::string f2(double v) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << v;
    return o.str();
}
std::string f3(double v) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(3) << v;
    return o.str();
}
std::string f4(double v) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(4) << v;
    return o.str();
}

struct SingleRun {
    double total_regret{0.0};
    double post_switch_regret{0.0};
    long long adapt_rounds{-1};
    double last_ratio{0.0};
    std::vector<double> window_curve;
};

SingleRun run_once(NonStationaryEnvironment& env,
                   Policy& policy,
                   std::size_t rounds,
                   long long switch_round,
                   std::size_t window,
                   std::size_t streak_needed,
                   double ratio_threshold) {
    SingleRun r;
    const std::size_t last_phase = std::max<std::size_t>(rounds / 10, 1);
    std::size_t last_best{0}, last_count{0};

    std::size_t win_best{0}, win_count{0};
    double win_regret{0.0};
    std::size_t streak{0};

    for (std::size_t t = 0; t < rounds; ++t) {
        env.set_round(t);
        const std::size_t arm = policy.select();
        const double reward = env.pull(arm);
        policy.update(arm, reward);

        const std::size_t best = env.best_arm();
        const double regret = env.best_mean() - env.true_mean(arm);
        r.total_regret += regret;
        if (switch_round >= 0 && static_cast<long long>(t) >= switch_round) {
            r.post_switch_regret += regret;
        }

        const bool is_best = (arm == best);
        if (t >= rounds - last_phase) { if (is_best) ++last_best; ++last_count; }
        if (is_best) ++win_best;
        ++win_count;
        win_regret += regret;

        if (win_count == window) {
            const double ratio = static_cast<double>(win_best) / static_cast<double>(window);
            r.window_curve.push_back(win_regret / static_cast<double>(window));
            if (switch_round >= 0 && static_cast<long long>(t) >= switch_round && r.adapt_rounds < 0) {
                if (ratio >= ratio_threshold) {
                    ++streak;
                    if (streak >= streak_needed) r.adapt_rounds = static_cast<long long>(t) + 1 - switch_round;
                } else {
                    streak = 0;
                }
            }
            win_best = 0; win_count = 0; win_regret = 0.0;
        }
    }
    if (win_count > 0) r.window_curve.push_back(win_regret / static_cast<double>(win_count));
    r.last_ratio = last_count == 0 ? 0.0 : static_cast<double>(last_best) / static_cast<double>(last_count);
    return r;
}

struct Agg {
    double total_mean{0.0}, total_std{0.0};
    double post_mean{0.0}, post_std{0.0};
    double adapt_mean{-1.0};
    int adapt_count{0};
    double last_mean{0.0}, last_std{0.0};
    std::vector<double> curve;
};

using EnvFactory = std::function<std::shared_ptr<NonStationaryEnvironment>(std::uint64_t)>;
using PolicyFactory = std::function<std::shared_ptr<Policy>(std::uint64_t)>;

Agg run_multi(const EnvFactory& env_factory,
              const PolicyFactory& policy_factory,
              std::size_t rounds,
              long long switch_round,
              std::size_t window,
              std::size_t streak_needed,
              double ratio_threshold,
              int num_seeds) {
    Agg a;
    std::vector<double> totals, posts, lasts;
    std::vector<long long> adapts;
    std::vector<std::vector<double>> curves;

    for (int s = 0; s < num_seeds; ++s) {
        const std::uint64_t seed = static_cast<std::uint64_t>(s + 1);
        auto env = env_factory(seed);
        auto policy = policy_factory(seed);
        SingleRun r = run_once(*env, *policy, rounds, switch_round, window, streak_needed, ratio_threshold);
        totals.push_back(r.total_regret);
        posts.push_back(r.post_switch_regret);
        lasts.push_back(r.last_ratio);
        const bool achieved = r.adapt_rounds >= 0;
        adapts.push_back(achieved ? r.adapt_rounds : -1);
        if (achieved) ++a.adapt_count;
        curves.push_back(r.window_curve);
    }

    a.total_mean = mean_of(totals); a.total_std = std_of(totals);
    a.post_mean = mean_of(posts); a.post_std = std_of(posts);
    a.last_mean = mean_of(lasts); a.last_std = std_of(lasts);
    {
        std::vector<long long> only;
        for (auto x : adapts) if (x >= 0) only.push_back(x);
        a.adapt_mean = only.empty() ? -1.0 : mean_of(only);
    }

    std::size_t max_len = 0;
    for (auto& c : curves) max_len = std::max(max_len, c.size());
    a.curve.assign(max_len, 0.0);
    for (std::size_t i = 0; i < max_len; ++i) {
        double s = 0.0; int cnt = 0;
        for (auto& c : curves) if (i < c.size()) { s += c[i]; ++cnt; }
        a.curve[i] = cnt == 0 ? 0.0 : s / static_cast<double>(cnt);
    }
    return a;
}

void add_features_adaptive(AdaptiveHCB3Policy& p, std::size_t K, std::size_t dim, std::size_t nnz, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < K; ++i) p.set_feature(i, make_sparse_random(dim, nnz, rng));
}
void add_features_dal(DAL_HCB3& p, std::size_t K, std::size_t dim, std::size_t nnz, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < K; ++i) p.set_feature(i, make_sparse_random(dim, nnz, rng));
}
void add_features_swlin(SWLinUCB& p, std::size_t K, std::size_t dim, std::size_t nnz, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < K; ++i) p.set_feature(i, make_sparse_random(dim, nnz, rng));
}
void add_features_adlasso(ADLasso& p, std::size_t K, std::size_t dim, std::size_t nnz, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < K; ++i) p.set_feature(i, make_sparse_random(dim, nnz, rng));
}

struct Row {
    std::string name;
    Agg agg;
};

void print_table(const std::string& title, const std::vector<Row>& rows) {
    std::cout << "\n===== " << title << " =====\n";
    std::cout << std::left << std::setw(24) << "Algorithm"
              << std::setw(20) << "CumRegret(mean+/-std)"
              << std::setw(20) << "PostSwitchRegret"
              << std::setw(18) << "Adapt rounds (mean/achieved)"
              << std::setw(18) << "LastSegBest%" << "\n";
    std::cout << std::string(100, '-') << "\n";
    for (const auto& r : rows) {
        const std::string total = f2(r.agg.total_mean) + "±" + f2(r.agg.total_std);
        const std::string post = f2(r.agg.post_mean) + "±" + f2(r.agg.post_std);
        std::string adapt = r.agg.adapt_mean < 0 ? std::string("—")
            : (f2(r.agg.adapt_mean) + "/" + std::to_string(r.agg.adapt_count) + "0%");
        adapt = r.agg.adapt_mean < 0 ? std::string("—") + "/" + std::to_string(r.agg.adapt_count)
            : (f2(r.agg.adapt_mean) + "/" + std::to_string(r.agg.adapt_count));
        const std::string last = f4(r.agg.last_mean * 100.0) + "%" ;
        std::cout << std::left << std::setw(24) << r.name
                  << std::setw(20) << total
                  << std::setw(20) << post
                  << std::setw(18) << adapt
                  << std::setw(18) << last << "\n";
    }
}

void print_curves(const std::string& title, const std::vector<Row>& rows) {
    std::cout << "\n===== " << title << " =====\n";
    std::cout << std::left << std::setw(24) << "Algorithm"
              << std::setw(16) << "first"
              << std::setw(16) << "mid"
              << std::setw(16) << "last"
              << std::setw(16) << "overall" << "\n";
    std::cout << std::string(90, '-') << "\n";
    for (const auto& r : rows) {
        const auto& c = r.agg.curve;
        if (c.empty()) continue;
        const std::size_t n = c.size();
        const std::size_t seg = std::max<std::size_t>(n / 3, 1);
        auto m = [&](std::size_t s, std::size_t e) {
            double sum = 0.0; std::size_t cnt = 0;
            for (std::size_t i = s; i < e && i < n; ++i) { sum += c[i]; ++cnt; }
            return cnt == 0 ? 0.0 : sum / static_cast<double>(cnt);
        };
        std::cout << std::left << std::setw(24) << r.name
                  << std::setw(16) << f4(m(0, seg))
                  << std::setw(16) << f4(m(seg, 2 * seg))
                  << std::setw(16) << f4(m(2 * seg, n))
                  << std::setw(16) << f4(m(0, n)) << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t rounds = 20000;
    std::size_t num_arms = 10;
    std::size_t feature_dim = 1024;
    std::size_t feature_nnz = 8;
    std::size_t window = 500;
    std::size_t streak_needed = 3;
    double ratio_threshold = 0.8;
    double gamma = 0.995;
    int num_seeds = 5;

    if (argc > 1) rounds = static_cast<std::size_t>(std::stoull(argv[1]));
    if (argc > 2) num_arms = static_cast<std::size_t>(std::stoull(argv[2]));
    if (argc > 3) feature_dim = static_cast<std::size_t>(std::stoull(argv[3]));
    if (argc > 4) window = static_cast<std::size_t>(std::stoull(argv[4]));
    if (argc > 5) num_seeds = static_cast<int>(std::stoi(argv[5]));

    std::cout << "HCB3 multi-seed benchmark\n";
    std::cout << "Params: rounds=" << rounds << " arms=" << num_arms
              << " feat_dim=" << feature_dim << " nnz/arm=" << feature_nnz
              << " detector window=" << window << " streak=" << streak_needed
              << " seeds=" << num_seeds << "\n";

    const std::size_t switch_round = rounds / 2;
    std::vector<double> means = make_spaced_means(num_arms, 0.20, 0.80);
    std::size_t best_arm = 0, second_arm = 1;
    for (std::size_t i = 0; i < num_arms; ++i) {
        if (means[i] > means[best_arm]) { second_arm = best_arm; best_arm = i; }
        else if (i != best_arm && means[i] > means[second_arm]) { second_arm = i; }
    }

    std::vector<Row> abrupt_rows;

    auto push_adaptive = [&](std::string name, StatsMode mode, std::size_t win, double gm) {
        EnvFactory ef = [&](std::uint64_t s) -> std::shared_ptr<NonStationaryEnvironment> {
            return std::make_shared<AbruptSwitchBandit>(means, switch_round, best_arm, second_arm,
                                                        static_cast<std::uint64_t>(300 + s));
        };
        PolicyFactory pf = [&, mode, win, gm](std::uint64_t s) -> std::shared_ptr<Policy> {
            AdaptiveConfig cfg;
            cfg.mode = mode; cfg.window_size = win; cfg.gamma = gm;
            cfg.exploration_dim = feature_dim; cfg.v1 = 1.0; cfg.v2 = 1.0; cfg.xi = 0.0;
            auto p = std::make_shared<AdaptiveHCB3Policy>(num_arms, cfg, static_cast<std::uint64_t>(7000 + s));
            add_features_adaptive(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9000 + s));
            return p;
        };
        abrupt_rows.push_back({name, run_multi(ef, pf, rounds, static_cast<long long>(switch_round), window, streak_needed, ratio_threshold, num_seeds)});
    };

    auto push_simple = [&](std::string name, PolicyFactory pf) {
        EnvFactory ef = [&](std::uint64_t s) -> std::shared_ptr<NonStationaryEnvironment> {
            return std::make_shared<AbruptSwitchBandit>(means, switch_round, best_arm, second_arm,
                                                        static_cast<std::uint64_t>(300 + s));
        };
        abrupt_rows.push_back({name, run_multi(ef, pf, rounds, static_cast<long long>(switch_round), window, streak_needed, ratio_threshold, num_seeds)});
    };

    push_adaptive("HCB3 Full", StatsMode::Full, window, gamma);
    push_adaptive("HCB3 SlidingWindow(500)", StatsMode::SlidingWindow, window, gamma);
    push_adaptive("HCB3 Discounted(0.995)", StatsMode::Discounted, window, gamma);

    push_simple("SW-UCB(500)", [&](std::uint64_t s){ return std::make_shared<SlidingWindowUCB>(num_arms, window, 1000 + s); });
    push_simple("Epsilon-Greedy(0.1)", [&](std::uint64_t s){ return std::make_shared<EpsilonGreedyPolicy>(num_arms, 0.1, 1100 + s); });

    const std::size_t intervals[] = {250, 500, 1000, 2000, 5000, 10000};
    for (std::size_t iv : intervals) {
        push_simple("RestartUCB(" + std::to_string(iv) + ")", [&, iv](std::uint64_t s){
            return std::make_shared<RestartUCB>(num_arms, iv, 1200 + s);
        });
    }

    {
        EnvFactory ef = [&](std::uint64_t s) -> std::shared_ptr<NonStationaryEnvironment> {
            return std::make_shared<AbruptSwitchBandit>(means, switch_round, best_arm, second_arm, static_cast<std::uint64_t>(300 + s));
        };
        PolicyFactory pf = [&](std::uint64_t s) -> std::shared_ptr<Policy> {
            auto p = std::make_shared<DAL_HCB3>(num_arms, 1.0, 1.0, 0.005, 0.6, static_cast<std::uint64_t>(2000 + s));
            add_features_dal(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9100 + s));
            return p;
        };
        abrupt_rows.push_back({"DAL+HCB3 (P0)", run_multi(ef, pf, rounds, static_cast<long long>(switch_round), window, streak_needed, ratio_threshold, num_seeds)});
    }
    push_simple("TS-CD (P0)", [&](std::uint64_t s){ return std::make_shared<TSCD>(num_arms, 0.005, 0.6, 2100 + s); });

    push_simple("BOB-SW-UCB (P1)", [&](std::uint64_t s){
        return std::make_shared<BOBSWUCB>(num_arms, std::vector<std::size_t>{100, 250, 500, 1000, 2000}, 1000, 2200 + s);
    });
    push_simple("SW-MOSS(500) (P1)", [&](std::uint64_t s){ return std::make_shared<SWMOSS>(num_arms, window, 2300 + s); });
    push_simple("D-TS(0.995) (P1)", [&](std::uint64_t s){ return std::make_shared<DiscountedTS>(num_arms, gamma, 2400 + s); });

    push_simple("ADR-bandit (P2)", [&](std::uint64_t s){ return std::make_shared<ADRBandit>(num_arms, 200, 5000, 0.005, 0.8, 2500 + s); });
    {
        EnvFactory ef = [&](std::uint64_t s) -> std::shared_ptr<NonStationaryEnvironment> {
            return std::make_shared<AbruptSwitchBandit>(means, switch_round, best_arm, second_arm, static_cast<std::uint64_t>(300 + s));
        };
        PolicyFactory pf = [&](std::uint64_t s) -> std::shared_ptr<Policy> {
            auto p = std::make_shared<SWLinUCB>(num_arms, window, 1.0, 1.0, static_cast<std::uint64_t>(2600 + s));
            add_features_swlin(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9200 + s));
            return p;
        };
        abrupt_rows.push_back({"SW-LinUCB (P2)", run_multi(ef, pf, rounds, static_cast<long long>(switch_round), window, streak_needed, ratio_threshold, num_seeds)});
    }
    {
        EnvFactory ef = [&](std::uint64_t s) -> std::shared_ptr<NonStationaryEnvironment> {
            return std::make_shared<AbruptSwitchBandit>(means, switch_round, best_arm, second_arm, static_cast<std::uint64_t>(300 + s));
        };
        PolicyFactory pf = [&](std::uint64_t s) -> std::shared_ptr<Policy> {
            auto p = std::make_shared<ADLasso>(num_arms, window, 1.0, 0.02, static_cast<std::uint64_t>(2700 + s));
            add_features_adlasso(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9300 + s));
            return p;
        };
        abrupt_rows.push_back({"AD-Lasso (P2)", run_multi(ef, pf, rounds, static_cast<long long>(switch_round), window, streak_needed, ratio_threshold, num_seeds)});
    }

    std::cout << "\n########## Abrupt env (switch at t= " << switch_round
              << "  swap best arm " << best_arm << "  with 2nd arm " << second_arm << ") ##########\n";
    print_table("Abrupt env metrics (5 seeds)", abrupt_rows);
    print_curves("Abrupt sliding regret rate (per 500 rounds)", abrupt_rows);

    std::vector<Row> drift_rows;

    auto drift_env = [&](std::uint64_t s) -> std::shared_ptr<NonStationaryEnvironment> {
        return std::make_shared<CyclicDriftBandit>(num_arms, 0.6, 0.3, 5000,
                                                   static_cast<std::uint64_t>(500 + s));
    };

    auto push_drift_adaptive = [&](std::string name, StatsMode mode, std::size_t win, double gm) {
        PolicyFactory pf = [&, mode, win, gm](std::uint64_t s) -> std::shared_ptr<Policy> {
            AdaptiveConfig cfg;
            cfg.mode = mode; cfg.window_size = win; cfg.gamma = gm; cfg.exploration_dim = feature_dim;
            auto p = std::make_shared<AdaptiveHCB3Policy>(num_arms, cfg, static_cast<std::uint64_t>(7000 + s));
            add_features_adaptive(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9000 + s));
            return p;
        };
        drift_rows.push_back({name, run_multi(drift_env, pf, rounds, -1, window, streak_needed, ratio_threshold, num_seeds)});
    };
    auto push_drift_simple = [&](std::string name, PolicyFactory pf) {
        drift_rows.push_back({name, run_multi(drift_env, pf, rounds, -1, window, streak_needed, ratio_threshold, num_seeds)});
    };

    push_drift_adaptive("HCB3 Full", StatsMode::Full, window, gamma);
    push_drift_adaptive("HCB3 SlidingWindow(500)", StatsMode::SlidingWindow, window, gamma);
    push_drift_adaptive("HCB3 Discounted(0.995)", StatsMode::Discounted, window, gamma);
    push_drift_simple("SW-UCB(500)", [&](std::uint64_t s){ return std::make_shared<SlidingWindowUCB>(num_arms, window, 1000 + s); });
    push_drift_simple("Epsilon-Greedy(0.1)", [&](std::uint64_t s){ return std::make_shared<EpsilonGreedyPolicy>(num_arms, 0.1, 1100 + s); });
    for (std::size_t iv : intervals) {
        push_drift_simple("RestartUCB(" + std::to_string(iv) + ")", [&, iv](std::uint64_t s){
            return std::make_shared<RestartUCB>(num_arms, iv, 1200 + s);
        });
    }
    {
        PolicyFactory pf = [&](std::uint64_t s) -> std::shared_ptr<Policy> {
            auto p = std::make_shared<DAL_HCB3>(num_arms, 1.0, 1.0, 0.005, 0.6, static_cast<std::uint64_t>(2000 + s));
            add_features_dal(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9100 + s));
            return p;
        };
        drift_rows.push_back({"DAL+HCB3 (P0)", run_multi(drift_env, pf, rounds, -1, window, streak_needed, ratio_threshold, num_seeds)});
    }
    push_drift_simple("TS-CD (P0)", [&](std::uint64_t s){ return std::make_shared<TSCD>(num_arms, 0.005, 0.6, 2100 + s); });
    push_drift_simple("BOB-SW-UCB (P1)", [&](std::uint64_t s){
        return std::make_shared<BOBSWUCB>(num_arms, std::vector<std::size_t>{100, 250, 500, 1000, 2000}, 1000, 2200 + s);
    });
    push_drift_simple("SW-MOSS(500) (P1)", [&](std::uint64_t s){ return std::make_shared<SWMOSS>(num_arms, window, 2300 + s); });
    push_drift_simple("D-TS(0.995) (P1)", [&](std::uint64_t s){ return std::make_shared<DiscountedTS>(num_arms, gamma, 2400 + s); });
    push_drift_simple("ADR-bandit (P2)", [&](std::uint64_t s){ return std::make_shared<ADRBandit>(num_arms, 200, 5000, 0.005, 0.8, 2500 + s); });
    {
        PolicyFactory pf = [&](std::uint64_t s) -> std::shared_ptr<Policy> {
            auto p = std::make_shared<SWLinUCB>(num_arms, window, 1.0, 1.0, static_cast<std::uint64_t>(2600 + s));
            add_features_swlin(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9200 + s));
            return p;
        };
        drift_rows.push_back({"SW-LinUCB (P2)", run_multi(drift_env, pf, rounds, -1, window, streak_needed, ratio_threshold, num_seeds)});
    }
    {
        PolicyFactory pf = [&](std::uint64_t s) -> std::shared_ptr<Policy> {
            auto p = std::make_shared<ADLasso>(num_arms, window, 1.0, 0.02, static_cast<std::uint64_t>(2700 + s));
            add_features_adlasso(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9300 + s));
            return p;
        };
        drift_rows.push_back({"AD-Lasso (P2)", run_multi(drift_env, pf, rounds, -1, window, streak_needed, ratio_threshold, num_seeds)});
    }

    std::cout << "\n########## Cyclic drift (0.3-0.9 sine, period 5000) ##########\n";
    print_table("Cyclic drift metrics (5 seeds, no switch point)", drift_rows);
    print_curves("Cyclic sliding regret rate (per 500 rounds)", drift_rows);

    std::cout << "\nBenchmark done.\n";
    return 0;
}
