#include "adaptive_ucb.hpp"
#include "advanced_policies.hpp"
#include "baselines.hpp"
#include "environment.hpp"
#include "nonstationary_policies.hpp"
#include "sparse_vector.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
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
std::string f4(double v) { std::ostringstream o; o << std::fixed << std::setprecision(4) << v; return o.str(); }

struct RunOut {
    double total_regret{0.0};
    double last_ratio{0.0};
    std::vector<double> curve;
};

RunOut run_once(BanditEnvironment& env, Policy& policy, std::size_t rounds, std::size_t window) {
    RunOut r;
    const std::size_t last_phase = std::max<std::size_t>(rounds / 10, 1);
    std::size_t last_best{0}, last_count{0};
    std::size_t win_count{0};
    double win_regret{0.0};
    for (std::size_t t = 0; t < rounds; ++t) {
        const std::size_t arm = policy.select();
        const double reward = env.pull(arm);
        policy.update(arm, reward);
        const double regret = env.best_mean() - env.true_mean(arm);
        r.total_regret += regret;
        if (t >= rounds - last_phase) { if (arm == env.best_arm()) ++last_best; ++last_count; }
        ++win_count; win_regret += regret;
        if (win_count == window) {
            r.curve.push_back(win_regret / static_cast<double>(window));
            win_count = 0; win_regret = 0.0;
        }
    }
    if (win_count > 0) r.curve.push_back(win_regret / static_cast<double>(win_count));
    r.last_ratio = last_count == 0 ? 0.0 : static_cast<double>(last_best) / static_cast<double>(last_count);
    return r;
}

struct Agg {
    double total_mean{0.0}, total_std{0.0};
    double last_mean{0.0}, last_std{0.0};
    double curve_front{0.0}, curve_mid{0.0}, curve_end{0.0};
};

using EnvFactory = std::function<std::shared_ptr<BanditEnvironment>(std::uint64_t)>;
using PolicyFactory = std::function<std::shared_ptr<Policy>(std::uint64_t)>;

Agg run_multi(const EnvFactory& ef, const PolicyFactory& pf, std::size_t rounds,
              std::size_t window, int num_seeds) {
    std::vector<double> totals, lasts;
    std::vector<std::vector<double>> curves;
    for (int s = 0; s < num_seeds; ++s) {
        const std::uint64_t seed = static_cast<std::uint64_t>(s + 1);
        auto env = ef(seed);
        auto policy = pf(seed);
        RunOut r = run_once(*env, *policy, rounds, window);
        totals.push_back(r.total_regret);
        lasts.push_back(r.last_ratio);
        curves.push_back(r.curve);
    }
    Agg a;
    a.total_mean = mean_of(totals); a.total_std = std_of(totals);
    a.last_mean = mean_of(lasts); a.last_std = std_of(lasts);
    std::size_t max_len = 0;
    for (auto& c : curves) max_len = std::max(max_len, c.size());
    if (max_len > 0) {
        std::vector<double> avg(max_len, 0.0);
        for (std::size_t i = 0; i < max_len; ++i) {
            double s = 0.0; int cnt = 0;
            for (auto& c : curves) if (i < c.size()) { s += c[i]; ++cnt; }
            avg[i] = cnt == 0 ? 0.0 : s / static_cast<double>(cnt);
        }
        const std::size_t seg = std::max<std::size_t>(max_len / 3, 1);
        auto segm = [&](std::size_t b, std::size_t e) {
            double s = 0.0; std::size_t cnt = 0;
            for (std::size_t i = b; i < e && i < max_len; ++i) { s += avg[i]; ++cnt; }
            return cnt == 0 ? 0.0 : s / static_cast<double>(cnt);
        };
        a.curve_front = segm(0, seg);
        a.curve_mid = segm(seg, 2 * seg);
        a.curve_end = segm(2 * seg, max_len);
    }
    return a;
}

struct Row { std::string name; Agg agg; };

void print_table(const std::string& title, const std::vector<Row>& rows) {
    std::cout << "\n===== " << title << " =====\n";
    std::cout << std::left << std::setw(26) << "Algorithm"
              << std::setw(22) << "CumRegret(mean+/-std)"
              << std::setw(20) << "LastSegBest%"
              << std::setw(30) << "RegretRate(first/mid/last)" << "\n";
    std::cout << std::string(98, '-') << "\n";
    for (const auto& r : rows) {
        std::cout << std::left << std::setw(26) << r.name
                  << std::setw(22) << (f2(r.agg.total_mean) + "±" + f2(r.agg.total_std))
                  << std::setw(20) << (f2(r.agg.last_mean * 100.0) + "%±" + f2(r.agg.last_std * 100.0))
                  << std::setw(30) << (f4(r.agg.curve_front) + "/" + f4(r.agg.curve_mid) + "/" + f4(r.agg.curve_end))
                  << "\n";
    }
}

void add_feat_adaptive(AdaptiveHCB3Policy& p, std::size_t K, std::size_t dim, std::size_t nnz, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < K; ++i) p.set_feature(i, make_sparse_random(dim, nnz, rng));
}
void add_feat_dal(DAL_HCB3& p, std::size_t K, std::size_t dim, std::size_t nnz, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < K; ++i) p.set_feature(i, make_sparse_random(dim, nnz, rng));
}
void add_feat_swlin(SWLinUCB& p, std::size_t K, std::size_t dim, std::size_t nnz, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < K; ++i) p.set_feature(i, make_sparse_random(dim, nnz, rng));
}
void add_feat_adlasso(ADLasso& p, std::size_t K, std::size_t dim, std::size_t nnz, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < K; ++i) p.set_feature(i, make_sparse_random(dim, nnz, rng));
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t rounds = 20000;
    std::size_t num_arms = 10;
    std::size_t feature_dim = 1024;
    std::size_t feature_nnz = 8;
    std::size_t window = 500;
    int num_seeds = 5;
    double gamma = 0.995;
    if (argc > 1) rounds = static_cast<std::size_t>(std::stoull(argv[1]));

    std::cout << "HCB3 stationary benchmark (S3 + S4 ablation)\n";
    std::cout << "Params: rounds=" << rounds << " arms=" << num_arms
              << " feat_dim=" << feature_dim << " nnz/arm=" << feature_nnz
              << " seeds=" << num_seeds << " env=Bernoulli(0.2~0.8, stationary)\n";

    const std::vector<double> probs = make_spaced_means(num_arms, 0.2, 0.8);
    EnvFactory env_of = [&](std::uint64_t s) -> std::shared_ptr<BanditEnvironment> {
        return std::make_shared<BernoulliBandit>(probs, static_cast<std::uint64_t>(300 + s));
    };

    std::vector<Row> s3;
    auto hcb3_factory = [&](StatsMode mode, std::size_t win, double gm) {
        return [&, mode, win, gm](std::uint64_t s) -> std::shared_ptr<Policy> {
            AdaptiveConfig cfg;
            cfg.mode = mode; cfg.window_size = win; cfg.gamma = gm;
            cfg.v1 = 1.0; cfg.v2 = 1.0; cfg.exploration_dim = feature_dim;
            auto p = std::make_shared<AdaptiveHCB3Policy>(num_arms, cfg, static_cast<std::uint64_t>(7000 + s));
            add_feat_adaptive(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9000 + s));
            return p;
        };
    };
    s3.push_back({"HCB3 Full", run_multi(env_of, hcb3_factory(StatsMode::Full, window, gamma), rounds, window, num_seeds)});
    s3.push_back({"HCB3 SlidingWindow(500)", run_multi(env_of, hcb3_factory(StatsMode::SlidingWindow, window, gamma), rounds, window, num_seeds)});
    s3.push_back({"HCB3 Discounted(0.995)", run_multi(env_of, hcb3_factory(StatsMode::Discounted, window, gamma), rounds, window, num_seeds)});
    s3.push_back({"SW-UCB(500)", run_multi(env_of, [&](std::uint64_t s){ return std::make_shared<SlidingWindowUCB>(num_arms, window, 1000 + s); }, rounds, window, num_seeds)});
    s3.push_back({"Epsilon-Greedy(0.1)", run_multi(env_of, [&](std::uint64_t s){ return std::make_shared<EpsilonGreedyPolicy>(num_arms, 0.1, 1100 + s); }, rounds, window, num_seeds)});
    const std::size_t intervals[] = {250, 500, 1000, 2000, 5000, 10000};
    for (std::size_t iv : intervals) {
        s3.push_back({"RestartUCB(" + std::to_string(iv) + ")", run_multi(env_of, [&, iv](std::uint64_t s){
            return std::make_shared<RestartUCB>(num_arms, iv, 1200 + s); }, rounds, window, num_seeds)});
    }
    s3.push_back({"DAL+HCB3 (P0)", run_multi(env_of, [&](std::uint64_t s) -> std::shared_ptr<Policy> {
        auto p = std::make_shared<DAL_HCB3>(num_arms, 1.0, 1.0, 0.005, 0.6, static_cast<std::uint64_t>(2000 + s));
        add_feat_dal(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9100 + s));
        return p;
    }, rounds, window, num_seeds)});
    s3.push_back({"TS-CD (P0)", run_multi(env_of, [&](std::uint64_t s){ return std::make_shared<TSCD>(num_arms, 0.005, 0.6, 2100 + s); }, rounds, window, num_seeds)});
    s3.push_back({"BOB-SW-UCB (P1)", run_multi(env_of, [&](std::uint64_t s){
        return std::make_shared<BOBSWUCB>(num_arms, std::vector<std::size_t>{100, 250, 500, 1000, 2000}, 1000, 2200 + s);
    }, rounds, window, num_seeds)});
    s3.push_back({"SW-MOSS(500) (P1)", run_multi(env_of, [&](std::uint64_t s){ return std::make_shared<SWMOSS>(num_arms, window, 2300 + s); }, rounds, window, num_seeds)});
    s3.push_back({"D-TS(0.995) (P1)", run_multi(env_of, [&](std::uint64_t s){ return std::make_shared<DiscountedTS>(num_arms, gamma, 2400 + s); }, rounds, window, num_seeds)});
    s3.push_back({"ADR-bandit (P2)", run_multi(env_of, [&](std::uint64_t s){ return std::make_shared<ADRBandit>(num_arms, 200, 5000, 0.005, 0.8, 2500 + s); }, rounds, window, num_seeds)});
    s3.push_back({"SW-LinUCB (P2)", run_multi(env_of, [&](std::uint64_t s) -> std::shared_ptr<Policy> {
        auto p = std::make_shared<SWLinUCB>(num_arms, window, 1.0, 1.0, static_cast<std::uint64_t>(2600 + s));
        add_feat_swlin(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9200 + s));
        return p;
    }, rounds, window, num_seeds)});
    s3.push_back({"AD-Lasso (P2)", run_multi(env_of, [&](std::uint64_t s) -> std::shared_ptr<Policy> {
        auto p = std::make_shared<ADLasso>(num_arms, window, 1.0, 0.02, static_cast<std::uint64_t>(2700 + s));
        add_feat_adlasso(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9300 + s));
        return p;
    }, rounds, window, num_seeds)});

    print_table("S3 stationary full benchmark (5 seeds)", s3);

    std::vector<Row> s4;
    auto run_abl = [&](const std::string& name, AdaptiveConfig cfg) {
        PolicyFactory pf = [&, cfg](std::uint64_t s) -> std::shared_ptr<Policy> {
            auto p = std::make_shared<AdaptiveHCB3Policy>(num_arms, cfg, static_cast<std::uint64_t>(7000 + s));
            add_feat_adaptive(*p, num_arms, feature_dim, feature_nnz, static_cast<std::uint64_t>(9000 + s));
            return p;
        };
        s4.push_back({name, run_multi(env_of, pf, rounds, window, num_seeds)});
    };

    {
        AdaptiveConfig base;
        base.mode = StatsMode::Discounted; base.gamma = 0.995;
        base.v1 = 1.0; base.v2 = 1.0; base.exploration_dim = feature_dim;
        auto c1 = base; c1.sparse_correction = false;
        auto c2 = base; c2.v1 = 0.0;
        auto c3 = base; c3.v2 = 0.0;
        auto c4 = base; c4.v1 = 0.0; c4.v2 = 0.0;
        auto c5 = base; c5.cap_variance = false;
        run_abl("Full HCB3-Disc", base);
        run_abl("- sparse corr", c1);
        run_abl("- v1=0", c2);
        run_abl("- v2=0", c3);
        run_abl("- no 2nd-order", c4);
        run_abl("- no cap", c5);
    }
    {
        AdaptiveConfig base;
        base.mode = StatsMode::Full;
        base.v1 = 1.0; base.v2 = 1.0; base.exploration_dim = feature_dim;
        auto c1 = base; c1.sparse_correction = false;
        auto c4 = base; c4.v1 = 0.0; c4.v2 = 0.0;
        auto c5 = base; c5.cap_variance = false;
        run_abl("Full HCB3-Full", base);
        run_abl("Full - sparse corr", c1);
        run_abl("Full - no 2nd-order", c4);
        run_abl("Full - no cap", c5);
    }
    {
        AdaptiveConfig base;
        base.mode = StatsMode::SlidingWindow;
        base.v1 = 1.0; base.v2 = 1.0; base.exploration_dim = feature_dim;
        for (std::size_t w : {250UL, 500UL, 1000UL}) {
            auto c = base; c.window_size = w;
            run_abl("HCB3-SW window=" + std::to_string(w), c);
        }
    }
    {
        AdaptiveConfig base;
        base.mode = StatsMode::Discounted;
        base.v1 = 1.0; base.v2 = 1.0; base.exploration_dim = feature_dim;
        for (double g : {0.99, 0.995, 0.999}) {
            auto c = base; c.gamma = g;
            std::ostringstream nm; nm << "HCB3-Disc gamma=" << g;
            run_abl(nm.str(), c);
        }
    }

    print_table("S4 stationary ablation (5 seeds)", s4);

    std::cout << "\nStationary test done.\n";
    return 0;
}
