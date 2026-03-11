// Sakina Saidi
// Created: Feb 24, 2026
//
// Stochastic Series Expansion (SSE) Quantum Monte Carlo
// 2D spin-1/2 antiferromagnetic Heisenberg model on a square lattice
// with periodic boundary conditions.
//
// H = J * sum_{<ij>} S_i . S_j   (J > 0, antiferromagnetic)
//
// Reference: A.W. Sandvik, Phys. Rev. B 59, R14157 (1999)

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <cassert>
#include <limits>
#include <ctime>
using namespace std;

// ---------------------------------------------------------------------------
// SSE operator struct
// ---------------------------------------------------------------------------
struct Op {
    int type;   // 0 = identity/filler, 1 = diagonal (H1), 2 = off-diagonal (H2)
    int bond;   // index into bond table [0, NB)
};

// ---------------------------------------------------------------------------
// Global simulation state
// ---------------------------------------------------------------------------
static int L;                          // linear lattice size
static int N;                          // number of sites = L*L
static int NB;                         // number of bonds = 2*N
static double J_coupling;              // coupling constant
static int mc_iter;                    // measurement sweeps
static int eq_iter;                    // equilibration sweeps
static int bin_size;                   // block size for binned error
static unsigned int seed_val;          // RNG seed
static int debug_flag;                 // debug mode

static vector<int> spin;              // spin configuration, +/-1 (= 2*Sz)
static vector<Op> ops;                // operator string, length M
static int M;                         // current operator string length
static int n_ops;                     // number of non-identity operators

static vector<int> bond_site1;        // bond table: first site
static vector<int> bond_site2;        // bond table: second site

static vector<int> vtx_link;          // linked vertex list, size 4*M
static vector<int> first_vtx;         // first vertex leg per site, size N

static mt19937 rng;                   // Mersenne Twister RNG

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
struct RunParams {
    int L;
    double J;
    double beta;          // -1 means use default sweep
    int mc_iter;
    int eq_iter;
    int bin_size;
    unsigned int seed;
    int debug;
};

static bool parse_int_str(const string& s, int& out) {
    istringstream iss(s);
    iss >> out;
    return iss && iss.eof();
}

static bool parse_uint_str(const string& s, unsigned int& out) {
    unsigned long long tmp = 0;
    istringstream iss(s);
    iss >> tmp;
    if (!(iss && iss.eof())) return false;
    if (tmp > (unsigned long long)numeric_limits<unsigned int>::max()) return false;
    out = (unsigned int)tmp;
    return true;
}

static bool parse_double_str(const string& s, double& out) {
    istringstream iss(s);
    iss >> out;
    return iss && iss.eof();
}

RunParams parse_args(int argc, char* argv[]) {
    RunParams p;
    p.L = 32;
    p.J = 1.0;
    p.beta = -1.0;       // sentinel: use default sweep
    p.mc_iter = 10000;
    p.eq_iter = 2000;
    p.bin_size = 50;
    p.seed = (unsigned int)time(NULL);
    p.debug = 0;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        auto next_value = [&](const string& flag) -> string {
            if (i + 1 >= argc) {
                cerr << "Missing value for " << flag << endl;
                exit(1);
            }
            return string(argv[++i]);
        };

        if (arg == "--L") {
            int v = 0;
            string s = next_value(arg);
            if (!parse_int_str(s, v)) {
                cerr << "Invalid integer for --L: " << s << endl;
                exit(1);
            }
            p.L = v;
        }
        else if (arg == "--J") {
            double v = 0.0;
            string s = next_value(arg);
            if (!parse_double_str(s, v)) {
                cerr << "Invalid floating-point value for --J: " << s << endl;
                exit(1);
            }
            p.J = v;
        }
        else if (arg == "--beta") {
            double v = 0.0;
            string s = next_value(arg);
            if (!parse_double_str(s, v)) {
                cerr << "Invalid floating-point value for --beta: " << s << endl;
                exit(1);
            }
            p.beta = v;
        }
        else if (arg == "--mc_iter") {
            int v = 0;
            string s = next_value(arg);
            if (!parse_int_str(s, v)) {
                cerr << "Invalid integer for --mc_iter: " << s << endl;
                exit(1);
            }
            p.mc_iter = v;
        }
        else if (arg == "--eq_iter") {
            int v = 0;
            string s = next_value(arg);
            if (!parse_int_str(s, v)) {
                cerr << "Invalid integer for --eq_iter: " << s << endl;
                exit(1);
            }
            p.eq_iter = v;
        }
        else if (arg == "--bin_size") {
            int v = 0;
            string s = next_value(arg);
            if (!parse_int_str(s, v)) {
                cerr << "Invalid integer for --bin_size: " << s << endl;
                exit(1);
            }
            p.bin_size = v;
        }
        else if (arg == "--seed") {
            unsigned int v = 0;
            string s = next_value(arg);
            if (!parse_uint_str(s, v)) {
                cerr << "Invalid unsigned integer for --seed: " << s << endl;
                exit(1);
            }
            p.seed = v;
        }
        else if (arg == "--debug") {
            int v = 0;
            string s = next_value(arg);
            if (!parse_int_str(s, v)) {
                cerr << "Invalid integer for --debug: " << s << endl;
                exit(1);
            }
            p.debug = v;
        }
        else {
            cerr << "Unknown argument: " << arg << endl;
            exit(1);
        }
    }
    return p;
}

void validate_run_params(const RunParams& p) {
    if (p.L < 2) {
        cerr << "Invalid --L: " << p.L << " (must be >= 2 for square-lattice PBC AFM)" << endl;
        exit(2);
    }
    if (p.L % 2 != 0) {
        cerr << "Invalid --L: " << p.L
             << " (must be even with periodic boundaries to keep the AFM lattice bipartite and sign-problem-free)"
             << endl;
        exit(2);
    }
    if (!(p.J > 0.0)) {
        cerr << "Invalid --J: " << p.J << " (must be > 0 for AFM coupling in this implementation)" << endl;
        exit(2);
    }
    if (!(p.beta > 0.0) && p.beta != -1.0) {
        cerr << "Invalid --beta: " << p.beta
             << " (use a positive value, or omit --beta for the default beta sweep)" << endl;
        exit(2);
    }
    if (p.mc_iter < 2) {
        cerr << "Invalid --mc_iter: " << p.mc_iter << " (must be >= 2 for error estimates)" << endl;
        exit(2);
    }
    if (p.eq_iter < 0) {
        cerr << "Invalid --eq_iter: " << p.eq_iter << " (must be >= 0)" << endl;
        exit(2);
    }
    if (p.bin_size <= 0) {
        cerr << "Invalid --bin_size: " << p.bin_size << " (must be >= 1)" << endl;
        exit(2);
    }
    if (p.debug != 0 && p.debug != 1) {
        cerr << "Invalid --debug: " << p.debug << " (use 0 or 1)" << endl;
        exit(2);
    }
}

// ---------------------------------------------------------------------------
// Random helpers
// ---------------------------------------------------------------------------
inline double rand_double() {
    return uniform_real_distribution<double>(0.0, 1.0)(rng);
}

inline int rand_int(int n) {
    return uniform_int_distribution<int>(0, n - 1)(rng);
}

// ---------------------------------------------------------------------------
// Lattice setup
// ---------------------------------------------------------------------------
void build_bonds() {
    NB = 2 * N;
    bond_site1.resize(NB);
    bond_site2.resize(NB);

    int b = 0;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            int site = y * L + x;

            // right neighbor (PBC)
            int xr = (x + 1) % L;
            bond_site1[b] = site;
            bond_site2[b] = y * L + xr;
            b++;

            // up neighbor (PBC)
            int yu = (y + 1) % L;
            bond_site1[b] = site;
            bond_site2[b] = yu * L + x;
            b++;
        }
    }
    assert(b == NB);
}

void init_spins() {
    spin.resize(N);
    for (int i = 0; i < N; i++) {
        spin[i] = (rand_double() < 0.5) ? 1 : -1;
    }
}

// ---------------------------------------------------------------------------
// Operator string management
// ---------------------------------------------------------------------------
void maybe_expand_ops() {
    if (n_ops > (int)(0.8 * M)) {
        int target = n_ops + max(16, n_ops / 3);
        int new_M = max(max(M + 16, (int)(1.25 * M)), target);
        ops.resize(new_M, {0, 0});
        M = new_M;
    }
}

// ---------------------------------------------------------------------------
// Diagonal update
// ---------------------------------------------------------------------------
void diagonal_update(double beta) {
    // For H = J * sum_{<ij>} S_i.S_j, SSE weights scale with beta * J.
    double prob_factor = 0.5 * beta * J_coupling * NB;

    // Make a working copy of spin state to propagate through imaginary time
    vector<int> sp(spin);

    for (int p = 0; p < M; p++) {
        if (ops[p].type == 0) {
            // Identity slot — attempt insertion
            int b = rand_int(NB);
            int s1 = bond_site1[b];
            int s2 = bond_site2[b];
            // Only antiparallel spins contribute (matrix element = 1/2)
            if (sp[s1] != sp[s2]) {
                double prob = prob_factor / (double)(M - n_ops);
                if (rand_double() < prob) {
                    ops[p].type = 1;
                    ops[p].bond = b;
                    n_ops++;
                }
            }
        }
        else if (ops[p].type == 1) {
            // Diagonal op — attempt removal
            double prob = (double)(M - n_ops + 1) / prob_factor;
            if (rand_double() < prob) {
                ops[p].type = 0;
                ops[p].bond = 0;
                n_ops--;
            }
        }
        else {
            // Off-diagonal op (type 2) — propagate spin state
            int b = ops[p].bond;
            sp[bond_site1[b]] = -sp[bond_site1[b]];
            sp[bond_site2[b]] = -sp[bond_site2[b]];
        }
    }
}

// ---------------------------------------------------------------------------
// Loop update
// ---------------------------------------------------------------------------
void loop_update() {
    int n4 = 4 * M;
    vtx_link.assign(n4, -1);
    first_vtx.assign(N, -1);

    // --- Part A: Build linked vertex list ---
    // last_vtx[site] = last exit leg seen for this site
    vector<int> last_vtx(N, -1);

    for (int p = 0; p < M; p++) {
        if (ops[p].type == 0) continue; // skip identity

        int b = ops[p].bond;
        int s1 = bond_site1[b];
        int s2 = bond_site2[b];

        // Leg indices for this operator:
        // 4p+0 = site1 entrance, 4p+1 = site2 entrance
        // 4p+2 = site1 exit,     4p+3 = site2 exit
        int v0 = 4 * p;       // site1 entrance
        int v1 = 4 * p + 1;   // site2 entrance

        // Link site1: entrance ↔ previous exit
        if (last_vtx[s1] != -1) {
            vtx_link[v0] = last_vtx[s1];
            vtx_link[last_vtx[s1]] = v0;
        } else {
            first_vtx[s1] = v0;
        }
        last_vtx[s1] = 4 * p + 2; // site1 exit

        // Link site2: entrance ↔ previous exit
        if (last_vtx[s2] != -1) {
            vtx_link[v1] = last_vtx[s2];
            vtx_link[last_vtx[s2]] = v1;
        } else {
            first_vtx[s2] = v1;
        }
        last_vtx[s2] = 4 * p + 3; // site2 exit
    }

    // Close periodic imaginary-time boundary: last exit → first entrance
    for (int i = 0; i < N; i++) {
        if (first_vtx[i] != -1) {
            vtx_link[first_vtx[i]] = last_vtx[i];
            vtx_link[last_vtx[i]] = first_vtx[i];
        }
    }

    // --- Part B: Trace loops and mark flipped legs ---
    // For the Heisenberg model (only antiparallel vertices have non-zero
    // weight), the crossing rule pairs entrance-entrance and exit-exit:
    //   exit_leg = leg XOR 1  (0↔1 at entrance level, 2↔3 at exit level)
    // This is the unique pairing that preserves the antiparallel constraint
    // when a loop is flipped: same-spin pairing (0↔2, 1↔3) would create
    // parallel spin vertices with zero weight.
    vector<bool> visited(n4, false);
    vector<bool> flip(n4, false);

    for (int v0 = 0; v0 < n4; v0++) {
        if (vtx_link[v0] == -1 || visited[v0]) continue;

        bool do_flip = (rand_double() < 0.5);

        int v = v0;
        do {
            visited[v] = true;
            if (do_flip) flip[v] = true;

            int p = v / 4;
            int leg = v % 4;

            // Cross the vertex: pair entrance legs (0↔1), exit legs (2↔3)
            int exit_leg = leg ^ 1;

            int v_exit = 4 * p + exit_leg;
            visited[v_exit] = true;
            if (do_flip) flip[v_exit] = true;

            v = vtx_link[v_exit];
        } while (v != v0);
    }

    // --- Part C: Apply operator type toggles ---
    // Pairs are {0,1} (entrance) and {2,3} (exit). Toggle type if exactly
    // one pair is on a flipped loop: flip[4p] XOR flip[4p+2].
    for (int p = 0; p < M; p++) {
        if (ops[p].type == 0) continue;
        if (flip[4 * p] != flip[4 * p + 2]) {
            ops[p].type = 3 - ops[p].type;
        }
    }

    // --- Part D: Update base spins ---
    // If the tau=0 worldline segment for site i was on a flipped loop,
    // flip spin[i]. The tau=0 segment connects to first_vtx[i] (entrance).
    for (int i = 0; i < N; i++) {
        if (first_vtx[i] != -1) {
            if (flip[first_vtx[i]]) {
                spin[i] = -spin[i];
            }
        } else {
            // Free spin (no operators): flip with probability 1/2
            if (rand_double() < 0.5) {
                spin[i] = -spin[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Debug checks
// ---------------------------------------------------------------------------
void debug_check(double /* beta */) {
    // 1. Verify n_ops count
    int count = 0;
    for (int p = 0; p < M; p++) {
        if (ops[p].type != 0) count++;
    }
    assert(count == n_ops);

    // 2. All spins are +/-1
    for (int i = 0; i < N; i++) {
        assert(spin[i] == 1 || spin[i] == -1);
    }

    // 3. All operator types are 0, 1, or 2
    for (int p = 0; p < M; p++) {
        assert(ops[p].type >= 0 && ops[p].type <= 2);
    }

    // 4. All bond indices are in [0, NB)
    for (int p = 0; p < M; p++) {
        if (ops[p].type != 0) {
            assert(ops[p].bond >= 0 && ops[p].bond < NB);
        }
    }

    // 5. All non-identity ops sit on antiparallel spin pairs
    //    (check by propagating spin state)
    vector<int> sp(spin);
    for (int p = 0; p < M; p++) {
        if (ops[p].type == 1) {
            int b = ops[p].bond;
            assert(sp[bond_site1[b]] != sp[bond_site2[b]]);
        } else if (ops[p].type == 2) {
            int b = ops[p].bond;
            assert(sp[bond_site1[b]] != sp[bond_site2[b]]);
            sp[bond_site1[b]] = -sp[bond_site1[b]];
            sp[bond_site2[b]] = -sp[bond_site2[b]];
        }
    }

    // 6. Imaginary-time periodicity: propagated state == base state
    for (int i = 0; i < N; i++) {
        assert(sp[i] == spin[i]);
    }
}

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------
double measure_staggered_mag() {
    double ms = 0.0;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            int site = y * L + x;
            int sublattice_sign = ((x + y) % 2 == 0) ? 1 : -1;
            ms += sublattice_sign * spin[site];
        }
    }
    // spin[i] = 2*Sz, so ms = sum / (2*N) for magnetization per site
    return fabs(ms) / (2.0 * N);
}

double std_error_binned(const vector<double>& data, int bsize) {
    int n = (int)data.size();
    if (n < 2 * bsize) {
        // Fall back to naive SEM
        double mean = 0.0;
        for (int i = 0; i < n; i++) mean += data[i];
        mean /= n;
        double var = 0.0;
        for (int i = 0; i < n; i++) {
            double d = data[i] - mean;
            var += d * d;
        }
        var /= (n - 1);
        return sqrt(var / n);
    }

    int n_bins = n / bsize;
    vector<double> bin_avg(n_bins, 0.0);
    for (int b = 0; b < n_bins; b++) {
        for (int k = 0; k < bsize; k++) {
            bin_avg[b] += data[b * bsize + k];
        }
        bin_avg[b] /= bsize;
    }

    double mean = 0.0;
    for (int b = 0; b < n_bins; b++) mean += bin_avg[b];
    mean /= n_bins;

    double var = 0.0;
    for (int b = 0; b < n_bins; b++) {
        double d = bin_avg[b] - mean;
        var += d * d;
    }
    var /= (n_bins - 1);
    return sqrt(var / n_bins);
}

// ---------------------------------------------------------------------------
// Result row
// ---------------------------------------------------------------------------
struct ResultRow {
    double beta;
    double E_per_site;
    double dE;
    double ms;
    double dms;
    double S_pipi;
    double dS;
    double Cv;
    double dCv;
    double n_avg;
    int M_final;
    int converged;
};

// ---------------------------------------------------------------------------
// mc_run: simulate one beta value
// ---------------------------------------------------------------------------
ResultRow mc_run(double beta) {
    // Initialize spins
    init_spins();

    // Initialize operator string
    // <n> grows with beta*J; start with headroom to reduce resize frequency.
    M = max(16, (int)(1.25 * beta * J_coupling * NB) + 16);
    ops.assign(M, {0, 0});
    n_ops = 0;

    // Equilibration
    for (int t = 0; t < eq_iter; t++) {
        diagonal_update(beta);
        loop_update();
        maybe_expand_ops();
        if (debug_flag) debug_check(beta);
    }

    // Measurement
    vector<double> n_samples;
    vector<double> n2_samples;
    vector<double> ms_samples;
    vector<double> ms2_samples;
    n_samples.reserve(mc_iter);
    n2_samples.reserve(mc_iter);
    ms_samples.reserve(mc_iter);
    ms2_samples.reserve(mc_iter);

    for (int t = 0; t < mc_iter; t++) {
        diagonal_update(beta);
        loop_update();
        maybe_expand_ops();

        if (debug_flag) debug_check(beta);

        n_samples.push_back((double)n_ops);
        n2_samples.push_back((double)n_ops * (double)n_ops);

        double ms_val = measure_staggered_mag();
        ms_samples.push_back(ms_val);
        ms2_samples.push_back(ms_val * ms_val);
    }

    // Compute averages
    double n_avg = 0.0, n2_avg = 0.0, ms_avg = 0.0, ms2_avg = 0.0;
    for (int i = 0; i < mc_iter; i++) {
        n_avg += n_samples[i];
        n2_avg += n2_samples[i];
        ms_avg += ms_samples[i];
        ms2_avg += ms2_samples[i];
    }
    n_avg /= mc_iter;
    n2_avg /= mc_iter;
    ms_avg /= mc_iter;
    ms2_avg /= mc_iter;

    // Energy per site: E/(J*N) = -<n>/(beta*J*N) + NB/(4*N)
    double E_per_site = -n_avg / (beta * J_coupling * N) + (double)NB / (4.0 * N);

    // Staggered magnetization
    double ms_mean = ms_avg;

    // Structure factor: S(pi,pi) = N * <ms^2>
    double S_pipi = N * ms2_avg;

    // Specific heat: Cv = (<n^2> - <n>^2 - <n>) / N
    // From SSE: beta^2*(<H^2>-<H>^2) = <n(n-1)>-<n>^2 = <n^2>-<n>^2-<n>
    double Cv = (n2_avg - n_avg * n_avg - n_avg) / N;

    // Error bars via binned blocking
    // Energy error: dE = d<n> / (beta * J * N)
    double dE = std_error_binned(n_samples, bin_size) / (beta * J_coupling * N);

    // ms error
    double dms = std_error_binned(ms_samples, bin_size);

    // S(pi,pi) error: d(N*<ms^2>) = N * d<ms^2>
    double dS = N * std_error_binned(ms2_samples, bin_size);

    // Cv error: compute Cv per bin, take SEM of bin values
    double dCv = 0.0;
    {
        int n_bins = mc_iter / bin_size;
        if (n_bins >= 2) {
            vector<double> Cv_bins(n_bins);
            for (int b = 0; b < n_bins; b++) {
                double bn_avg = 0.0, bn2_avg = 0.0;
                for (int k = 0; k < bin_size; k++) {
                    int idx = b * bin_size + k;
                    bn_avg += n_samples[idx];
                    bn2_avg += n2_samples[idx];
                }
                bn_avg /= bin_size;
                bn2_avg /= bin_size;
                Cv_bins[b] = (bn2_avg - bn_avg * bn_avg - bn_avg) / N;
            }
            // SEM of Cv_bins
            double cv_mean = 0.0;
            for (int b = 0; b < n_bins; b++) cv_mean += Cv_bins[b];
            cv_mean /= n_bins;
            double cv_var = 0.0;
            for (int b = 0; b < n_bins; b++) {
                double d = Cv_bins[b] - cv_mean;
                cv_var += d * d;
            }
            cv_var /= (n_bins - 1);
            dCv = sqrt(cv_var / n_bins);
        }
    }

    ResultRow row;
    row.beta = beta;
    row.E_per_site = E_per_site;
    row.dE = dE;
    row.ms = ms_mean;
    row.dms = dms;
    row.S_pipi = S_pipi;
    row.dS = dS;
    row.Cv = Cv;
    row.dCv = dCv;
    row.n_avg = n_avg;
    row.M_final = M;
    row.converged = 0; // set later
    return row;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------
void write_to_file(const vector<ResultRow>& results, unsigned int seed) {
    ostringstream fname;
    fname << "mc_qheis_seed_" << seed << ".csv";
    ofstream fout(fname.str().c_str());

    if (!fout.is_open()) {
        cerr << "Unable to open file " << fname.str() << endl;
        exit(10);
    }

    fout << "beta,E_per_site,dE,ms,dms,S_pipi,dS,Cv,dCv,n_avg,M,converged" << endl;

    fout << fixed << setprecision(8);
    for (size_t i = 0; i < results.size(); i++) {
        const ResultRow& r = results[i];
        fout << r.beta << ","
             << r.E_per_site << ","
             << r.dE << ","
             << r.ms << ","
             << r.dms << ","
             << r.S_pipi << ","
             << r.dS << ","
             << r.Cv << ","
             << r.dCv << ","
             << r.n_avg << ","
             << r.M_final << ","
             << r.converged << endl;
    }

    fout.close();
    cout << "Results written to " << fname.str() << endl;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    RunParams p = parse_args(argc, argv);
    validate_run_params(p);

    L = p.L;
    N = L * L;
    J_coupling = p.J;
    mc_iter = p.mc_iter;
    eq_iter = p.eq_iter;
    bin_size = p.bin_size;
    seed_val = p.seed;
    debug_flag = p.debug;

    rng.seed(seed_val);

    cout << "SSE QMC: 2D Heisenberg AFM, L=" << L << ", N=" << N
         << ", J=" << J_coupling << ", seed=" << seed_val << endl;

    build_bonds();

    // Beta values to simulate
    vector<double> betas;
    if (p.beta > 0.0) {
        betas.push_back(p.beta);
    } else {
        double default_betas[] = {
            0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
            2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0,
            4.2, 4.4, 4.6, 4.8, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 20.0
        };
        for (double b : default_betas) betas.push_back(b);
    }

    vector<ResultRow> results;

    for (size_t i = 0; i < betas.size(); i++) {
        double beta = betas[i];
        cout << "Running beta = " << beta << " ..." << flush;

        ResultRow row = mc_run(beta);
        results.push_back(row);

        cout << " E/JN = " << fixed << setprecision(6) << row.E_per_site
             << " +/- " << row.dE
             << ", ms = " << row.ms
             << " +/- " << row.dms
             << ", Cv = " << row.Cv
             << ", <n> = " << (int)row.n_avg
             << ", M = " << row.M_final << endl;
    }

    // Convergence check: consecutive beta points within 2*combined_sigma
    for (size_t i = 1; i < results.size(); i++) {
        double dE_combined = 2.0 * sqrt(results[i].dE * results[i].dE +
                                         results[i-1].dE * results[i-1].dE);
        double diff = fabs(results[i].E_per_site - results[i-1].E_per_site);
        if (diff <= dE_combined) {
            results[i].converged = 1;
            results[i-1].converged = 1;
        }
    }

    write_to_file(results, seed_val);

    return 0;
}