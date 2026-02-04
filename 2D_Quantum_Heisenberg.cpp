// Sakina Saidi
// Created: Nov 15, 2025
// Last edited: Feb 04, 2026
//
// Minimal SSE QMC for the 2D spin-1/2 Heisenberg model (periodic boundary conditions)
// H = J * sum_{<ij>} (S_i · S_j)  (no external field)
// SSE operator decomposition used here:
// H = -J * sum_b (H1_b - 1/4) - J * sum_b H2_b, with
// H1_b = 1/4 - Sz_i Sz_j (diagonal), H2_b = 1/2 (S+_i S-_j + S-_i S+_j) (off-diagonal).
// Allowed vertices have antiparallel spins, with matrix element w = 1/2.

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <ctime>
#include <string>
#include <cstdlib>

using namespace std;

struct Op
{
    int type; // 0 = identity, 1 = diagonal, 2 = off-diagonal
    int bond;
};

int L = 10;                      // lattice length
int N = L * L;                   // number of sites
int NB = 2 * N;                  // number of bonds (right + up)
double J = 1.0;                  // coupling
int mc_iter = 10000;             // measurement sweeps
int eq_iter = 2000;              // equilibration sweeps
int bin_size = 50;               // bin size for error bars (block averaging)
bool debug_mode = false;

static inline int idx(int x, int y)
{
    return y * L + x;
}

void build_bonds(vector<int> &bond_i, vector<int> &bond_j)
{
    bond_i.resize(NB);
    bond_j.resize(NB);
    int b = 0;
    for (int y = 0; y < L; y++)
    {
        for (int x = 0; x < L; x++)
        {
            int s = idx(x, y);
            int right = idx((x + 1) % L, y);
            int up = idx(x, (y + 1) % L);
            bond_i[b] = s;
            bond_j[b] = right;
            b++;
            bond_i[b] = s;
            bond_j[b] = up;
            b++;
        }
    }
}

void random_spins(vector<int> &spins, mt19937 &rng)
{
    uniform_int_distribution<int> coin(0, 1);
    spins.resize(N);
    for (int i = 0; i < N; i++)
    {
        spins[i] = coin(rng) ? 1 : -1; // +1 or -1 represents 2*Sz
    }
}

void maybe_expand_ops(vector<Op> &ops, int n_ops)
{
    int M = static_cast<int>(ops.size());
    if (n_ops > static_cast<int>(0.8 * M))
    {
        int new_M = static_cast<int>(1.5 * M) + 10;
        ops.resize(new_M);
        for (int p = M; p < new_M; p++)
        {
            ops[p].type = 0;
            ops[p].bond = 0;
        }
    }
}

void diagonal_update(double beta, vector<int> &spins, vector<Op> &ops, int &n_ops,
                     const vector<int> &bond_i, const vector<int> &bond_j, mt19937 &rng)
{
    uniform_real_distribution<double> uni(0.0, 1.0);
    uniform_int_distribution<int> bond_pick(0, NB - 1);

    int M = static_cast<int>(ops.size());
    // SSE operators:
    // H1 (diagonal) = 1/4 - Sz_i Sz_j, H2 (off-diagonal) = 1/2 (S+_i S-_j + S-_i S+_j).
    // Only antiparallel spins contribute with weight w = 1/2; constant shift handled in energy estimator.
    const double w = 0.5;

    for (int p = 0; p < M; p++)
    {
        if (ops[p].type == 0)
        {
            int b = bond_pick(rng);
            int i = bond_i[b];
            int j = bond_j[b];
            if (spins[i] != spins[j])
            {
                double denom = static_cast<double>(M - n_ops);
                if (denom <= 0.0)
                    continue;
                double prob = (beta * J * w * NB) / denom;
                if (prob > 1.0) prob = 1.0;
                if (uni(rng) < prob)
                {
                    ops[p].type = 1;
                    ops[p].bond = b;
                    n_ops++;
                }
            }
        }
        else if (ops[p].type == 1)
        {
            int b = ops[p].bond;
            int i = bond_i[b];
            int j = bond_j[b];
            if (spins[i] == spins[j])
            {
                ops[p].type = 0;
                n_ops--;
            }
            else
            {
                double prob = (static_cast<double>(M - n_ops + 1)) / (beta * J * w * NB);
                if (prob > 1.0) prob = 1.0;
                if (uni(rng) < prob)
                {
                    ops[p].type = 0;
                    n_ops--;
                }
            }
        }
        else if (ops[p].type == 2)
        {
            int b = ops[p].bond;
            int i = bond_i[b];
            int j = bond_j[b];
            spins[i] *= -1;
            spins[j] *= -1;
        }
    }
}

void loop_update(vector<int> &spins, vector<Op> &ops,
                 const vector<int> &bond_i, const vector<int> &bond_j,
                 mt19937 &rng)
{
    int M = static_cast<int>(ops.size());
    vector<int> link(4 * M, -1);
    vector<int> first_leg(N, -1);
    vector<int> last_leg(N, -1);
    vector<int> leg_spin(4 * M, 0);

    vector<int> spins_tmp = spins;

    for (int p = 0; p < M; p++)
    {
        if (ops[p].type == 0)
            continue;

        int b = ops[p].bond;
        int i = bond_i[b];
        int j = bond_j[b];
        int v = 4 * p;

        // link along imaginary time
        if (last_leg[i] != -1)
        {
            link[last_leg[i]] = v + 0;
            link[v + 0] = last_leg[i];
        }
        else
        {
            first_leg[i] = v + 0;
        }
        last_leg[i] = v + 2;

        if (last_leg[j] != -1)
        {
            link[last_leg[j]] = v + 1;
            link[v + 1] = last_leg[j];
        }
        else
        {
            first_leg[j] = v + 1;
        }
        last_leg[j] = v + 3;

        // record spins on legs
        leg_spin[v + 0] = spins_tmp[i];
        leg_spin[v + 1] = spins_tmp[j];
        if (ops[p].type == 2)
        {
            spins_tmp[i] *= -1;
            spins_tmp[j] *= -1;
        }
        leg_spin[v + 2] = spins_tmp[i];
        leg_spin[v + 3] = spins_tmp[j];
    }

    // close worldlines
    for (int s = 0; s < N; s++)
    {
        if (last_leg[s] != -1)
        {
            link[last_leg[s]] = first_leg[s];
            link[first_leg[s]] = last_leg[s];
        }
    }

    vector<char> visited(4 * M, 0);
    uniform_real_distribution<double> uni(0.0, 1.0);
    for (int l = 0; l < 4 * M; l++)
    {
        int v = l / 4;
        if (v >= M || ops[v].type == 0)
            continue;
        if (visited[l])
            continue;

        bool do_flip = (uni(rng) < 0.5);
        int start = l;
        int curr = l;
        do
        {
            visited[curr] = 1;
            int vert = curr / 4;
            int leg = curr % 4;

            if (do_flip)
                leg_spin[curr] *= -1;

            int exit_leg = -1;
            if (ops[vert].type == 1)
            {
                // diagonal -> straight pairing (0-2, 1-3)
                if (leg == 0) exit_leg = vert * 4 + 2;
                if (leg == 1) exit_leg = vert * 4 + 3;
                if (leg == 2) exit_leg = vert * 4 + 0;
                if (leg == 3) exit_leg = vert * 4 + 1;
            }
            else
            {
                // off-diagonal -> cross pairing (0-3, 1-2)
                if (leg == 0) exit_leg = vert * 4 + 3;
                if (leg == 1) exit_leg = vert * 4 + 2;
                if (leg == 2) exit_leg = vert * 4 + 1;
                if (leg == 3) exit_leg = vert * 4 + 0;
            }

            visited[exit_leg] = 1;
            if (do_flip)
            {
                leg_spin[exit_leg] *= -1;
                ops[vert].type = (ops[vert].type == 1) ? 2 : 1;
            }

            curr = link[exit_leg];
        } while (curr != start);
    }

    // update spins at time 0
    for (int s = 0; s < N; s++)
    {
        if (first_leg[s] != -1)
        {
            spins[s] = leg_spin[first_leg[s]];
        }
    }
}

double avg(const vector<double> &v)
{
    if (v.empty())
        return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); i++)
        sum += v[i];
    return sum / static_cast<double>(v.size());
}

double std_error(const vector<double> &v, int bsize)
{
    if (v.size() <= 1)
        return 0.0;

    int n = static_cast<int>(v.size());
    if (bsize < 1)
        bsize = 1;
    int nbins = n / bsize;
    if (nbins < 2)
    {
        double mean = avg(v);
        double var = 0.0;
        for (size_t i = 0; i < v.size(); i++)
        {
            double d = v[i] - mean;
            var += d * d;
        }
        var /= static_cast<double>(v.size() - 1);
        double stddev = sqrt(var);
        return stddev / sqrt(static_cast<double>(v.size()));
    }

    vector<double> bin_means;
    bin_means.reserve(nbins);
    for (int b = 0; b < nbins; b++)
    {
        double sum = 0.0;
        int start = b * bsize;
        for (int i = 0; i < bsize; i++)
            sum += v[start + i];
        bin_means.push_back(sum / static_cast<double>(bsize));
    }

    double mean = avg(bin_means);
    double var = 0.0;
    for (size_t i = 0; i < bin_means.size(); i++)
    {
        double d = bin_means[i] - mean;
        var += d * d;
    }
    var /= static_cast<double>(bin_means.size() - 1);
    double stddev = sqrt(var);
    return stddev / sqrt(static_cast<double>(bin_means.size()));
}

int count_non_identity_ops(const vector<Op> &ops)
{
    int count = 0;
    for (size_t i = 0; i < ops.size(); i++)
    {
        if (ops[i].type != 0)
            count++;
    }
    return count;
}

bool spins_are_pm_one(const vector<int> &spins)
{
    for (size_t i = 0; i < spins.size(); i++)
    {
        if (spins[i] != 1 && spins[i] != -1)
            return false;
    }
    return true;
}

int main(int argc, char* argv[])
{
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    double single_beta = -1.0;

    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        auto need_value = [&](const string &flag) -> string {
            if (i + 1 >= argc)
            {
                cerr << "Missing value for " << flag << "." << endl;
                exit(1);
            }
            return string(argv[++i]);
        };

        if (arg == "--L")
        {
            L = stoi(need_value(arg));
        }
        else if (arg == "--J")
        {
            J = stod(need_value(arg));
        }
        else if (arg == "--eq_iter")
        {
            eq_iter = stoi(need_value(arg));
        }
        else if (arg == "--mc_iter")
        {
            mc_iter = stoi(need_value(arg));
        }
        else if (arg == "--bin_size")
        {
            bin_size = stoi(need_value(arg));
        }
        else if (arg == "--seed")
        {
            seed = static_cast<unsigned int>(stoul(need_value(arg)));
        }
        else if (arg == "--beta")
        {
            single_beta = stod(need_value(arg));
        }
        else if (arg == "--debug")
        {
            int v = stoi(need_value(arg));
            debug_mode = (v != 0);
        }
        else
        {
            cerr << "Unknown argument: " << arg << endl;
            return 1;
        }
    }

    if (L <= 0)
    {
        cerr << "L must be positive." << endl;
        return 1;
    }
    if (J == 0.0)
    {
        cerr << "J must be nonzero." << endl;
        return 1;
    }
    if (eq_iter < 0 || mc_iter < 1)
    {
        cerr << "eq_iter must be >= 0 and mc_iter must be >= 1." << endl;
        return 1;
    }
    if (bin_size < 1)
    {
        cerr << "bin_size must be >= 1." << endl;
        return 1;
    }

    N = L * L;
    NB = 2 * N;

    mt19937 rng(seed);

    vector<int> bond_i, bond_j;
    build_bonds(bond_i, bond_j);

    vector<double> beta_list = {2, 4, 8, 12, 16, 20, 24, 32};
    if (single_beta > 0.0)
        beta_list = {single_beta};

    vector<double> mc_E, mc_E_err, mc_ms2, mc_ms2_err, mc_ms_abs, mc_ms_abs_err;
    vector<double> mc_Spi, beta_vec, T_vec, nops_avg;
    vector<int> converged;

    for (size_t bidx = 0; bidx < beta_list.size(); bidx++)
    {
        double beta = beta_list[bidx];
        if (beta <= 0.0)
            continue;

        vector<int> spins;
        random_spins(spins, rng);

        int n_ops = 0;
        int M_target = max(4 * NB, static_cast<int>(1.5 * beta * NB) + 10);
        vector<Op> ops(M_target);
        for (int p = 0; p < M_target; p++)
        {
            ops[p].type = 0;
            ops[p].bond = 0;
        }

        vector<double> E_config;
        vector<double> ms2_config;
        vector<double> ms_abs_config;
        vector<double> nops_config;

        int total_sweeps = eq_iter + mc_iter;
        for (int sweep = 0; sweep < total_sweeps; sweep++)
        {
            diagonal_update(beta, spins, ops, n_ops, bond_i, bond_j, rng);
            loop_update(spins, ops, bond_i, bond_j, rng);
            maybe_expand_ops(ops, n_ops);

            if (n_ops < 0 || n_ops > static_cast<int>(ops.size()))
            {
                cerr << "n_ops out of bounds: " << n_ops << " with M=" << ops.size() << endl;
                return 1;
            }

            if (debug_mode)
            {
                int counted = count_non_identity_ops(ops);
                if (counted != n_ops)
                {
                    cerr << "Debug: n_ops mismatch. n_ops=" << n_ops
                         << " counted=" << counted << endl;
                    return 1;
                }
                if (!spins_are_pm_one(spins))
                {
                    cerr << "Debug: spins not +/-1." << endl;
                    return 1;
                }
                for (size_t p = 0; p < ops.size(); p++)
                {
                    if (ops[p].type < 0 || ops[p].type > 2)
                    {
                        cerr << "Debug: invalid op type at p=" << p << endl;
                        return 1;
                    }
                }
            }

            if (sweep >= eq_iter)
            {
                double energy = -static_cast<double>(n_ops) / beta + J * NB * 0.25;
                double ms_sum = 0.0;
                for (int s = 0; s < N; s++)
                {
                    int x = s % L;
                    int y = s / L;
                    int stagger = ((x + y) % 2 == 0) ? 1 : -1;
                    ms_sum += stagger * spins[s];
                }
                double ms = (0.5 * ms_sum) / static_cast<double>(N);
                double ms2 = ms * ms;
                double ms_abs = fabs(ms);
                double s_pi_pi = (ms_sum * ms_sum) / (4.0 * static_cast<double>(N));

                E_config.push_back(energy);
                ms2_config.push_back(ms2);
                ms_abs_config.push_back(ms_abs);
                nops_config.push_back(static_cast<double>(n_ops));
            }
        }

        int n_measurements = static_cast<int>(E_config.size());
        int nbins = (bin_size > 0) ? (n_measurements / bin_size) : 0;
        cout << "beta=" << beta << " measurements=" << n_measurements
             << " nbins=" << nbins << endl;
        if (nbins < 10)
        {
            cout << "Warning: nbins < 10; error bars may be unreliable." << endl;
        }

        double E_mean = avg(E_config);
        double E_err = std_error(E_config, bin_size);
        double ms2_mean = avg(ms2_config);
        double ms2_err = std_error(ms2_config, bin_size);
        double ms_abs_mean = avg(ms_abs_config);
        double ms_abs_err = std_error(ms_abs_config, bin_size);
        double nops_mean = avg(nops_config);
        double s_pi_pi_mean = ms2_mean * static_cast<double>(N);

        mc_E.push_back(E_mean);
        mc_E_err.push_back(E_err);
        mc_ms2.push_back(ms2_mean);
        mc_ms2_err.push_back(ms2_err);
        mc_ms_abs.push_back(ms_abs_mean);
        mc_ms_abs_err.push_back(ms_abs_err);
        mc_Spi.push_back(s_pi_pi_mean);
        nops_avg.push_back(nops_mean);
        beta_vec.push_back(beta);
        T_vec.push_back(1.0 / beta);

        int conv_flag = 0;
        if (!mc_E.empty() && mc_E.size() >= 2)
        {
            size_t k = mc_E.size() - 1;
            double diff = fabs(mc_E[k] - mc_E[k - 1]);
            double thresh = 2.0 * sqrt(mc_E_err[k] * mc_E_err[k] + mc_E_err[k - 1] * mc_E_err[k - 1]);
            if (diff < thresh)
                conv_flag = 1;
        }
        converged.push_back(conv_flag);
    }

    string fname = "mc_qheis_seed_" + to_string(seed) + ".csv";
    ofstream fout(fname.c_str(), ios::out);
    if (!fout.is_open())
    {
        cerr << "Unable to open file " << fname << "." << endl;
        return 1;
    }

    fout << "L,N,beta,T_over_J,seed,eq_iter,mc_iter,bin_size,"
         << "E_per_site,dE_per_site,ms2,dms2,ms_abs,dms_abs,S_pi_pi,n_ops_avg,converged" << endl;

    size_t data_points = mc_E.size();
    for (size_t n = 0; n < data_points; n++)
    {
        double E_per_spin = mc_E[n] / (J * N);
        double dE_per_spin = mc_E_err[n] / (J * N);
        double T_over_J = T_vec[n];

        fout << setprecision(10)
             << L << ","
             << N << ","
             << beta_vec[n] << ","
             << T_over_J << ","
             << seed << ","
             << eq_iter << ","
             << mc_iter << ","
             << bin_size << ","
             << E_per_spin << ","
             << dE_per_spin << ","
             << mc_ms2[n] << ","
             << mc_ms2_err[n] << ","
             << mc_ms_abs[n] << ","
             << mc_ms_abs_err[n] << ","
             << mc_Spi[n] << ","
             << nops_avg[n] << ","
             << converged[n] << endl;
    }

    fout.close();

    return 0;
}

// end of 2D_Quantum_Heisenberg.cpp
