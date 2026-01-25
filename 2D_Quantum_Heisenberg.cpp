// Sakina Saidi
// Created: Nov 15, 2025
// Last edited: Jan 15, 2026
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

using namespace std;

struct Op
{
    int type; // 0 = identity, 1 = diagonal, 2 = off-diagonal
    int bond;
};

const int L = 10;               // lattice length
const int N = L * L;            // number of sites
const int NB = 2 * N;           // number of bonds (right + up)
const double J = 1.0;           // coupling
const int mc_iter = 10000;      // measurement sweeps
const int eq_iter = 2000;       // equilibration sweeps
const int bin_size = 50;        // bin size for error bars (block averaging)
const double Tmax = 4.0;
const double Tmin = 0.4;
const double dT = 0.4;

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

void maybe_expand_ops(vector<Op> &ops, int &M, int n_ops)
{
    if (n_ops > static_cast<int>(0.8 * M))
    {
        int new_M = static_cast<int>(1.5 * M) + 10;
        ops.resize(new_M);
        for (int p = M; p < new_M; p++)
        {
            ops[p].type = 0;
            ops[p].bond = 0;
        }
        M = new_M;
    }
}

void diagonal_update(double beta, vector<int> &spins, vector<Op> &ops, int &n_ops,
                     const vector<int> &bond_i, const vector<int> &bond_j, mt19937 &rng)
{
    uniform_real_distribution<double> uni(0.0, 1.0);
    uniform_int_distribution<int> bond_pick(0, NB - 1);

    int M = static_cast<int>(ops.size());
    const double w = 0.5; // matrix element for allowed antiparallel vertices

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

double std_error(const vector<double> &v)
{
    if (v.size() <= 1)
        return 0.0;

    int n = static_cast<int>(v.size());
    int bsize = bin_size;
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

int main()
{
    mt19937 rng(static_cast<unsigned int>(time(nullptr)));

    vector<int> bond_i, bond_j;
    build_bonds(bond_i, bond_j);

    vector<double> mc_E, mc_M, mc_E_err, mc_M_err, temp_vec;

    vector<int> spins;
    random_spins(spins, rng);

    int n_ops = 0;
    int M_init = max(4 * NB, static_cast<int>(1.5 * (1.0 / Tmax) * NB) + 10);
    vector<Op> ops(M_init);
    for (int p = 0; p < M_init; p++)
    {
        ops[p].type = 0;
        ops[p].bond = 0;
    }

    for (double T = Tmax; T >= Tmin; T -= dT)
    {
        if (T <= 0.0)
            continue;

        double beta = 1.0 / T;

        int M_target = max(4 * NB, static_cast<int>(1.5 * beta * NB) + 10);
        if (M_target > static_cast<int>(ops.size()))
        {
            int old_M = static_cast<int>(ops.size());
            ops.resize(M_target);
            for (int p = old_M; p < M_target; p++)
            {
                ops[p].type = 0;
                ops[p].bond = 0;
            }
        }

        vector<double> E_config;
        vector<double> M_config;

        int M = static_cast<int>(ops.size());
        int total_sweeps = eq_iter + mc_iter;
        for (int sweep = 0; sweep < total_sweeps; sweep++)
        {
            diagonal_update(beta, spins, ops, n_ops, bond_i, bond_j, rng);
            loop_update(spins, ops, bond_i, bond_j, rng);
            maybe_expand_ops(ops, M, n_ops);

            if (sweep >= eq_iter)
            {
                double energy = -static_cast<double>(n_ops) / beta + J * NB * 0.25;
                double mz = 0.0;
                double ms = 0.0;
                for (int s = 0; s < N; s++)
                {
                    int x = s % L;
                    int y = s / L;
                    int stagger = ((x + y) % 2 == 0) ? 1 : -1;
                    mz += spins[s];
                    ms += stagger * spins[s];
                }
                double mag = fabs(ms) / (2.0 * N);

                E_config.push_back(energy);
                M_config.push_back(mag);
            }
        }

        mc_E.push_back(avg(E_config));
        mc_M.push_back(avg(M_config));
        mc_E_err.push_back(std_error(E_config));
        mc_M_err.push_back(std_error(M_config));
        temp_vec.push_back(T);
    }

    string fname = "mc_heisenberg_data.csv";
    ofstream fout(fname.c_str(), ios::out);
    if (!fout.is_open())
    {
        cerr << "Unable to open file " << fname << "." << endl;
        return 1;
    }

    fout << "E/N qmc, dE, Ms/N qmc, dMs, T/J" << endl;

    size_t data_points = mc_E.size();
    for (size_t n = 0; n < data_points; n++)
    {
        double E_per_spin = mc_E[n] / (J * N);
        double dE_per_spin = mc_E_err[n] / (J * N);
        double M_per_spin = mc_M[n];
        double dM_per_spin = mc_M_err[n];
        double T_over_J = temp_vec[n] / fabs(J);

        fout << setprecision(10)
             << E_per_spin << ","
             << dE_per_spin << ","
             << M_per_spin << ","
             << dM_per_spin << ","
             << T_over_J << endl;
    }

    fout.close();

    return 0;
}

// end of 2D_Quantum_Heisenberg.cpp