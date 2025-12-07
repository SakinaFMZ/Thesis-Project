// Sakina Saidi
// Created: Nov 15, 2025
// Last edited: Dec 7, 2025
//
// using J = 1.0 and h = 0.0

// include files
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <string>
#include <stdio.h>
#include <sstream>
#include <time.h>

using namespace std;

// Heisenberg spin type & helpers
struct Spin {
    double x;
    double y;
    double z;
};

const double PI = 3.14159265358979323846;

// draw a random unit vector on the sphere
Spin random_spin() {
    double u = ((double)rand()) / ((double)RAND_MAX); // in [0,1]
    double v = ((double)rand()) / ((double)RAND_MAX); // in [0,1]

    double z = 2.0 * u - 1.0;              // cos(theta) in [-1,1]
    double phi = 2.0 * PI * v;             // azimuthal angle
    double r_xy = sqrt(1.0 - z * z);       // sin(theta)

    Spin s;
    s.x = r_xy * cos(phi);
    s.y = r_xy * sin(phi);
    s.z = z;
    return s;
}

double dot(const Spin &a, const Spin &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// global variables
const int len = 10;          // length of lattice
const int l_end = len - 1;   // last spot on lattice
const int num = len * len;   // total number of spins
double Tmax = 5.0;           // max temp
double Tmin = 0.0;           // min temp
double dT = 0.2;             // temperature iterator
const int mc_iter = 1000000; // number of monte carlo iterations
const int eq_iter = 1000;    // number of iterations for equilibration
double J = 1.0;              // interaction strength

// function declarations
void make_lattice(Spin (&Lattice)[len][len]);
double calc_E(Spin Lattice[len][len], int i, int j);
void flip_spin(Spin (&Lattice)[len][len], int i, int j, double T, double &dE);
double tot_E(Spin Lattice[len][len]);
void mc_sol(vector<double> &mc_E, vector<double> &mc_M,
            Spin (&Lattice)[len][len], vector<double> &temp_vec);
double avg(const vector<double> &sample_vec);
double magnetization(Spin Lattice[len][len]);
void print_lattice(Spin Lattice[len][len]);
void write_to_file(vector<double> &mc_E, vector<double> &mc_M,
                   vector<double> &temp_vec);
void equilibrate(Spin (&Lattice)[len][len], double T);

int main()
{
    srand(time(NULL)); // seed random number

    vector<double> mc_E;      // Monte Carlo energies
    vector<double> mc_M;      // Monte Carlo magnetizations
    vector<double> temp_vec;  // temperatures
    Spin Lattice[len][len];   // lattice configuration

    mc_sol(mc_E, mc_M, Lattice, temp_vec);
    write_to_file(mc_E, mc_M, temp_vec);

    return 0;
}

// initialize random Heisenberg spins
void make_lattice(Spin (&Lattice)[len][len])
{
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            Lattice[i][j] = random_spin();
        }
    }
}

// local energy at site (i,j) for Heisenberg model
double calc_E(Spin Lattice[len][len], int i, int j)
{
    double Energy = 0.0;

    // periodic neighbor indices
    int ip = (i + 1) % len;
    int im = (i - 1 + len) % len;
    int jp = (j + 1) % len;
    int jm = (j - 1 + len) % len;

    Spin s = Lattice[i][j];

    // sum of nearest neighbors
    Spin sum;
    sum.x = 0.0; sum.y = 0.0; sum.z = 0.0;

    // up, down, left, right
    sum.x += Lattice[im][j].x + Lattice[ip][j].x + Lattice[i][jm].x + Lattice[i][jp].x;
    sum.y += Lattice[im][j].y + Lattice[ip][j].y + Lattice[i][jm].y + Lattice[i][jp].y;
    sum.z += Lattice[im][j].z + Lattice[ip][j].z + Lattice[i][jm].z + Lattice[i][jp].z;

    Energy = -1.0 * J * dot(s, sum);

    if (Energy == -0.0)
        return 0.0;
    else
        return Energy;
}

// Metropolis: propose a new random orientation at (i,j)
void flip_spin(Spin (&Lattice)[len][len], int i, int j, double T, double &dE)
{
    double E0 = calc_E(Lattice, i, j); // initial local energy

    Spin old_spin = Lattice[i][j];
    Spin trial_spin = random_spin();

    // assign trial spin temporarily
    Lattice[i][j] = trial_spin;
    double Ef = calc_E(Lattice, i, j);

    dE = Ef - E0;

    double r = ((double)rand()) / ((double)RAND_MAX);
    if (Ef <= E0)
    {
        // accept trial (already assigned)
    }
    else
    {
        if (T > 0.0 && r <= exp(-1.0 * dE / T))
        {
            // accept trial (already assigned)
        }
        else
        {
            // reject trial, restore old spin
            Lattice[i][j] = old_spin;
            dE = 0.0;
        }
    }
}

// total energy of entire lattice
double tot_E(Spin Lattice[len][len])
{
    double Energy = 0.0;
    double dE = 0.0;
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            dE = calc_E(Lattice, i, j);
            Energy = Energy + dE;
        }
    }
    // each bond counted twice
    return 0.5 * Energy;
}

// Monte Carlo solution vs temperature
void mc_sol(vector<double> &mc_E, vector<double> &mc_M,
            Spin (&Lattice)[len][len], vector<double> &temp_vec)
{
    double T = Tmax; // current temp

    // start from a random configuration
    make_lattice(Lattice);

    // temperature loop
    while (T >= Tmin)
    {
        double Eavg = 0.0;
        double Mavg = 0.0;
        vector<double> E_config; // energies at this T
        vector<double> M_config; // magnetizations at this T

        equilibrate(Lattice, T); // let the lattice equilibrate

        // Monte Carlo loop
        for (int n = 0; n < mc_iter; n++)
        {
            // Metropolis sweep across every lattice site
            for (int i = 0; i < len; i++)
            {
                for (int j = 0; j < len; j++)
                {
                    double dE = 0.0;
                    flip_spin(Lattice, i, j, T, dE);
                }
            }

            double Energy = tot_E(Lattice);
            E_config.push_back(Energy);

            double M = magnetization(Lattice);
            M_config.push_back(M);
        }

        Eavg = avg(E_config);
        Mavg = avg(M_config);

        mc_E.push_back(Eavg);
        mc_M.push_back(Mavg);
        temp_vec.push_back(T);

        E_config.clear();
        M_config.clear();

        T = T - dT;
    }
}

double avg(const vector<double> &sample_vec)
{
    if (sample_vec.empty())
        return 0.0;

    double sum = 0.0;
    for (unsigned int n = 0; n < sample_vec.size(); n++)
    {
        sum += sample_vec[n];
    }
    return sum / static_cast<double>(sample_vec.size());
}

// compute total magnetization magnitude per spin
double magnetization(Spin Lattice[len][len])
{
    double mx = 0.0, my = 0.0, mz = 0.0;

    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            mx += Lattice[i][j].x;
            my += Lattice[i][j].y;
            mz += Lattice[i][j].z;
        }
    }

    double M = sqrt(mx * mx + my * my + mz * mz);
    return M / num;   // return magnetization per spin
}

// simple visualization: sign of Sz
void print_lattice(Spin Lattice[len][len])
{
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            if (Lattice[i][j].z < 0.0)
                cout << " - ";
            else
                cout << " + ";
        }
        cout << endl;
    }
    cout << endl;
}

void write_to_file(vector<double> &mc_E, vector<double> &mc_M,
                   vector<double> &temp_vec)
{
    string fname = "mc_heisenberg_data.csv";
    ofstream fout;
    fout.open(fname.c_str(), ios::out);

    if (!fout.is_open())
    {
        cerr << "Unable to open file " << fname << "." << endl;
        exit(10);
    }

    fout << "E/N mc, M/N mc, T/J" << endl;

    unsigned long data_points = mc_E.size();
    if (mc_M.size() < data_points) data_points = mc_M.size();
    if (temp_vec.size() < data_points) data_points = temp_vec.size();

    for (unsigned long n = 0; n < data_points; n++)
    {
        double E_per_spin = mc_E[n] / (J * num);
        double M_per_spin = mc_M[n];
        double T_over_J   = temp_vec[n] / fabs(J);

        fout << E_per_spin << "," << M_per_spin << "," << T_over_J << endl;
    }

    fout.close();
    mc_E.clear();
    mc_M.clear();
    temp_vec.clear();
}

void equilibrate(Spin (&Lattice)[len][len], double T)
{
    double dE = 0.0;
    int count = 0;
    while (count < eq_iter)
    {
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < len; j++)
            {
                flip_spin(Lattice, i, j, T, dE);
                count++;
            }
        }
    }
}

// end of 2D_Heisenberg_Model.cpp