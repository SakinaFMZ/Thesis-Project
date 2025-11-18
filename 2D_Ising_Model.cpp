// Casey Berger
// Created: May 7 2014
// Last edited: Sep 17, 2025 (updated for Bates)
//
// using j = 1.0 and h = 0.0

// include files
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdio.h>
#include <sstream>
#include <time.h>

using namespace std;

//global variables
const int len = 10; //length of lattice
const int l_end = len-1; //last spot on lattice
const int num = len*len;  //total number of spins
double Tmax = 5.0;  //max temp
double Tmin = 0.0; //min temp
double dT = 0.2; //temperature iterator
const int mc_iter = 1000000; //number of monte carlo iterations
const int eq_iter = 1000; //number of iterations for equilibration
double J = 1.0; //interaction strength

//function declaration
void make_lattice(int (&Lattice)[len][len]);
double calc_E(int Lattice[len][len], int i, int j);
void flip_spin(int (&Lattice)[len][len], int i, int j, double T, double &dE);
double tot_E(int Lattice[len][len]);
void mc_sol(vector<double> &mc_E, int (&Lattice)[len][len], vector<double> &temp_vec);
double avg(vector<double> sample_vec);
void print_lattice(int Lattice[len][len]);
void exact_sol(vector<double> &exact_E);
void write_to_file(vector<double> &exact_E, vector<double> &mc_E, vector<double> &temp_vec);
void equilibrate(int (&Lattice)[len][len], double T);

int main ()
{
    srand(time(NULL)); //seed random number
    vector<double> exact_E; //stores exact solution
    vector<double> mc_E; //stores monte carlo solution
    vector<double> temp_vec; //stores temperatures
    int Lattice[len][len]; //stores lattice configuration
    
    mc_sol(mc_E, Lattice, temp_vec);
    exact_sol(exact_E);
    write_to_file(exact_E, mc_E, temp_vec);
    return 0;
}

void make_lattice(int (&Lattice)[len][len])
{
    //cout << "make_lattice" << endl;
    //generates random configuration
    for (int i = 0; i<len; i++)
    {
        for (int j = 0; j<len; j++)
        {
            double r = ((double)rand())/((double)RAND_MAX);
            if (r<0.5)
            {Lattice[i][j] = -1;}
            else
            {Lattice[i][j] = 1;}
        }
    }
}

double calc_E(int Lattice[len][len], int i, int j)
{
    //cout << "calc_E" << endl;
    //calculates energy at a particular point (i,j) on the lattice
    double Energy = 0.0;
    if (i == 0 && j == 0)
    {Energy = -1.0*J*Lattice[i][j]*(Lattice[l_end][j]+Lattice[i][l_end]+Lattice[i+1][j]+Lattice[i][j+1]);}
    else if (i == 0 && j != 0)
    {
        if (j == l_end)
        {Energy = -1.0*J*Lattice[i][j]*(Lattice[l_end][j]+Lattice[i][j-1]+Lattice[i+1][j]+Lattice[i][0]);}
        else
        {Energy = -1.0*J*Lattice[i][j]*(Lattice[l_end][j]+Lattice[i][j-1]+Lattice[i+1][j]+Lattice[i][j+1]);}
    }
    else if (i != 0 && j == 0)
    {
        if (i == l_end)
        {Energy = -1.0*J*Lattice[i][j]*(Lattice[i-1][j]+Lattice[i][j-1]+Lattice[0][j]+Lattice[i][j+1]);}
        else
        {Energy = -1.0*J*Lattice[i][j]*(Lattice[i-1][j]+Lattice[i][l_end]+Lattice[i+1][j]+Lattice[i][j+1]);}
    }
    else if (i != 0 && j != 0)
    {
        if (i==l_end && j == l_end)
        {Energy = -1.0*J*Lattice[i][j]*(Lattice[i-1][j]+Lattice[i][j-1]+Lattice[0][j]+Lattice[i][0]);}
        else if (i == l_end && j != l_end)
        {Energy = -1.0*J*Lattice[i][j]*(Lattice[i-1][j]+Lattice[i][j-1]+Lattice[0][j]+Lattice[i][j+1]);}
        else if (i != l_end && j == l_end)
        {Energy = -1.0*J*Lattice[i][j]*(Lattice[i-1][j]+Lattice[i][j-1]+Lattice[i+1][j]+Lattice[i][0]);}
        else if (i != l_end && j != l_end)
        {Energy = -1.0*J*Lattice[i][j]*(Lattice[i-1][j]+Lattice[i][j-1]+Lattice[i+1][j]+Lattice[i][j+1]);}
    }
    if (Energy == -0)
    {return 0;}
    else
    {return Energy;}
}

void flip_spin(int (&Lattice)[len][len], int i, int j, double T, double &dE)
{
    //cout << "flip_spin" << endl;
    //flips spin if correct conditions are met
    double E0 = calc_E(Lattice, i, j); //calculate initial energy at that site
    double Ef = -1.0*E0;
    dE = Ef-E0;
    //cout << "dE = " << dE << endl;
    double r = ((double)rand())/((double)RAND_MAX);
    if (Ef <= E0)
    {Lattice[i][j] = -1*Lattice[i][j];}
    else
    {
        if (r<=exp(-1.0*dE/T))
        {Lattice[i][j] = -1*Lattice[i][j];}
        else
        {dE = 0.0;}
    }
}

double tot_E(int Lattice[len][len])
{
    //cout << "tot_E" << endl;
    //calculates energy of entire configuration
    double Energy = 0.0;
    //J = -1.0*J;
    double dE = 0.0;
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            dE  = calc_E(Lattice, i, j);
            Energy = Energy+dE;
        }
    }
    return 0.5*Energy;
}

void mc_sol(vector<double> &mc_E, int (&Lattice)[len][len], vector<double> &temp_vec)
{
    //cout << "mc_sol" << endl;
    //calculates average energy (over many lattice configs) at each temperature interval
    double T=Tmax; //current temp
    //temperature loop
    make_lattice(Lattice);
    while (T>=Tmin)
    {
        double Eavg = 0.0;
        vector<double> E_config; //stores energy of each configuration within one temp iteration
        equilibrate(Lattice, T); //let the lattice equilibrate after each temperature change
        //print_lattice(Lattice,T);
        //monte carlo loop
        for (int n = 0; n < mc_iter; n++)
        {
            //print_lattice(Lattice);
            
            //pick a random spot on the lattice
            int i = len*((double)rand())/((double)RAND_MAX); //random integer
            int j = len*((double)rand())/((double)RAND_MAX); //random integer
            int count = 0;
            
            double Energy = tot_E(Lattice);
            //cout << "E0 = " << Energy << endl;
            //metropolis loop
            while (count < num)
            {
                double dE = 0.0; //declare and initialize the change in energy variable
                flip_spin(Lattice, i, j, T, dE); //flip spin if conditions are right
                
                //if (i==l_end && count%len == 0)
                if(i==l_end)
                {i = 0;}
                else
                {i++;}
                if (j == l_end)
                {j = 0;}
                else
                {j++;}
                
                count++;
            }//close metropolis loop
            Energy = tot_E(Lattice);
            E_config.push_back(Energy);
        }//close mc loop
        Eavg = avg(E_config);
        mc_E.push_back(Eavg);
        temp_vec.push_back(T);
        E_config.clear();
        T=T-dT;
    }//close temperature loop
}

double avg(vector<double> sample_vec)
{
    //cout << "avg" << endl;
    //calculates average energy of many configurations
    double avg = 0;
    int elements = sample_vec.size();
    for (int n = 0; n<elements; n++)
    {
        avg=avg+sample_vec[n];
    }
    avg = (1.0/(1.0*elements))*avg;
    //cout << avg/(1.0*num) << endl;
    return avg;
}

void print_lattice(int Lattice[len][len])
{
    //cout << "print_lattice" << endl;
    //prints lattice to screen
    for (int i = 0; i<len; i++)
    {
        for (int j = 0; j<len; j++)
        {
            if (Lattice[i][j] == -1)
            {cout << " - ";}
            else
            {cout << " + ";;}
        }
        cout << endl;
    }
    cout << endl;
}

void exact_sol(vector<double> &exact_E)
{
    //cout << "exact_sol" << endl;
    //reads in data from the file and puts it into the exact_E vector
    string fname = "exact_ising_sol.txt";
    ifstream fin;
    //open file
    fin.open(fname.c_str(),ios::in);
    
    //check validity
    if (!fin.is_open())
    {
        cerr << "Unable to open file "<< fname <<endl;
        exit(20);
    }
    
    //read in file
    string input;
    vector<string> inputvec;
    fin >> input;
    while(!fin.fail())
    {
        //cout << input << endl;
        if((input != "\n") && (input != "\t") && (input != " "))
        {
            inputvec.push_back(input);
        }
        //inputvec.push_back(input);
        fin >> input;
    }
    
    //cout << inputvec.size()<< endl;
    //close filestream
    fin.close();
    
    //cut out first, unnecessary lines of doc
    
    int j = 0;
    for (unsigned int i=0; i<inputvec.size(); i++)
    {
        if (inputvec[i]=="T/J")
        {
            j = i+1;
        }
    }
    for(unsigned int i=j;i<inputvec.size();i=i+2)
    {
        string s = inputvec[i];
        //cout << s.c_str() << "end" << endl;
        //stringstream os(s.c_str());
        double d = atof(s.c_str());
        //cout << d << endl;
        //os >> d;
        exact_E.push_back(d);
    }
    /*
     for (unsigned int n = 0; n<exact_E.size(); n++)
     {
     cout << exact_E[n] << endl;
     }*/
}

void write_to_file(vector<double> &exact_E, vector<double> &mc_E, vector<double> &temp_vec)
{
    //cout << "write_to_file" << endl;
    //output both solutions to a .txt file to open in gnuplot
    string fname = "mc_ising_data.csv";
    ofstream fout; //output stream
    fout.open(fname.c_str(),ios::out);
    
    // check if files are open
    if (!fout.is_open())
    {
        cerr << "Unable to open file " << fname <<"." << endl;
        exit(10);
    }
    fout << "E/N exact, E/N mc, T/J" << endl;
    for (unsigned int n = 0; n<exact_E.size(); n++)
    {
        //fout.setf(ios::fixed);
        fout << exact_E[n] << "," << 1.0*mc_E[n]/(1.0*J*num) << "," << temp_vec[n]/(abs(J)) << endl;
    }
    fout.close();
    exact_E.clear();
    mc_E.clear();
    temp_vec.clear();
}

void equilibrate(int (&Lattice)[len][len], double T)
{
    //cout << "equilibrate" << endl;
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