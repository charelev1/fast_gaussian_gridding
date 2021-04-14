#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "lombmethods.h"
#include "myfft.h"

extern "C" {
#include "rrgenV3.h"
}

#define LOGVEC(x, N)                                           \
    for (int i = 0; i < N; i++) {                              \
        std::cout.precision(17);                               \
        std::cout << #x << "[" << i << "] = " << x[i] << "\n"; \
    }                                                          \
    std::cout << "\n"
#define LOGVECDIFF(x, y, N)                                     \
    for (int i = 0; i < N; i++) {                               \
        std::cout << "diff"                                     \
                  << "[" << i << "] = " << x[i] - y[i] << "\n"; \
    }                                                           \
    std::cout << "\n"
#define LOGDIFF(x, y, N)                                       \
    for (int i = 0; i < N; i++) {                              \
        std::cout << #x << "[" << i << "]- " << #y << "[" << i \
                  << "] = " << x[i] - y[i] << "\n";            \
    }                                                          \
    std::cout << "\n"

#define SWAP(x, y) \
    auto tmp = x;  \
    x = y;         \
    y = tmp

#define LOGVAR(x) std::cout << #x << " = " << x << "\n"
#define LOG(x) std::cout << x << "\n"
#define MOD(y, x) (x + y) % x
#define MODF(x, y) std::fmod((double)x, (double)y)
#define MIN(x, y) (x < y) ? x : y
#define MAX(x, y) (x > y) ? x : y
#define PI 3.1415926535897932384626433832795028841971693993751058

constexpr double mean(double *x, int N) {
    double sum = 0;
    for (int i = 0; i < N; i++) {
        sum += x[i];
    }
    return sum / N;
}

long double norm2(double *a, double *b, int size) {
    long double sumab = 0;
    long double suma = 0;

    for (int i = 0; i < size; i++) {
        sumab += pow(a[i] - b[i], 2);
        suma += pow(b[i], 2);
    }
    return sqrt(sumab) / sqrt(suma);
}




void comparemethods(unsigned int N, unsigned int Nout, double fmax,
                    unsigned int MACC, unsigned int R, unsigned int Msp,
                    _TIME &golden, _TIME &extrip, _TIME &gaussi,
                    double &n2error_extri, double &n2error_nfft) {


    double *rr_out = new double[N];
    double *t = new double[N];
    double *fin = new double[Nout];
    double *Pgolden = new double[Nout];
    double *Pnfft = new double[Nout];
    double *Pextri = new double[Nout];

    double seed = ceil(rand() % 10 + 1);  // random seed between 1 100
    // double seed = 10; // random seed between 1 100
    rrgen((double)N, seed, rr_out);

    t[0] = 0;
    for (int i = 1; i < N; i++) t[i] = t[i - 1] + rr_out[i - 1];

    double T = t[N - 1] - t[0];
    // LOGVAR(T);
    double aver = mean(rr_out, N);

    for (int i = 0; i < N; i++) rr_out[i] -= aver;

    // power-freq initializations

    for (int i = 0; i < Nout; i++) fin[i] = fmax / Nout * (i + 1);

    lombgolden(Pgolden, rr_out, t, fin, N, Nout, golden);
    // fastlomb_extri(Pextri, rr_out, t, fin, N, Nout, MACC, extrip);
    fastlomb_gaussian(Pnfft, rr_out, t, fin, N, Nout, R, Msp, gaussi);

    // std::cout << "FGG Method: Elapsed time: " << time_nfft << " s\n";

    n2error_extri = norm2(Pextri, Pgolden, Nout);
    n2error_nfft = norm2(Pnfft, Pgolden, Nout);

    // LOGVAR(n2error_nfft);
    // LOGVAR(n2error_extri);
    // LOGVAR(N);
    // LOGVAR(Nout);

    delete[] rr_out;
    delete[] t;
    delete[] fin;
    delete[] Pgolden;
    delete[] Pnfft;
    delete[] Pextri;
}

int main() {
    const int loops_size = 3;
    const int loops_aprx = 3;
    const int loops_iter = 10;

    _TIME direct[loops_size][loops_aprx][loops_iter];
    _TIME extrip[loops_size][loops_aprx][loops_iter];
    _TIME gaussi[loops_size][loops_aprx][loops_iter];

    double n2error_extri[loops_size][loops_aprx][loops_iter];
    double n2error_nfft[loops_size][loops_aprx][loops_iter];

    // unsigned int Nin[loops_size] = {64, 128, 256, 512, 1024, 2048, 4096};
    unsigned int Nin[loops_size] = {64, 128, 256};
    // unsigned int Nout[loops_size] = {64, 128, 256, 512, 1024, 2048, 4096};
    unsigned int Nout[loops_size] = {64, 128, 256};
    unsigned int MACC[loops_aprx] = {2, 4, 8};
    unsigned int Msp[loops_aprx] = {3, 6, 12};
    double fmax = 0.4;
    unsigned int R = 2;

    for (int i = 0; i < loops_size; i++) {
        for (int j = 0; j < loops_aprx; j++) {
            for (int k = 0; k < loops_iter; k++) {
                comparemethods(Nin[i], Nout[k], fmax, MACC[j], R, Msp[j],
                               direct[i][j][k], extrip[i][j][k], gaussi[i][j][k],
                               n2error_extri[i][j][k], n2error_nfft[i][j][k]);

                direct[i][j][k].maketotal();
                extrip[i][j][k].maketotal();
                gaussi[i][j][k].maketotal();
            }
        }
    }

    std::ofstream myFILE;
    myFILE.open("timing.csv");
    const char *Header =
        "Nin,Nout,MACC,Msp,Gold_time,Extrip_Total,Extrip_Grid,Extrip_fft,FGG_"
        "Total,FGG_Grid,FGG_fft";
    myFILE << Header << std::endl;

    for (int j = 0; j < loops_aprx; j++) {
        for (int i = 0; i < loops_size; i++) {
            double Gold = 0;
            double Extrip_Total = 0;
            double Extrip_Grid = 0;
            double Extrip_fft = 0;
            double FGG_Total = 0;
            double FGG_Grid = 0;
            double FGG_fft = 0;

            for (int k = 0; k < loops_iter; k++) {
                Gold += direct[i][j][k].total / loops_iter;
                Extrip_Total += extrip[i][j][k].total / loops_iter;
                Extrip_Grid += extrip[i][j][k].grid / loops_iter;
                Extrip_fft += extrip[i][j][k].fft / loops_iter;
                FGG_Total = gaussi[i][j][k].total / loops_iter;
                FGG_Grid = gaussi[i][j][k].grid / loops_iter;
                FGG_fft = gaussi[i][j][k].fft / loops_iter;
            }

            myFILE << Nin[i] << ',' << Nout[i] << ',' << MACC[j] << ","
                   << Msp[j] << ",";
            myFILE << Gold << "," << Extrip_Total<< ","
                   << Extrip_Grid << ",";
            myFILE << Extrip_fft << "," << FGG_Total << ","
                   << FGG_Grid << ",";
            myFILE << FGG_fft <<  std::endl;
        }
        myFILE << "\n";
    }
    myFILE.close();
    return 0;
}

// int main() {
//     int N = 290;  // input points
//     int Nout = 128;
//     double fmax = 0.4;
//     int macc = 8;
//     int r = 2;
//     int msp = 12;
//     // int N;  // input points
//     // int Nout;
//     // double fmax;
//     // int macc;
//     // int r;
//     // int msp;
//     // printf("*********Starting program*********\n\n");
//     // printf("Number of input points:");
//     // std::cin >> N;
//     // printf("Number of output points:");
//     // std::cin >> Nout;
//     // printf("Max frequency in Hz:");
//     // std::cin >> fmax;
//     // printf("Extripolation Oversampling factor [1:10]");
//     // std::cin >> macc;
//     // printf("FGG Oversampling factor [2:3]");
//     // std::cin >> r;
//     // printf("FGG gaussian approximation points [3:12]");
//     // std::cin >> msp;

//     double *rr_out = new double[N];
//     double *t = new double[N];
//     double *fin = new double[Nout];
//     double *Pgolden = new double[Nout];
//     double *Pnfft = new double[Nout];
//     double *Pextri = new double[Nout];

//     double seed = ceil(rand() % 10 + 1);  // random seed between 1 100
//     rrgen((double)N, seed, rr_out);

//     t[0] = 0;
//     for (int i = 1; i < N; i++) t[i] = t[i - 1] + rr_out[i - 1];

//     double T = t[N - 1] - t[0];
//     double aver = mean(rr_out, N);

//     for (int i = 0; i < N; i++) rr_out[i] -= aver;

//     // power-freq initializations
//     for (int i = 0; i < Nout; i++) fin[i] = fmax / Nout * (i + 1);

//     _TIME golden;
//     _TIME extrip;
//     _TIME gaussi;

//     lombgolden(Pgolden, rr_out, t, fin, N, Nout, golden);
//     fastlomb_extri(Pextri, rr_out, t, fin, N, Nout, macc, extrip);
//     fastlomb_gaussian(Pnfft, rr_out, t, fin, N, Nout, r, msp, gaussi);

//     // std::cout << "FGG Method: Elapsed time: " << time_nfft << " s\n";
//     // LOGVAR(extrip.grid);
//     // LOGVAR(extrip.fft);
//     // LOGVAR(extrip.lomb);
//     extrip.maketotal();
//     golden.maketotal();
//     gaussi.maketotal();

//     LOGVAR(extrip.total);
//     LOGVAR(golden.total);
//     LOGVAR(gaussi.total);
//     // LOGVAR(golden.lomb);
//     // LOGVAR(gaussi.grid);
//     // LOGVAR(gaussi.fft);
//     // LOGVAR(gaussi.lomb);

//     double n2error_extri = norm2(Pextri, Pgolden, Nout);
//     double n2error_nfft = norm2(Pnfft, Pgolden, Nout);
//     printf("*********Error outputs*********\n\n");

//     // LOGVEC(Pnfft, Nout);
//     // LOGVEC(Pextri, Nout);
//     // LOGVEC(Pgolden, Nout);
//     // LOGDIFF(Pgolden,Pextri, Nout);
//     LOGVAR(n2error_extri);
//     LOGVAR(n2error_nfft);

//     delete[] rr_out;
//     delete[] t;
//     delete[] fin;
//     delete[] Pgolden;
//     delete[] Pnfft;
//     delete[] Pextri;

//     return 0;
// }
