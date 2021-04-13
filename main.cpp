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
        sumab += pow(a[i] - b[i],2);
        suma += pow(b[i],2);
    }
     return sqrt(sumab )/ sqrt(suma);
}

int main() {
    // int N = 290;  // input points
    // int Nout = 128;
    // double fmax = 0.4;
    // int macc = 8;
    // int r = 2;
    // int msp = 12;
    int N;  // input points
    int Nout;
    double fmax;
    int macc;
    int r;
    int msp;
    printf("*********Starting program*********\n\n");
    printf("Number of input points:");
    std::cin >> N;
    printf("Number of output points:");
    std::cin >> Nout;
    printf("Max frequency in Hz:");
    std::cin >> fmax;
    printf("Extripolation Oversampling factor [1:10]");
    std::cin >> macc;
    printf("FGG Oversampling factor [2:3]");
    std::cin >> r;
    printf("FGG gaussian approximation points [3:12]");
    std::cin >> msp;

    double *rr_out = new double[N];
    double *t = new double[N];
    double *fin = new double[Nout];
    double *Pgolden = new double[Nout];
    double *Pnfft = new double[Nout];
    double *Pextri = new double[Nout];

    double seed = ceil(rand() % 10 + 1);  // random seed between 1 100
    rrgen((double)N, seed, rr_out);

    t[0] = 0;
    for (int i = 1; i < N; i++) t[i] = t[i - 1] + rr_out[i - 1];


    double T = t[N - 1] - t[0];
    double aver = mean(rr_out, N);

    for (int i = 0; i < N; i++) rr_out[i] -= aver;

    // power-freq initializations
    for (int i = 0; i < Nout; i++) fin[i] = fmax / Nout * (i + 1);
    lombgolden(Pgolden, rr_out, t, fin, N, Nout);
    fastlomb_extri(Pextri, rr_out, t, fin, N, Nout, macc);
    fastlomb_gaussian(Pnfft, rr_out, t, fin, N, Nout, r, msp);

    // std::cout << "FGG Method: Elapsed time: " << time_nfft << " s\n";

    double n2error_extri = norm2(Pextri, Pgolden, Nout);
    double n2error_nfft = norm2(Pnfft, Pgolden, Nout);
    printf("*********Error outputs*********\n\n");

    // LOGVEC(Pnfft, Nout);
    // LOGVEC(Pextri, Nout);
    // LOGVEC(Pgolden, Nout);
    // LOGDIFF(Pgolden,Pextri, Nout);
    LOGVAR(n2error_extri);
    LOGVAR(n2error_nfft);

    delete[] rr_out;
    delete[] t;
    delete[] fin;
    delete[] Pgolden;
    delete[] Pnfft;
    delete[] Pextri;

    return 0;
}