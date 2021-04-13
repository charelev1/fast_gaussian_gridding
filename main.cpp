
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
extern "C" {
#include "rrgenV3.h"
}
#include "myfft.h"

#define LOGVEC(x, N)                                           \
    for (int i = 0; i < N; i++) {                              \
        std::cout << #x << "[" << i << "] = " << x[i] << "\n"; \
    }                                                          \
    std::cout << "\n"
#define LOGVECDIFF(x, y, N)                                     \
    for (int i = 0; i < N; i++) {                               \
        std::cout << "diff"                                     \
                  << "[" << i << "] = " << x[i] - y[i] << "\n"; \
    }                                                           \
    std::cout << "\n"
#define LOGDIFF(x, y, N)                                                                         \
    for (int i = 0; i < N; i++) {                                                                \
        std::cout << #x << "[" << i << "]- " << #y << "[" << i << "] = " << x[i] - y[i] << "\n"; \
    }                                                                                            \
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

#define PI 3.141592653589793238462643383279502884197169399375105820974944
#define _CMPXD std::complex<double>

using namespace std::literals::complex_literals;

void fftc(_CMPXD *xin, _CMPXD *xout, int N);
void fftd(double *xin, _CMPXD *xout, int N);
void ndftdir(double *x, double *f, _CMPXD *Fk, int N, int M, int Mr);
void nufft(double *x, double *f, _CMPXD *Fk, int N, int M, int R, int Msp);
void spread(double y, double *yy, double x, int n, int m);
void spread_iter(double y, _MYCMPX *yy, double x, int n, int m);

void nonuni2uni1(double *f, double *x, double *fout, int N, int Mr, int Msp, double tau, double *E3, double df) {

    for (int j = 0; j < N; j++) {

        int m1 = ceil(x[j] * 2 * PI * df / (2 * PI / Mr));
        double diff = x[j] * 2 * PI * df - 2 * PI * m1 / Mr;
        double E1 = exp(-diff * diff / (4 * tau));
        double E2 = exp(PI * diff / (Mr * tau));

        for (int l = -Msp + 1; l <= Msp; l++) {

            int idx = MOD(m1 + l, Mr);
            fout[idx] = fout[idx] + f[j] * E1 * pow(E2, l) * E3[abs(l)];
        }
    }
}

void nonuni2uni2(double *x, double *fout, int N, int Mr, int Msp, double tau, double *E3, double df) {

    for (int j = 0; j < N; j++) {

        int m1 = ceil(x[j] * 2 * PI * df / (2 * PI / Mr));
        double diff = x[j] * 2 * PI * df - 2 * PI * m1 / Mr;
        double E1 = exp(-diff * diff / (4 * tau));
        double E2 = exp(PI * diff / (Mr * tau));

        for (int l = -Msp + 1; l <= Msp; l++) {

            int idx = MOD(m1 + l, Mr);
            fout[idx] = fout[idx] + E1 * pow(E2, l) * E3[abs(l)];
        }
    }
}

void nonuni2uni1_iter(double *f, double *x, _MYCMPX *fout, int N, int Mr, int Msp, double tau, double *E3, double df) {

    for (int j = 0; j < N; j++) {

        int m1 = ceil(x[j] * 2 * PI * df / (2 * PI / Mr));
        double diff = x[j] * 2 * PI * df - 2 * PI * m1 / Mr;
        double E1 = exp(-diff * diff / (4 * tau));
        double E2 = exp(PI * diff / (Mr * tau));

        for (int l = -Msp + 1; l <= Msp; l++) {

            int idx = MOD(m1 + l, Mr);
            fout[idx] = fout[idx] + f[j] * E1 * pow(E2, l) * E3[abs(l)];
        }
    }
}

void nonuni2uni2_iter(double *x, _MYCMPX *fout, int N, int Mr, int Msp, double tau, double *E3, double df) {

    for (int j = 0; j < N; j++) {

        int m1 = ceil(x[j] * 2 * PI * df / (2 * PI / Mr));
        double diff = x[j] * 2 * PI * df - 2 * PI * m1 / Mr;
        double E1 = exp(-diff * diff / (4 * tau));
        double E2 = exp(PI * diff / (Mr * tau));

        for (int l = -Msp + 1; l <= Msp; l++) {

            int idx = MOD(m1 + l, Mr);
            fout[idx] = fout[idx] + E1 * pow(E2, l) * E3[abs(l)];
        }
    }
}

void fftd(double *xin, _CMPXD *xout, int N) {

    if (N == 1) {

        xout[0] = xin[0];
    } else {

        _CMPXD *x1 = new _CMPXD[N / 2];
        _CMPXD *x2 = new _CMPXD[N / 2];

        for (int i = 0; i < N / 2; i++) {
            x1[i] = xin[2 * i];
            x2[i] = xin[2 * i + 1];
        }
        fftc(x1, x1, N / 2);
        fftc(x2, x2, N / 2);

        for (int j = 0; j < N / 2; j++) {
            _CMPXD phasor = std::exp(1i * (-2 * PI * j / N));
            xout[j] = x1[j] + x2[j] * phasor;
            xout[j + N / 2] = x1[j] - x2[j] * phasor;
        }

        delete[] x1;
        delete[] x2;
    }
}

constexpr double mean(double *x, int N) {
    double sum = 0;
    for (int i = 0; i < N; i++) {
        sum += x[i];
    }
    return sum / N;
}

unsigned int reversebitsLookUp(unsigned int val_in, unsigned int m) {
    static const unsigned char BitReverseTable256[] =
        {
            0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
            0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
            0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
            0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
            0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
            0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
            0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
            0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
            0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
            0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
            0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
            0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
            0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
            0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
            0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
            0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF};

    unsigned int var = (BitReverseTable256[val_in & 0xff] << 24) |
                       (BitReverseTable256[(val_in >> 8) & 0xff] << 16) |
                       (BitReverseTable256[(val_in >> 16) & 0xff] << 8) |
                       (BitReverseTable256[(val_in >> 24) & 0xff]);
    return var >> (32 - m);
}

void fft_iter(_MYCMPX *data, unsigned int N) {

    unsigned int m = (int)log2(N);
    unsigned int j;
    for (unsigned int i = 1; i < N - 1; i++) {
        j = reversebitsLookUp(i, m);
        if (i < j) {
            SWAP(data[j], data[i]);
        }
    }
    _MYCMPX factor;
    unsigned int Ninter = 1;

    for (unsigned int i = 0; i < m; i++) {
        double step = -6.283185307179586 / (Ninter * 2);
        double a = 0;

        for (unsigned int j = 0; j < Ninter; j++) { //inter loop
            factor.real = cos(a);
            factor.imag = sin(a);
            a += step;

            for (unsigned int k = 0; k < N; k += 2 * Ninter) {
                _MYCMPX tmp;
                tmp = data[k + j + Ninter] * factor;
                data[k + j + Ninter] = data[k + j] - tmp;
                data[k + j] = data[k + j] + tmp;
            }
        }
        Ninter *= 2;
    }
}

void fftc(_CMPXD *xin, _CMPXD *xout, int N) {

    if (N == 1) {

        xout[0] = xin[0];
    } else {

        _CMPXD *x1 = new _CMPXD[N / 2];
        _CMPXD *x2 = new _CMPXD[N / 2];

        for (int i = 0; i < N / 2; i++) {
            x1[i] = xin[2 * i];
            x2[i] = xin[2 * i + 1];
        }

        fftc(x1, x1, N / 2);
        fftc(x2, x2, N / 2);

        for (int j = 0; j < N / 2; j++) {
            _CMPXD phasor = std::exp(1i * (-2 * PI * j / N));
            xout[j] = x1[j] + x2[j] * phasor;
            xout[j + N / 2] = x1[j] - x2[j] * phasor;
        }

        delete[] x1;
        delete[] x2;
    }
}

void ndftdir(double *x, double *f, _CMPXD *Fk, int N, int M, int Mr) {

    for (int k = -M / 2; k <= (M - 1) / 2; k++) {
        _CMPXD sum = 0;

        for (int j = 0; j < N; j++) {
            sum = sum + f[j] * std::exp(1i * (-k * x[j]));
        }
        Fk[k + M / 2] = sum / (double)N;
    }
}

void nufft(double *x, double *f, _CMPXD *Fk, int N, int M, int R, int Msp, double df, bool flag) {

    const int Mr = R * M;
    const double tau = PI * (Msp) / M / M / R / (R - .5);

    double *E3 = new double[Msp + 1];
    _CMPXD *Fktau1 = new _CMPXD[Mr];
    double *finter = new double[Mr]();

    for (int l = 0; l <= Msp; l++)
        E3[l] = exp(-pow(PI * l / Mr, 2) / tau);

    if (!flag) {

        nonuni2uni1(f, x, finter, N, Mr, Msp, tau, E3, df);
    } else {
        nonuni2uni2(x, finter, N, Mr, Msp, tau, E3, df);
    }

    fftd(finter, Fktau1, Mr);

    for (int k = -M / 2; k <= (M - 1) / 2; k++) {

        int idx1 = MOD(k, Mr);
        int idx2 = MOD(k, M);
        Fk[idx2] = sqrt(PI / tau) * exp(tau * k * k) * Fktau1[idx1] / (double)(Mr); //can optimize here
    }

    delete[] E3;
    delete[] Fktau1;
    delete[] finter;
}

void nufft_iter(double *x, double *f, _MYCMPX *Fk, int N, int M, int R, int Msp, double df, bool flag) {

    const int Mr = R * M;
    const double tau = PI * (Msp) / M / M / R / (R - .5);

    double *E3 = new double[Msp + 1];
    _MYCMPX *Fktau = new _MYCMPX[Mr];

    for (int l = 0; l <= Msp; l++)
        E3[l] = exp(-pow(PI * l / Mr, 2) / tau);

    if (!flag) {

        nonuni2uni1_iter(f, x, Fktau, N, Mr, Msp, tau, E3, df);
    } else {
        nonuni2uni2_iter(x, Fktau, N, Mr, Msp, tau, E3, df);
    }

    fft_iter(Fktau, Mr);
    for (int k = -M / 2; k <= (M - 1) / 2; k++) {

        int idx1 = MOD(k, Mr);
        int idx2 = MOD(k, M);
        Fk[idx2] = Fktau[idx1] * sqrt(PI / tau) * exp(tau * k * k) / (double)(Mr); //can optimize here
    }

    delete[] E3;
    delete[] Fktau;
}

void extripolation(double *t, double *x, double fac, int nfreq, int macc, double *wk1, double *wk2, int nt) {

    int ndim = 2 * nfreq;
    for (int j = 0; j < nt; j++) {

        double ck = MODF(t[j] * fac, ndim) + 1;
        double ckk = MODF(2 * (ck - 1), ndim) + 1;
        spread(x[j], wk1, ck, ndim, macc);
        spread(1, wk2, ckk, ndim, macc);
    }
}

void extripolation_iter(double *t, double *x, double fac, int nfreq, int macc, _MYCMPX *wk1, _MYCMPX *wk2, int nt) {

    int ndim = 2 * nfreq;
    for (int j = 0; j < nt; j++) {

        double ck = MODF(t[j] * fac, ndim) + 1;
        double ckk = MODF(2 * (ck - 1), ndim) + 1;
        spread_iter(x[j], wk1, ck, ndim, macc);
        spread_iter(1, wk2, ckk, ndim, macc);
    }
}

void spread(double y, double *yy, double x, int n, int m) {

    const double nfac[10] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880};

    if (round(x) == x) {

        yy[(int)(x - 1)] = yy[(int)(x - 1)] + y;
    } else {

        double tmp1 = MAX(floor(x - .5 * m + 1), 1);
        int i1 = MIN(tmp1, n - m + 1);

        int i2 = i1 + m - 1;
        double nden = nfac[m - 1];
        double fac = x - i1;

        double prod = 1;
        for (int k = i1 + 1; k <= i2; k++)
            prod *= (x - (double)k);

        fac = fac * prod;
        yy[i2 - 1] += y * fac / (nden * (x - i2));

        for (int j = i2 - 1; j >= i1; j--) {
            nden = (nden / (j + 1 - i1)) * (j - i2);
            yy[j - 1] += y * fac / (nden * (x - j));
        }
    }
}

void spread_iter(double y, _MYCMPX *yy, double x, int n, int m) {

    const double nfac[10] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880};

    if (round(x) == x) {

        yy[(int)(x - 1)] = yy[(int)(x - 1)] + y;
    } else {

        double tmp1 = MAX(floor(x - .5 * m + 1), 1);
        int i1 = MIN(tmp1, n - m + 1);

        int i2 = i1 + m - 1;
        double nden = nfac[m - 1];
        double fac = x - i1;

        double prod = 1;
        for (int k = i1 + 1; k <= i2; k++)
            prod *= (x - (double)k);

        fac = fac * prod;
        yy[i2 - 1] = yy[i2 - 1] + y * fac / (nden * (x - i2));

        for (int j = i2 - 1; j >= i1; j--) {
            nden = (nden / (j + 1 - i1)) * (j - i2);
            yy[j - 1] = yy[j - 1] + y * fac / (nden * (x - j));
        }
    }
}

constexpr double evaluate_trig(double sumh_real, double sumh_image, double sum2_real, double sum2_image, double nt) {
    double hypo = sqrt(sum2_real * sum2_real + sum2_image * sum2_image);
    double hc2wt = 0.5 * sum2_real / hypo;
    double hs2wt = 0.5 * sum2_image / hypo;
    double cwt = sqrt(0.5 + hc2wt);
    double swt = ((hs2wt >= 0) ? 1 : -1) * sqrt(0.5 - hc2wt);
    double den = 0.5 * nt + hc2wt * sum2_real + hs2wt * sum2_image;
    double cterm = pow(cwt * sumh_real + swt * sumh_image, 2) / den;
    double sterm = pow(cwt * sumh_image - swt * sumh_real, 2) / (nt - den);
    return (cterm + sterm) * nt;
}

void fastlomb_extri(double *Pextri, double *x, double *t, double *fin, int nt, int nf, const unsigned int macc) {

    double expo = floor(2 * nf * macc);
    int nfreq = (int)pow(2, ceil(log2(expo)));
    double *wk1 = new double[2 * nfreq];
    double *wk2 = new double[2 * nfreq];
    _CMPXD *Wk1 = new _CMPXD[2 * nfreq];
    _CMPXD *Wk2 = new _CMPXD[2 * nfreq];

    double fac = 2 * nfreq * fin[0];

    extripolation(t, x, fac, nfreq, macc, wk1, wk2, nt);

    fftd(wk1, Wk1, 2 * nfreq);
    fftd(wk2, Wk2, 2 * nfreq);

    for (int i = 0; i < nf; i++)
        Pextri[i] = evaluate_trig(Wk1[i + 1].real(), Wk1[i + 1].imag(), Wk2[i + 1].real(), Wk2[i + 1].imag(), nt);

    delete[] wk1;
    delete[] wk2;
    delete[] Wk1;
    delete[] Wk2;
}

void fastlomb_extri_iter(double *Pextri, double *x, double *t, double *fin, int nt, int nf, const unsigned int macc) {

    double expo = floor(2 * nf * macc);
    int nfreq = (int)pow(2, ceil(log2(expo)));

    _MYCMPX *wk1 = new _MYCMPX[2 * nfreq];
    _MYCMPX *wk2 = new _MYCMPX[2 * nfreq];

    double fac = 2 * nfreq * fin[0];

    extripolation_iter(t, x, fac, nfreq, macc, wk1, wk2, nt);

    fft_iter(wk1, (unsigned int)2 * nfreq);
    fft_iter(wk2, (unsigned int)2 * nfreq);

    for (int i = 0; i < nf; i++)
        Pextri[i] = evaluate_trig(wk1[i + 1].real, wk1[i + 1].imag, wk2[i + 1].real, wk2[i + 1].imag, nt);
}

void fastlomb_gaussian(double *Pnfft, double *x, double *t, double *fin, int nt, int nf, const unsigned int R, const unsigned int Msp) {

    _CMPXD *Fkh = new _CMPXD[2 * nf];
    _CMPXD *Fk2 = new _CMPXD[4 * nf];
    double df = fin[0];

    nufft(t, x, Fkh, nt, 2 * nf, R, Msp, df, 0);
    nufft(t, x, Fk2, nt, 4 * nf, R, Msp, df, 1);

    for (int i = 0; i < nf; i++)
        Pnfft[i] = evaluate_trig(Fkh[i + 1].real(), Fkh[i + 1].imag(), Fk2[2 * i + 2].real(), Fk2[2 * i + 2].imag(), nt);

    delete[] Fkh;
    delete[] Fk2;
}

void fastlomb_gaussian_iter(double *Pnfft, double *x, double *t, double *fin, int nt, int nf, const unsigned int R, const unsigned int Msp) {

    _MYCMPX *Fkh = new _MYCMPX[2 * nf];
    _MYCMPX *Fk2 = new _MYCMPX[4 * nf];
    double df = fin[0];

    nufft_iter(t, x, Fkh, nt, 2 * nf, R, Msp, df, 0);
    nufft_iter(t, x, Fk2, nt, 4 * nf, R, Msp, df, 1);

    for (int i = 0; i < nf; i++)
        Pnfft[i] = evaluate_trig(Fkh[i + 1].real, Fkh[i + 1].imag, Fk2[2 * i + 2].real, Fk2[2 * i + 2].imag, nt);

    delete[] Fkh;
    delete[] Fk2;
}
void lombgolden(double *Pgolden, double *x, double *t, double *fin, int nt, int nf) {

    for (int i = 0; i < nf; i++) {
        double sumh_real = 0;
        double sumh_image = 0;
        double sum2_real = 0;
        double sum2_image = 0;
        for (int j = 0; j < nt; j++) {
            sumh_real += x[j] * cos(2 * PI * fin[i] * t[j]);
            sumh_image += x[j] * sin(2 * PI * fin[i] * t[j]);
            sum2_real += cos(4 * PI * fin[i] * t[j]);
            sum2_image += sin(4 * PI * fin[i] * t[j]);
        }
        Pgolden[i] = evaluate_trig(sumh_real, sumh_image, sum2_real, sum2_image, (double)nt);
    }
}

int main() {
    //time initializations
    double seed = 100;
    int N = 16; // input points
    double fmax = 0.4;
    double T;
    int Nout;
    double *rr_out = new double[N];
    double *t = new double[N];

    const unsigned int MACC = 8;
    const unsigned int R = 2;
    const unsigned int Msp = 12;

    rrgen((double)N, seed, rr_out);

    t[0] = 0;
    for (int i = 1; i < N; i++)
        t[i] = t[i - 1] + rr_out[i - 1];

    T = t[N - 1] - t[0];
    double aver = mean(rr_out, N);

    for (int i = 0; i < N; i++)
        rr_out[i] -= aver;

    //power-freq initializations
    Nout = pow(2, ceil(log2(T * fmax)));
    double *fin = new double[Nout];
    double *Pgolden = new double[Nout];
    double *Pnfft = new double[Nout];
    double *Pextri = new double[Nout];

    for (int i = 0; i < Nout; i++)
        fin[i] = fmax / Nout * (i + 1);

    lombgolden(Pgolden, rr_out, t, fin, N, Nout);
    fastlomb_extri_iter(Pextri, rr_out, t, fin, N, Nout, MACC);
    fastlomb_gaussian_iter(Pnfft, rr_out, t, fin, N, Nout, R, Msp);

    LOGVEC(Pnfft, Nout);
    LOGVECDIFF(Pnfft, Pgolden, Nout);
    LOGVECDIFF(Pextri, Pgolden, Nout);

    delete[] Pgolden;
    delete[] Pextri;
    delete[] Pnfft;
    delete[] fin;
    delete[] rr_out;
    delete[] t;

    return 0;
}
