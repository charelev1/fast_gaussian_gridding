#include <cmath>
#include <cstdlib>
#include <iostream>

#include "myfft.h"

#define PI 3.1415926535897932384626433832795028841971693993751058
#define MOD(y, x) (x + y) % x
#define MODF(x, y) std::fmod((double)x, (double)y)
#define MIN(x, y) (x < y) ? x : y
#define MAX(x, y) (x > y) ? x : y
#define LOGVAR(x) std::cout << #x << " = " << x << "\n"
#define LOG(x) std::cout << x << "\n"
constexpr double evaluate_trig(double sumh_real, double sumh_image,
                               double sum2_real, double sum2_image, double nt) {
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

void spread(double y, _MYCMPX *yy, double x, int n, int m) {
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
        for (int k = i1 + 1; k <= i2; k++) {
            prod *= (x - (double)k);
        }

        fac = fac * prod;
        yy[i2 - 1] = yy[i2 - 1] + y * fac / (nden * (x - i2));

        for (int j = i2 - 1; j >= i1; j--) {
            nden = (nden / (j + 1 - i1)) * (j - i2);
            yy[j - 1] = yy[j - 1] + y * fac / (nden * (x - j));
        }
    }
}

void extripolation(double *t, double *x, double fac, int nfreq, int macc,
                   _MYCMPX *wk1, _MYCMPX *wk2, int nt) {
    int ndim = 2 * nfreq;
    for (int j = 0; j < nt; j++) {
        double ck = MODF(t[j] * fac, ndim) + 1;
        double ckk = MODF(2 * (ck - 1), ndim) + 1;
        spread(x[j], wk1, ck, ndim, macc);
        spread(1, wk2, ckk, ndim, macc);
    }
}

void fastlomb_extri(double *Pextri, double *x, double *t, double *fin, int nt,
                    int nf, const unsigned int macc) {
    double expo = floor(2 * nf * macc);
    int nfreq = (int)pow(2, ceil(log2(expo)));

    _MYCMPX *wk1 = new _MYCMPX[2 * nfreq];
    _MYCMPX *wk2 = new _MYCMPX[2 * nfreq];

    double fac = 2 * nfreq * fin[0];

    extripolation(t, x, fac, nfreq, macc, wk1, wk2, nt);

    fft_iter(wk1, (unsigned int)2 * nfreq);
    LOGVAR(2 * nfreq);
    fft_iter(wk2, (unsigned int)2 * nfreq);

    for (int i = 0; i < nf; i++) {
        Pextri[i] = evaluate_trig(wk1[i + 1].real, wk1[i + 1].imag,
                                  wk2[i + 1].real, wk2[i + 1].imag, nt);
    }
    delete[] wk1;
    delete[] wk2;
}

void lombgolden(double *Pgolden, double *x, double *t, double *fin, int nt,
                int nf) {
    for (int i = 0; i < nf; i++) {
        double sumh_real = 0;
        double sumh_image = 0;
        double sum2_real = 0;
        double sum2_image = 0;
        for (int j = 0; j < nt; j++) {
            double ft = 2 * PI * fin[i] * t[j];
            sumh_real += x[j] * cos(ft);
            sumh_image += x[j] * sin(ft);
            sum2_real += cos(2 * ft);
            sum2_image += sin(2 * ft);
        }

        Pgolden[i] = evaluate_trig(sumh_real, sumh_image, sum2_real, sum2_image,
                                   (double)nt);
    }
}

void nonuni2uni1(double *f, double *x, _MYCMPX *fout, int N, int Mr, int Msp,
                 double tau, double *E3, double df) {
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

void nonuni2uni2(double *x, _MYCMPX *fout, int N, int Mr, int Msp, double tau,
                 double *E3, double df) {
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

void nufft(double *x, double *f, _MYCMPX *Fk, int N, int M, int R, int Msp,
           double df, bool flag) {
    const unsigned int Mr = R * M;
    const double tau = PI * (Msp) / M / M / R / (R - .5);

    double *E3 = new double[Msp + 1];
    _MYCMPX *Fktau = new _MYCMPX[Mr];

    for (int l = 0; l <= Msp; l++) E3[l] = exp(-pow(PI * l / Mr, 2) / tau);

    if (!flag) {
        nonuni2uni1(f, x, Fktau, N, Mr, Msp, tau, E3, df);
    } else {
        nonuni2uni2(x, Fktau, N, Mr, Msp, tau, E3, df);
    }

    fft_iter(Fktau, Mr);
        LOGVAR(Mr);

    for (int k = -M / 2; k <= (M - 1) / 2; k++) {
        int idx1 = MOD(k, Mr);
        int idx2 = MOD(k, M);
        Fk[idx2] = Fktau[idx1] * sqrt(PI / tau) * exp(tau * k * k) /
                   (double)(Mr);  // can optimize here
    }

    delete[] E3;
    delete[] Fktau;
}

void fastlomb_gaussian(double *Pnfft, double *x, double *t, double *fin, int nt,
                       int nf, const unsigned int R, const unsigned int Msp) {
    _MYCMPX *Fkh = new _MYCMPX[2 * nf];
    _MYCMPX *Fk2 = new _MYCMPX[4 * nf];
    double df = fin[0];

    nufft(t, x, Fkh, nt, 2 * nf, R, Msp, df, 0);
    nufft(t, x, Fk2, nt, 4 * nf, R, Msp, df, 1);

    for (int i = 0; i < nf; i++) {
        Pnfft[i] = evaluate_trig(Fkh[i + 1].real, Fkh[i + 1].imag,
                                 Fk2[2 * i + 2].real, Fk2[2 * i + 2].imag, nt);
    }

    delete[] Fkh;
    delete[] Fk2;
}
