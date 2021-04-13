
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