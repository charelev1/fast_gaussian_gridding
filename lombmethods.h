#ifndef _LOMB
#define _LOMB

class _TIME {
   public:
    double grid;
    double fft;
    double lomb;

    _TIME() {
        grid = 0;
        fft = 0;
        lomb = 0;
    }

};

void lombgolden(double *Pgolden, double *x, double *t, double *fin, int nt,
                int nf, _TIME &golden);

void fastlomb_extri(double *Pextri, double *x, double *t, double *fin, int nt,
                    int nf, const unsigned int macc, _TIME &extrip);

void fastlomb_gaussian(double *Pnfft, double *x, double *t, double *fin, int nt,
                       int nf, const unsigned int R, const unsigned int Msp,
                       _TIME &gaussi);

#endif