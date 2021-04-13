#ifndef _LOMB
#define _LOMB

void fastlomb_extri(double *Pextri, double *x, double *t, double *fin, int nt,
                    int nf, const unsigned int macc);

void lombgolden(double *Pgolden, double *x, double *t, double *fin, int nt,
                int nf);

void fastlomb_gaussian(double *Pnfft, double *x, double *t, double *fin, int nt,
                       int nf, const unsigned int R, const unsigned int Msp);

#endif