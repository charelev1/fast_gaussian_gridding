#ifndef _RRGEN
#define __RRGEN

void rrgen(long tmax, long seed, double *rr_inter);

float randn();

void bsort(int *a, int *b, int n);

void initialize(long seed, long tmax);

float generate();
#endif