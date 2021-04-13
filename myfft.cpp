#include <cmath>

#include "myfft.h"
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#define SWAP(x, y) \
    auto tmp = x;  \
    x = y;         \
    y = tmp

#define LOGVEC(x, N)                                           \
    for (int i = 0; i < N; i++) {                              \
        std::cout << #x << "[" << i << "] = " << x[i] << "\n"; \
    }                                                          \
    std::cout << "\n"

#define PI 3.141592653589793238462643383279502884197169399375105820974944

constexpr unsigned int reversebits(unsigned int val_in, unsigned int k) {

    unsigned int val_out = 0;
    unsigned int mask = 1;

    for (int i = k - 1; i >= 0; i--) {
        if (val_in & mask) {
            val_out += 1 << i;
        }
        mask <<= 1;
    }
    return val_out;
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
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 1; i < N - 1; i++) {
        j = reversebitsLookUp(i, m);
        if (i < j) {
            SWAP(data[j], data[i]);
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;

    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

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

/*
void fft_iter(_MYCMPX *data, unsigned int N) {

    unsigned int m = (int)log2(N);
    unsigned int j;
    printf("m = %i\n", m);

    for (unsigned int i = 1; i < N - 1; i++) {
        j = reversebits(i, m);
        printf("%i %i \n ", i, j);

        if (i < j) {
            SWAP(data[j], data[i]);
        }
    }

    for (unsigned int i = 0; i < N; i++) {
        printf("data[%i] = (%f,%f)\n ", i, data[i].real, data[i].imag);
    }

    _MYCMPX factor;
    unsigned int Ninter = 1;
    // unsigned int Nintra = N / 2;

    for (unsigned int i = 0; i < m; i++) {
        double step = -6.283185307179586 / (Ninter * 2);
        double a = 0;
        printf("i = %i\n ", i);
        printf("Ninter = %i\n ", Ninter);

        for (unsigned int j = 0; j < Ninter; j++) { //inter loop
            factor.real = cos(a);
            factor.imag = sin(a);
            a += step;
            printf("    j = %i\n ", j);
            printf("    factor = (%f,%f)\n ", factor.real, factor.imag);
            for (unsigned int k = 0; k < N; k += 2 * Ninter) {

                //  printf("        idx1 = %i ", k + j);
                //  printf(" idx2 = %i\n ", k + j + Ninter);
                //   printf(" factor = (%f,%f)\n ", factor.real, factor.imag);
                int idx1 = k + j;
                int idx2 = k + j + Ninter;

                printf("        data[%i] = (%f,%f)\n ", idx1, data[idx1].real, data[idx1].imag);
                printf("        data[%i] = (%f,%f)\n ", idx2, data[idx2].real, data[idx2].imag);

                _MYCMPX tmp;
                tmp = data[idx2] * factor;
                data[idx2] = data[idx1] - tmp;
                data[idx1] = data[idx1] + tmp;
                printf("        tmp = (%f,%f)\n ", tmp.real, tmp.imag);
                printf("        data[%i] = (%f,%f)\n ", idx1, data[idx1].real, data[idx1].imag);
                printf("        data[%i] = (%f,%f)\n\n ", idx2, data[idx2].real, data[idx2].imag);

                // data[k+Ninter] = data[k+Ninter]*factor;
            }
        }

        Ninter *= 2;
        // Nintra /= 2;
    }
}
*/
int main() {

    const unsigned int N = 8;

    _MYCMPX data[N];
    _MYCMPX test;

    for (unsigned int i = 0; i < N; i++) {
        data[i].real = i;
    }
    test = 10;
    test = test-1;
    test = test+3;
    test = test*2;
    test(4,5);
    fft_iter(data, N);
    printf("\ndtest(%f,%f)\n ", test.real, test.imag);

    for (unsigned int i = 0; i < N; i++) {
        //   printf("\ndata[%i] = (%f,%f)\n ", i, data[i].real, data[i].imag);
    }
    return 0;
}