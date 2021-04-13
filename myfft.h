#ifndef _MYFFT
#define __MYFFT

class _MYCMPX {
public:
    double real;
    double imag;

    _MYCMPX() {
        real = 0;
        imag = 0;
    }

    _MYCMPX(double a, double b) {
        real = a;
        imag = b;
    }

    _MYCMPX(const _MYCMPX &b) {
        this->real = b.real;
        this->imag = b.imag;
    }

    void set(double a, double b) {
        real = a;
        real = b;
    }

    _MYCMPX &operator=(const double &b) {
        this->real = b;
        this->imag = 0;
        return *this;
    }

    _MYCMPX &operator=(const _MYCMPX &b) {
        this->real = b.real;
        this->imag = b.imag;
        return *this;
    }

    _MYCMPX operator+(const _MYCMPX &b) {
        _MYCMPX complex;
        complex.real = this->real + b.real;
        complex.imag = this->imag + b.imag;
        return complex;
    }

    _MYCMPX operator+(const double &b) {
        _MYCMPX complex;
        complex.real = this->real + b;
        complex.imag = this->imag + 0;
        return complex;
    }

    _MYCMPX operator-(const _MYCMPX &b) {
        _MYCMPX complex;
        complex.real = this->real - b.real;
        complex.imag = this->imag - b.imag;
        return complex;
    }

    _MYCMPX operator-(const double &b) {
        _MYCMPX complex;
        complex.real = this->real - b;
        complex.imag = this->imag + 0;
        return complex;
    }

    _MYCMPX operator*(const _MYCMPX &b) {
        _MYCMPX complex;
        complex.real = this->real * b.real - this->imag * b.imag;
        complex.imag = this->imag * b.real + this->real * b.imag;
        return complex;
    }

    _MYCMPX operator*(const double &b) {
        _MYCMPX complex;
        complex.real = this->real * b;
        complex.imag = this->imag * b;
        return complex;
    }


    
    _MYCMPX operator/(const double &b) {
        _MYCMPX complex;
        complex.real = this->real / b;
        complex.imag = this->imag / b;
        return complex;
    }

    void operator()(const double &a,const double &b) {
        this->real = a;
        this->imag = b;
    }

};

void fft_iter(_MYCMPX *data, unsigned int N);

#endif