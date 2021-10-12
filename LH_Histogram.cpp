#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_trans.h"
#include "mkl_scalapack.h"
#include "omp.h"
#include "mt19937ar.h"
#include "ziggurat.hpp"

class RandomGeneratorMT19937ar {
private:
    bool ready;
    double second = 0.0;
    double mean, stddev;

public:
    explicit RandomGeneratorMT19937ar(double mean = 0.0, double stddev = 1.0, size_t seed = 0)
        : mean(mean), stddev(stddev), ready(false) {
        init_genrand(seed);
    }

    double Generate() {
        if (ready) {
            ready = false;
            return second * stddev + mean;
        }
        else {
            double u, v, s;
            do {
                u = 2.0 * genrand_real2() - 1.0;
                v = 2.0 * genrand_real2() - 1.0;
                s = u * u + v * v;
            } while (s > 1.0 || s == 0.0);

            double r = std::sqrt(-2.0 * std::log(s) / s);
            second = r * u;
            ready = true;
            return r * v * stddev + mean;
        }
    }
};

class ComplexMatrix {
private:
    MKL_Complex16* matrix;
    size_t rows, cols;

public:
    enum class PrintType {
        ONLY_REAL,
        ONLY_IMAGINARY,
        BOTH
    };

    ComplexMatrix(size_t rows, size_t cols, MKL_Complex16 value) : rows(rows), cols(cols) {
        matrix = new MKL_Complex16[rows * cols];
        //#pragma omp parallel for
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                matrix[i * rows + j] = value;
            }
        }
    }

    ComplexMatrix(const ComplexMatrix& src) : rows(src.rows), cols(src.cols) {
        matrix = new MKL_Complex16[rows * cols];
        //#pragma omp parallel for
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                matrix[i * rows + j] = src.matrix[i * rows + j];
            }
        }
    }

    ComplexMatrix(ComplexMatrix&& src) noexcept {
        rows = src.rows;
        cols = src.cols;

        matrix = src.matrix;
        src.matrix = nullptr;
    }

    ComplexMatrix& operator=(const ComplexMatrix& src) {
        rows = src.rows;
        cols = src.cols;

        if (matrix != nullptr) {
            this->~ComplexMatrix();
        }

        matrix = new MKL_Complex16[rows * cols];
        //#pragma omp parallel for
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                matrix[i * rows + j] = src.matrix[i * rows + j];
            }
        }

        return *this;
    }

    ComplexMatrix& operator=(ComplexMatrix&& src) noexcept {
        rows = src.rows;
        cols = src.cols;
        std::swap(matrix, src.matrix);

        return *this;
    }

    ~ComplexMatrix() {
        delete matrix;
    }

    MKL_Complex16& operator[](size_t i) {
        return matrix[i];
    }

    const MKL_Complex16& operator[](size_t i) const {
        return matrix[i];
    }

    MKL_Complex16& operator()(size_t i, size_t j) {
        return matrix[i * rows + j];
    }

    const MKL_Complex16& operator()(size_t i, size_t j) const {
        return matrix[i * rows + j];
    }

    double Trace() const {
        double trace = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            trace += this->operator()(i, i).real;
        }
        return trace;
    }

    ComplexMatrix Transpose() const {
        MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
        ComplexMatrix result(cols, rows, zero);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = this->operator()(i, j);
            }
        }

        return result;
    }

    friend ComplexMatrix operator+(const ComplexMatrix& lhs, const ComplexMatrix& rhs) {
        if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
            throw "dsf";
        }

        MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
        ComplexMatrix result(lhs.rows, lhs.cols, zero);
        //#pragma omp parallel for
        for (size_t i = 0; i < lhs.rows; ++i) {
            for (size_t j = 0; j < lhs.cols; ++j) {
                result(i, j).real = lhs(i, j).real + rhs(i, j).real;
                result(i, j).imag = lhs(i, j).imag + rhs(i, j).imag;
            }
        }

        return result;
    }

    ComplexMatrix& operator+=(const ComplexMatrix& rhs) {
        if (rows != rhs.rows || cols != rhs.cols) {
            throw "dsf";
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                this->operator()(i, j).real += rhs(i, j).real;
                this->operator()(i, j).imag += rhs(i, j).imag;
            }
        }

        return *this;
    }

    friend ComplexMatrix operator-(const ComplexMatrix& lhs, const ComplexMatrix& rhs) {
        if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
            throw "dsf";
        }

        MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
        ComplexMatrix result(lhs.rows, lhs.cols, zero);
        //#pragma omp parallel for
        for (size_t i = 0; i < lhs.rows; ++i) {
            for (size_t j = 0; j < lhs.cols; ++j) {
                result(i, j).real = lhs(i, j).real - rhs(i, j).real;
                result(i, j).imag = lhs(i, j).imag - rhs(i, j).imag;
            }
        }

        return result;
    }

    ComplexMatrix& operator-=(const ComplexMatrix& rhs) {
        if (rows != rhs.rows || cols != rhs.cols) {
            throw "dsf";
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                this->operator()(i, j).real -= rhs(i, j).real;
                this->operator()(i, j).imag -= rhs(i, j).imag;
            }
        }

        return *this;
    }

    friend ComplexMatrix operator*(const ComplexMatrix& lhs, const ComplexMatrix& rhs) {
        if (lhs.cols != rhs.rows) {
            throw "dsf";
        }

        MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
        ComplexMatrix result(lhs.rows, lhs.cols, zero);
        //#pragma omp parallel for
        for (size_t i = 0; i < lhs.rows; ++i) {
            for (size_t j = 0; j < rhs.cols; ++j) {
                for (size_t k = 0; k < lhs.cols; ++k) {
                    result(i, j).real += lhs(i, k).real * rhs(k, j).real - lhs(i, k).imag * rhs(k, j).imag;
                    result(i, j).imag += lhs(i, k).real * rhs(k, j).imag + lhs(i, k).imag * rhs(k, j).real;
                }
            }
        }

        result.normalZero();
        return result;
    }

    friend ComplexMatrix operator*(const ComplexMatrix& src, double value) {
        MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
        ComplexMatrix result(src.rows, src.cols, zero);
        //#pragma omp parallel for
        for (size_t i = 0; i < src.rows; ++i) {
            for (size_t j = 0; j < src.cols; ++j) {
                result(i, j).real = src(i, j).real * value;
                result(i, j).imag = src(i, j).imag * value;
            }
        }

        result.normalZero();
        return result;
    }

    friend ComplexMatrix operator*(double value, const ComplexMatrix& src) {
        MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
        ComplexMatrix result(src.rows, src.cols, zero);
        //#pragma omp parallel for
        for (size_t i = 0; i < src.rows; ++i) {
            for (size_t j = 0; j < src.cols; ++j) {
                result(i, j).real = src(i, j).real * value;
                result(i, j).imag = src(i, j).imag * value;
            }
        }

        result.normalZero();
        return result;
    }

    friend ComplexMatrix operator*(const ComplexMatrix& src, MKL_Complex16 value) {
        MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
        ComplexMatrix result(src.rows, src.cols, zero);
        //#pragma omp parallel for
        for (size_t i = 0; i < src.rows; ++i) {
            for (size_t j = 0; j < src.cols; ++j) {
                result(i, j).real = src(i, j).real * value.real - src(i, j).imag * value.imag;
                result(i, j).imag = src(i, j).real * value.imag + src(i, j).imag * value.real;
            }
        }

        result.normalZero();
        return result;
    }

    friend ComplexMatrix operator*(MKL_Complex16 value, const ComplexMatrix& src) {
        MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
        ComplexMatrix result(src.rows, src.cols, zero);
        //#pragma omp parallel for
        for (size_t i = 0; i < src.rows; ++i) {
            for (size_t j = 0; j < src.cols; ++j) {
                result(i, j).real = src(i, j).real * value.real - src(i, j).imag * value.imag;
                result(i, j).imag = src(i, j).real * value.imag + src(i, j).imag * value.real;
            }
        }

        result.normalZero();
        return result;
    }

    friend ComplexMatrix KronekerProduct(const ComplexMatrix& lhs, const ComplexMatrix& rhs) {
        MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
        ComplexMatrix result(lhs.rows * rhs.rows, lhs.cols * rhs.cols, zero);

        //#pragma omp parallel for
        for (size_t i = 0; i < lhs.rows; ++i) {
            for (size_t j = 0; j < lhs.cols; ++j) {
                for (size_t k = 0; k < rhs.rows; ++k) {
                    for (size_t l = 0; l < rhs.cols; ++l) {
                        result(i * rhs.rows + k, j * rhs.cols + l).real =
                            lhs(i, j).real * rhs(k, l).real - lhs(i, j).imag * rhs(k, l).imag;
                        result(i * rhs.rows + k, j * rhs.cols + l).imag =
                            lhs(i, j).real * rhs(k, l).imag + lhs(i, j).imag * rhs(k, l).real;
                    }
                }
            }
        }

        result.normalZero();
        return result;
    }

    MKL_Complex16* GetMatrix() const {
        return matrix;
    }

    void PrintMatrix(PrintType printType) const {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                if (printType == PrintType::ONLY_REAL) {
                    std::cout << "[" << i << "," << j << "]:" << matrix[i * rows + j].real << "\t";
                }
                else if (printType == PrintType::ONLY_IMAGINARY) {
                    std::cout << "[" << i << "," << j << "]:" << matrix[i * rows + j].imag << "\t";
                }
                else if (printType == PrintType::BOTH) {
                    std::cout << "[" << i << "," << j << "]:" << matrix[i * rows + j].real << "+" << matrix[i * rows + j].imag << "\t";
                }
            }
            std::cout << std::endl;
        }
    }

    void SaveMatrix(const std::string path, PrintType printType) const {
        std::ofstream file(path.c_str());
        if (file.is_open()) {
            for (size_t i = 0; i < this->rows; ++i) {
                for (size_t j = 0; j < this->cols; ++j) {
                    if (printType == PrintType::ONLY_REAL) {
                        file << "[" << i << "," << j << "]:" << matrix[i * rows + j].real << "\t";
                    }
                    else if (printType == PrintType::ONLY_IMAGINARY) {
                        file << "[" << i << "," << j << "]:" << matrix[i * rows + j].imag << "\t";
                    }
                    else if (printType == PrintType::BOTH) {
                        file << "[" << i << "," << j << "]:" << matrix[i * rows + j].real << "+" << matrix[i * rows + j].imag << "\t";
                    }
                }
                file << std::endl;
            }
            file.close();
        }
        else {
            throw "Writing to file failed";
        }
    }

    void SaveMatrixForMatlab(const std::string path, PrintType printType) const {
        std::ofstream file(path.c_str());
        if (file.is_open()) {
            for (size_t i = 0; i < this->rows; ++i) {
                for (size_t j = 0; j < this->cols; ++j) {
                    if (printType == PrintType::ONLY_REAL) {
                        file << matrix[i * rows + j].real << "\t";
                    }
                    else if (printType == PrintType::ONLY_IMAGINARY) {
                        file << matrix[i * rows + j].imag << "\t";
                    }
                }
                file << std::endl;
            }
            file.close();
        }
        else {
            throw "Writing to file failed";
        }
    }

    void normalZero() {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                if (matrix[i * rows + j].imag == -0) {
                    matrix[i * rows + j].imag = 0;
                }
                else if (matrix[i * rows + j].real == -0) {
                    matrix[i * rows + j].real = 0;
                }
            }
        }
    }
};

constexpr size_t N = 4;
constexpr size_t M = N * N - 1;
constexpr size_t NUMBER_OF_IMPLEMENTATIONS = 10000; // 1'000'000;

MKL_Complex16 mul(MKL_Complex16 complex, double value) {
    MKL_Complex16 result;

    result.real = complex.real * value;
    result.imag = complex.imag * value;

    return result;
}

int main() {

    auto t1 = std::chrono::high_resolution_clock::now();

    MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
    double alpha_H = 1;

    std::vector<ComplexMatrix> F(N * N, ComplexMatrix(N, N, zero));

    ComplexMatrix eye(N, N, zero);
    for (size_t i = 0; i < N; ++i) {
        eye(i, i).real = 1;
    }

    for (size_t i = 0; i < N; ++i) {
        F[0](i, i).real = 1.0 / std::sqrt(N);
    }

    size_t k = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            ++k;
            F[k](i, j).real = F[k](j, i).real = 1.0 / std::sqrt(2);

            ++k;
            F[k](i, j).imag = -1.0 / std::sqrt(2);
            F[k](j, i).imag = 1.0 / std::sqrt(2);
        }
    }

    for (size_t i = 1; i < N; ++i) {
        ++k;
        std::vector<double> tmp(i + 1, 1);
        tmp.back() = -static_cast<double>(i);

        for (size_t j = 0; j <= i; ++j) {
            F[k](j, j).real = tmp[j] / std::sqrt(i * (i + 1));
        }
    }

    // ziggurat
    //uint32_t kn[128];
    //float fn[128], wn[128];
    //r4_nor_setup(kn, fn, wn);

    for (size_t number = 0; number < NUMBER_OF_IMPLEMENTATIONS; ++number) {
        ComplexMatrix X(M, M, zero);
        size_t seed = number;

        // uint32_t seed = number;
        /*RandomGeneratorMT19937ar gen_y(0, 1, seed);*/

        //for (int i = 0; i < M; ++i) {
        //    for (int j = 0; j < M; ++j) {
        //        /*X(i, j).real = gen_y.Generate() / 2.0;*/
        //        /*X(i, j).real = r4_nor(seed, kn, fn, wn) / 2.0;*/
        //        X(i, j).real = 1;
        //    }
        //}

        std::mt19937 gen_y;
        gen_y.seed(seed);
        std::normal_distribution<double> distribution_y{ 0.0, 1.0 };

        seed = NUMBER_OF_IMPLEMENTATIONS + number;

        /*RandomGeneratorMT19937ar gen_z(0, 1, seed);*/

        std::mt19937 gen_z;
        gen_z.seed(seed);
        std::normal_distribution<double> distribution_z{ 0.0, 1.0 };

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                X(i, j).real = distribution_y(gen_y) / 2.0;
                X(i, j).imag = distribution_z(gen_z) / 2.0;

                /*X(i, j).imag = gen_z.Generate() / 2.0;*/
                /*X(i, j).imag = r4_nor(seed, kn, fn, wn) / 2.0;*/
                // X(i, j).imag = 1;
            }
        }

        MKL_Complex16 alpha;
        alpha.real = 1.0;
        alpha.imag = 0.0;

        ComplexMatrix X_conj(M, M, zero);
        mkl_zomatcopy('R', 'C', M, M, alpha, X.GetMatrix(), M, X_conj.GetMatrix(), M);  // row major, conjugate transposed

        ComplexMatrix G(M, M, zero);

        G = X * X_conj;

        double trace_G = G.Trace();
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < M; ++j) {
                G(i, j).real *= N / trace_G;
                G(i, j).imag *= N / trace_G;
            }
        }

        // Генерация Y - матрицы для  Гамильтоновой части - унитарной матрицы с нулевым следом
        ComplexMatrix Y(N, N, zero);

        seed = NUMBER_OF_IMPLEMENTATIONS * 2 + number;
        std::mt19937 gen_y_H;
        gen_y_H.seed(seed);
        std::normal_distribution<double> distribution_y_H{ 0.0, 1.0 };

        seed = NUMBER_OF_IMPLEMENTATIONS * 3 + number;
        std::mt19937 gen_z_H;
        gen_z_H.seed(seed);
        std::normal_distribution<double> distribution_z_H{ 0.0, 1.0 };

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                Y(i, j).real = distribution_y_H(gen_y_H);
                Y(i, j).imag = distribution_z_H(gen_z_H);
            }
        }

        // Генерация матриц Гамильтониана - H
        ComplexMatrix Y_conj(N, N, zero);
        mkl_zomatcopy('R', 'C', N, N, alpha, Y.GetMatrix(), N, Y_conj.GetMatrix(), N);  // row major, conjugate transposed

        ComplexMatrix H(N, N, zero);
        H = (Y + Y_conj) * 0.5;

        ComplexMatrix H_temp(N, N, zero);
        H_temp = H * H;

        double trace_H_temp = H_temp.Trace();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                H(i, j).real *= 1 / std::sqrt(trace_H_temp);
                H(i, j).imag *= 1 / std::sqrt(trace_H_temp);
            }
        }

        ComplexMatrix P(N * N, N * N, zero);
        for (size_t k1 = 0; k1 < M; ++k1) {
            for (size_t k2 = 0; k2 < M; ++k2) {
                ComplexMatrix F_conj(N, N, zero);
                ComplexMatrix F_conj_trans(N, N, zero);
                ComplexMatrix F_conj_trans_temp(N, N, zero);
                ComplexMatrix F_trans(N, N, zero);

                mkl_zomatcopy('R', 'R', N, N, alpha, F[k2 + 1].GetMatrix(), N, F_conj.GetMatrix(), N); // row major, only conjugated
                mkl_zomatcopy('R', 'C', N, N, alpha, F[k2 + 1].GetMatrix(), N, F_conj_trans.GetMatrix(), N); // row major, conjugate transposed
                mkl_zomatcopy('R', 'C', N, N, alpha, F[k2 + 1].GetMatrix(), N, F_conj_trans_temp.GetMatrix(), N); // row major, conjugate transposed
                mkl_zomatcopy('R', 'T', N, N, alpha, (F_conj_trans_temp * F[k1 + 1]).GetMatrix(), N, F_trans.GetMatrix(), N); // row major, only transposed

                F_conj.normalZero();
                F_conj_trans.normalZero();
                F_conj_trans_temp.normalZero();
                F_trans.normalZero();

                // cout of values for checking
                //std::cout << std::endl << "F_conj: " << std::endl;
                //F_conj.PrintMatrix(ComplexMatrix::PrintType::BOTH);
                //std::cout << std::endl << "F_conj_trans: " << std::endl;
                //F_conj_trans.PrintMatrix(ComplexMatrix::PrintType::BOTH);
                //std::cout << std::endl << "F_trans: " << std::endl;
                //F_trans.PrintMatrix(ComplexMatrix::PrintType::BOTH);
                //

                P += mul(G(k1, k2), 0.5) * (
                    2 * KronekerProduct(eye, F[k1 + 1]) * KronekerProduct(F_conj, eye) -
                    KronekerProduct(F_trans, eye) - KronekerProduct(eye, (F_conj_trans * F[k1 + 1]))
                    );

                // cout of values for checking
                //std::cout << std::endl << "KronekerProduct(eye, F[k1 + 1]): " << std::endl;
                //KronekerProduct(eye, F[k1 + 1]).PrintMatrix(ComplexMatrix::PrintType::BOTH);
                //std::cout << std::endl << "KronekerProduct(F_conj, eye): " << std::endl;
                //KronekerProduct(F_conj, eye).PrintMatrix(ComplexMatrix::PrintType::BOTH);
                //std::cout << std::endl << "KronekerProduct(F_trans, eye): " << std::endl;
                //KronekerProduct(F_trans, eye).PrintMatrix(ComplexMatrix::PrintType::BOTH);
                //std::cout << std::endl << "(F_conj_trans * F[k1 + 1]): " << std::endl;
                //(F_conj_trans * F[k1 + 1]).PrintMatrix(ComplexMatrix::PrintType::BOTH);
                //std::cout << std::endl << "KronekerProduct(eye, (F_conj_trans * F[k1 + 1])): " << std::endl;
                //KronekerProduct(eye, (F_conj_trans * F[k1 + 1])).PrintMatrix(ComplexMatrix::PrintType::BOTH);
                //std::cout << std::endl << "P: " << std::endl;
                //P.PrintMatrix(ComplexMatrix::PrintType::BOTH); // error
                //
            }
        }

        char str_re[128] = "";
        char str_im[128] = "";
        snprintf(str_re, sizeof(str_re), "Data_test\\P_matrix_re_%d.txt", number);
        snprintf(str_im, sizeof(str_im), "Data_test\\P_matrix_im_%d.txt", number);
        P.SaveMatrixForMatlab(str_re, ComplexMatrix::PrintType::ONLY_REAL);
        P.SaveMatrixForMatlab(str_im, ComplexMatrix::PrintType::ONLY_IMAGINARY);

        ComplexMatrix H_trans(N, N, zero);
        mkl_zomatcopy('R', 'T', N, N, alpha, H.GetMatrix(), N, H_trans.GetMatrix(), N); // row major, only transposed

        MKL_Complex16 imag; imag.real = 0, imag.imag = 1;

        // P = P - 1.0 * alpha * Im * (kron(H, eye(N)) - kron(eye(N), transpose(H))); // Гамильтонова часть
        P -= mul(imag, 1.0 * alpha_H) * (KronekerProduct(H, eye) - KronekerProduct(eye, H_trans));

        // cout of values for checking 
        //std::cout << std::endl << "H: " << std::endl;
        //H.PrintMatrix(ComplexMatrix::PrintType::BOTH);
        //std::cout << std::endl << "H_trans: " << std::endl;
        //H_trans.PrintMatrix(ComplexMatrix::PrintType::BOTH);
        //std::cout << std::endl << "KronekerProduct(H, eye): " << std::endl;
        //KronekerProduct(H, eye).PrintMatrix(ComplexMatrix::PrintType::BOTH);
        //std::cout << std::endl << "KronekerProduct(eye, H_trans): " << std::endl;
        //KronekerProduct(eye, H_trans).PrintMatrix(ComplexMatrix::PrintType::BOTH);
        //std::cout << std::endl << "P: " << std::endl;
        //P.PrintMatrix(ComplexMatrix::PrintType::BOTH); // error

        char str_re_H[128] = "";
        char str_im_H[128] = "";
        snprintf(str_re_H, sizeof(str_re_H), "Data_test\\P_H_matrix_re_%d.txt", number);
        snprintf(str_im_H, sizeof(str_im_H), "Data_test\\P_H_matrix_im_%d.txt", number);
        P.SaveMatrixForMatlab(str_re_H, ComplexMatrix::PrintType::ONLY_REAL);
        P.SaveMatrixForMatlab(str_im_H, ComplexMatrix::PrintType::ONLY_IMAGINARY);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1);
    std::cout << ms_int.count() << "sec.\n";

    return EXIT_SUCCESS;
}