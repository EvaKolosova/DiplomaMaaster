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
#pragma omp parallel for
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				matrix[i * rows + j] = value;
			}
		}
	}

	ComplexMatrix(const ComplexMatrix& src) : rows(src.rows), cols(src.cols) {
		matrix = new MKL_Complex16[rows * cols];
#pragma omp parallel for
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				matrix[i * rows + j] = src.matrix[i * rows + j];
			}
		}
	}

	ComplexMatrix& operator=(const ComplexMatrix& src) {
		rows = src.rows;
		cols = src.cols;

		if (matrix != nullptr) {
			this->~ComplexMatrix();
		}

		matrix = new MKL_Complex16[rows * cols];
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				matrix[i * rows + j] = src.matrix[i * rows + j];
			}
		}

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
			trace += matrix[i * rows + i].real;
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
#pragma omp parallel for
		for (size_t i = 0; i < lhs.rows; ++i) {
			for (size_t j = 0; j < lhs.cols; ++j) {
				result(i, j).real = lhs(i, j).real + rhs(i, j).real;
				result(i, j).imag = lhs(i, j).imag + rhs(i, j).imag;
			}
		}

		return result;
	}

	friend ComplexMatrix operator-(const ComplexMatrix& lhs, const ComplexMatrix& rhs) {
		if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
			throw "dsf";
		}

		MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
		ComplexMatrix result(lhs.rows, lhs.cols, zero);
#pragma omp parallel for
		for (size_t i = 0; i < lhs.rows; ++i) {
			for (size_t j = 0; j < lhs.cols; ++j) {
				result(i, j).real = lhs(i, j).real - rhs(i, j).real;
				result(i, j).imag = lhs(i, j).imag - rhs(i, j).imag;
			}
		}

		return result;
	}

	friend ComplexMatrix operator*(const ComplexMatrix& lhs, const ComplexMatrix& rhs) {
		if (lhs.cols != rhs.rows) {
			throw "dsf";
		}

		MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
		ComplexMatrix result(lhs.rows, lhs.cols, zero);
#pragma omp parallel for
		for (size_t i = 0; i < lhs.rows; ++i) {
			for (size_t j = 0; j < rhs.cols; ++j) {
				for (size_t k = 0; k < lhs.cols; ++k) {
					result(i, j).real += lhs(i, k).real * rhs(k, j).real - lhs(i, k).imag * rhs(k, j).imag;
					result(i, j).imag += lhs(i, k).real * rhs(k, j).imag + lhs(i, k).imag * rhs(k, j).real;
				}
			}
		}

		return result;
	}

	friend ComplexMatrix operator*(const ComplexMatrix& src, double value) {
		MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
		ComplexMatrix result(src.rows, src.cols, zero);
#pragma omp parallel for
		for (size_t i = 0; i < src.rows; ++i) {
			for (size_t j = 0; j < src.cols; ++j) {
				result(i, j).real = src(i, j).real * value;
				result(i, j).imag = src(i, j).imag * value;
			}
		}

		return result;
	}

	friend ComplexMatrix operator*(double value, const ComplexMatrix& src) {
		MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
		ComplexMatrix result(src.rows, src.cols, zero);
#pragma omp parallel for
		for (size_t i = 0; i < src.rows; ++i) {
			for (size_t j = 0; j < src.cols; ++j) {
				result(i, j).real = src(i, j).real * value;
				result(i, j).imag = src(i, j).imag * value;
			}
		}

		return result;
	}

	friend ComplexMatrix operator*(const ComplexMatrix& src, MKL_Complex16 value) {
		MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
		ComplexMatrix result(src.rows, src.cols, zero);
#pragma omp parallel for
		for (size_t i = 0; i < src.rows; ++i) {
			for (size_t j = 0; j < src.cols; ++j) {
				result(i, j).real = src(i, j).real * value.real - src(i, j).imag * value.imag;;
				result(i, j).imag = src(i, j).real * value.imag + src(i, j).imag * value.real;;
			}
		}

		return result;
	}

	friend ComplexMatrix operator*(MKL_Complex16 value, const ComplexMatrix& src) {
		MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
		ComplexMatrix result(src.rows, src.cols, zero);
#pragma omp parallel for
		for (size_t i = 0; i < src.rows; ++i) {
			for (size_t j = 0; j < src.cols; ++j) {
				result(i, j).real = src(i, j).real * value.real - src(i, j).imag * value.imag;;
				result(i, j).imag = src(i, j).real * value.imag + src(i, j).imag * value.real;;
			}
		}

		return result;
	}

	friend ComplexMatrix KronekerProduct(const ComplexMatrix& lhs, const ComplexMatrix& rhs) {
		MKL_Complex16 zero; zero.real = 0, zero.imag = 0;
		ComplexMatrix result(lhs.rows * rhs.rows, lhs.cols * rhs.cols, zero);
#pragma omp parallel for
		for (size_t i = 0; i < lhs.rows; ++i) {
			for (size_t j = 0; j < lhs.cols; ++j) {
				for (size_t k = 0; k < rhs.rows; ++k) {
					for (size_t l = 0; l < rhs.cols; ++l) {
						result(i * rhs.rows + l, j * rhs.cols + k).real =
							lhs(i, j).real * rhs(k, l).real - lhs(i, j).imag * rhs(k, l).imag;
						result(i * rhs.rows + l, j * rhs.cols + k).imag =
							lhs(i, j).real * rhs(k, l).imag + lhs(i, j).imag * rhs(k, l).real;
					}
				}
			}
		}

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
};

constexpr size_t N = 4;
constexpr size_t M = N * N - 1;
constexpr size_t NUMBER_OF_IMPLEMENTATIONS = 1'00;
constexpr size_t SEED = 1;

MKL_Complex16 mul(MKL_Complex16 complex, double value) {
	MKL_Complex16 result;

	result.real = complex.real * value;
	result.imag = complex.imag * value;

	return result;
}

int main() {
	auto t1 = std::chrono::high_resolution_clock::now();

	MKL_Complex16 zero; zero.real = 0, zero.imag = 0;

	std::vector<ComplexMatrix> F(N * N, ComplexMatrix(N, N, zero));
	ComplexMatrix P(N * N, N * N, zero);

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

	ComplexMatrix X(M, M, zero);
	ComplexMatrix G(M, M, zero);

	for (size_t number = 0; number < NUMBER_OF_IMPLEMENTATIONS; ++number) {
		size_t seed = NUMBER_OF_IMPLEMENTATIONS + 1 + number;
		std::mt19937 gen_y;
		gen_y.seed(seed);
		std::normal_distribution<double> distribution_y{ 0, 1 };

		seed = NUMBER_OF_IMPLEMENTATIONS * 2 + 1 + number;
		std::mt19937 gen_z;
		gen_z.seed(seed);
		std::normal_distribution<double> distribution_z{ 0, 1 };

		for (int i = 0; i < M; i++) {
			for (int j = 0; j < M; j++) {
				X(i, j).real = distribution_y(gen_y) / 2.0;
				X(i, j).imag = distribution_z(gen_z) / 2.0;
			}
		}

		MKL_Complex16 alpha;
		alpha.real = 1.0;
		alpha.imag = 0.0;

		ComplexMatrix X_conj(M, M, zero);
		mkl_zomatcopy('R', 'C', M, M, alpha, X.GetMatrix(), M, X_conj.GetMatrix(), M);

		G = X * X_conj;

		double trace = G.Trace();
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < M; ++j) {
				G(i, j).real *= N / trace;
			}
		}

		for (size_t k1 = 0; k1 < M; ++k1) {
			for (size_t k2 = 0; k2 < M; ++k2) {
				ComplexMatrix F_conj(N, N, zero);
				ComplexMatrix F_conj_trans(N, N, zero);
				ComplexMatrix F_trans(N, N, zero);

				mkl_zomatcopy('R', 'R', N, N, alpha, F[k2 + 1].GetMatrix(), N, F_conj.GetMatrix(), N);
				mkl_zomatcopy('R', 'C', N, N, alpha, F[k2 + 1].GetMatrix(), N, F_conj_trans.GetMatrix(), N);
				mkl_zomatcopy('R', 'T', N, N, alpha, (F_conj_trans * F[k1 + 1]).GetMatrix(), N, F_trans.GetMatrix(), N);

				P = P + mul(G(k1, k2), 0.5) * (
					2 * KronekerProduct(eye, F[k1 + 1]) * KronekerProduct(F_conj, eye) -
					KronekerProduct(F_trans, eye) - KronekerProduct(eye, F_conj_trans * F[k1 + 1])
					);
			}
		}

		char str_re[128] = "";
		char str_im[128] = "";
		snprintf(str_re, sizeof(str_re), "Data\\P_matrix_re_%d.txt", number);
		snprintf(str_im, sizeof(str_im), "Data\\P_matrix_im_%d.txt", number);
		P.SaveMatrixForMatlab(str_re, ComplexMatrix::PrintType::ONLY_REAL);
		P.SaveMatrixForMatlab(str_im, ComplexMatrix::PrintType::ONLY_IMAGINARY);
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	auto ms_int = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1);
	std::cout << ms_int.count() << "sec.\n";

	return EXIT_SUCCESS;
}