#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <iomanip>
#include <tuple>
#include "mt19937ar.h"

double round(double value, int precision) {
    const int adjustment = std::pow(10, precision);
    return std::floor(value * adjustment + 0.5) / adjustment;
}

double rand_2_rand() {
    auto sum = 0.0;

    for (size_t i = 0; i < 12; ++i) {
        sum += genrand_real1();
    }

    return sum - 6.0;
}

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
        } else {
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


int main() {
//    std::random_device rd{};
//    std::mt19937 gen{rd()};
//
//    // values near the mean are the most likely
//    // standard deviation affects the dispersion of generated values from the mean
//    std::normal_distribution<> d{5,2};
//
//    std::map<int, int> hist{};
//    for (int n=0; n<10000; ++n) {
//        ++hist[std::round(d(gen))];
//    }
//    for (auto p : hist) {
//        std::cout << std::setw(2)
//                  << p.first << ' ' << std::string(p.second/200, '*') << '\n';
//    }

    RandomGeneratorMT19937ar random(0, 1);

//    init_genrand(0);
    std::map<double, int> hist {};
    for (int n = 0; n < 1000000; ++n) {
//        auto val = rand_2_rand();
        auto val = random.Generate();
        ++hist[round(val, 2)];
    }

    for (auto p : hist) {
        std::cout << std::setw(2)
                  << p.first << ' ' << std::string(p.second/200, '*') << '\n';
    }

    return EXIT_SUCCESS;
}
