//   g++ -O3 -march=native -mavx2 -mfma -fopenmp -std=c++17 knapsackOptimized.cpp -o knap

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cstring>     // memcpy, memset
#include <immintrin.h> // AVX2 intrinsics
#include <omp.h>       // OpenMP
#include <fstream>     // ofstream za CSV
#include <cmath>       // std::max

//  ContainerLoader klasa

class ContainerLoader
{
private:
    std::vector<int> weights; // tezine
    std::vector<int> values;  // vrijednosti
    int capacity;             // kapacitet
    int n;                    // br predmeta

public:
    ContainerLoader(const std::vector<int> &w,
                    const std::vector<int> &v,
                    int cap)
        : weights(w), values(v), capacity(cap), n((int)w.size()) {}

    // 1) Klasični 2D DP (za provjeru rezultata)
    int knapsackClassic2D() const
    {
        // dp[i][w] = max vrijednost koristeći prvih i predmeta
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));

        for (int i = 1; i <= n; ++i)
        {
            int wi = weights[i - 1];
            int vi = values[i - 1];

            for (int w = 0; w <= capacity; ++w)
            {
                if (wi <= w)
                {
                    int take = dp[i - 1][w - wi] + vi;
                    int skip = dp[i - 1][w];
                    dp[i][w] = std::max(skip, take);
                }
                else
                {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        return dp[n][capacity];
    }

    // 2) Originalni 1D DP (sekvencijalno, ali bez SIMD-a) - REFERENTNA METODA
    int knapsack1D_baseline() const
    {
        std::vector<int> dp(capacity + 1, 0);

        for (int i = 0; i < n; ++i)
        {
            int wi = weights[i];
            int vi = values[i];

            // UNAZAD po w da ne pregazimo dp[w - wi]
            for (int w = capacity; w >= wi; --w)
            {
                int cand = dp[w - wi] + vi;
                if (cand > dp[w])
                {
                    dp[w] = cand;
                }
            }
        }
        return dp[capacity];
    }

    // 3) SIMD 1D DP (AVX2, streaming, single-thread)
    int knapsackSIMD_stream() const
    {
        const int N = capacity + 8; // mali padding

        // C-nizovi umjesto std::vector
        int *prev = new int[N];
        int *dp = new int[N];

        // inicijalno sve 0
        std::memset(prev, 0, sizeof(int) * N);
        std::memset(dp, 0, sizeof(int) * N);

        for (int i = 0; i < n; ++i)
        {
            const int wi = weights[i];
            const int vi = values[i];

            // jeftin swap pointera umjesto std::swap()
            int *tmp = prev;
            prev = dp;
            dp = tmp;

            // ispod wi: predmet ne može stati -> kopiraj staro stanje (linearni memcpy)
            if (wi > 0)
            {
                const int count = std::min(wi, capacity + 1);
                std::memcpy(dp, prev, sizeof(int) * count);
            }

            __m256i add_vi = _mm256_set1_epi32(vi);

            int w = wi;
            const int Wlim = capacity - 7; // posljednji blok koji staje u [w..w+7]

            for (; w <= Wlim; w += 8)
            {
                // oba loada su kontigventna => cache-friendly
                __m256i prev_curr =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&prev[w]));
                __m256i prev_shift =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&prev[w - wi]));

                __m256i cand = _mm256_add_epi32(prev_shift, add_vi);
                __m256i res = _mm256_max_epi32(prev_curr, cand);

                _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dp[w]), res);
            }

            // "rep" (tail) skalarno do kraja capacity-ja
            for (; w <= capacity; ++w)
            {
                int take = prev[w - wi] + vi;
                int skip = prev[w];
                dp[w] = (take > skip) ? take : skip;
            }
        }

        int result = dp[capacity];

        delete[] prev;
        delete[] dp;

        return result;
    }

    // 4) SIMD 1D DP + OpenMP (AVX2 + paralelni for)
    int knapsackSIMD_stream_omp() const
    {
        const int N = capacity + 8;

        int *prev = new int[N];
        int *dp = new int[N];

        std::memset(prev, 0, sizeof(int) * N);
        std::memset(dp, 0, sizeof(int) * N);

        for (int i = 0; i < n; ++i)
        {
            const int wi = weights[i];
            const int vi = values[i];

            // swap pointera (O(1))
            int *tmp = prev;
            prev = dp;
            dp = tmp;

            // prefiks: w < wi -> predmet ne može stati, kopiramo staro stanje
            if (wi > 0)
            {
                const int count = std::min(wi, capacity + 1);
                std::memcpy(dp, prev, sizeof(int) * count);
            }

            __m256i add_vi = _mm256_set1_epi32(vi);

            const int Wlim = capacity - 7;
            const int w_start = wi;

// Paralelizovana AVX2 petlja
#pragma omp parallel
            {
                int *__restrict loc_prev = prev;
                int *__restrict loc_dp = dp;
                __m256i loc_add_vi = add_vi;

#pragma omp for schedule(static)
                for (int w = w_start; w <= Wlim; w += 8)
                {
                    __m256i prev_curr =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&loc_prev[w]));
                    __m256i prev_shift =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&loc_prev[w - wi]));

                    __m256i cand = _mm256_add_epi32(prev_shift, loc_add_vi);
                    __m256i res = _mm256_max_epi32(prev_curr, cand);

                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&loc_dp[w]), res);
                }
            }

            // rep (tail) skalarno – OpenMP overhead se ne isplati za malo elemenata
            int w_tail = std::max(w_start, Wlim + 1);
            for (int w = w_tail; w <= capacity; ++w)
            {
                int take = prev[w - wi] + vi;
                int skip = prev[w];
                dp[w] = (take > skip) ? take : skip;
            }
        }

        int result = dp[capacity];

        delete[] prev;
        delete[] dp;

        return result;
    }
};

//  Generisanje test podataka

void generateTestData(int n,
                      std::vector<int> &weights,
                      std::vector<int> &values,
                      int &capacity)
{
    weights.resize(n);
    values.resize(n);

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    for (int i = 0; i < n; ++i)
    {
        weights[i] = std::rand() % 100 + 1; // 1..100
        values[i] = std::rand() % 500 + 50; // 50..549
    }
    capacity = n * 50; // dosta velik kapacitet za stres-test
}

//  Helper za mjerenje vremena

template <typename Func>
double measure_seconds(Func f, int &result_out)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    result_out = f();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    return diff.count();
}

int main()
{
    const int num_items = 2000;

    std::vector<int> weights, values;
    int capacity;
    generateTestData(num_items, weights, values, capacity);

    ContainerLoader loader(weights, values, capacity);

    std::cout << "Poredjenje knapsack metoda (n = " << num_items
              << ", capacity = " << capacity << ")\n";
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";

    int r_classic2D = 0, r_1d_base = 0;
    int r_simd_stream = 0, r_simd_stream_omp = 0;

    double t_classic2D, t_1d_base;
    double t_simd_stream, t_simd_stream_omp;

    // Mjerenje vremena
    t_classic2D = measure_seconds([&]()
                                  { return loader.knapsackClassic2D(); }, r_classic2D);
    t_1d_base = measure_seconds([&]()
                                { return loader.knapsack1D_baseline(); }, r_1d_base); // REFERENTNA
    t_simd_stream = measure_seconds([&]()
                                    { return loader.knapsackSIMD_stream(); }, r_simd_stream);
    t_simd_stream_omp = measure_seconds([&]()
                                        { return loader.knapsackSIMD_stream_omp(); }, r_simd_stream_omp);

    std::cout << "Rezultati:\n";
    std::cout << "Classic 2D:            " << r_classic2D << "  (t = " << t_classic2D << " s)\n";
    std::cout << "1D baseline (REF):     " << r_1d_base << "  (t = " << t_1d_base << " s)\n";
    std::cout << "SIMD stream:           " << r_simd_stream << "  (t = " << t_simd_stream << " s)\n";
    std::cout << "SIMD stream + OpenMP:  " << r_simd_stream_omp << "  (t = " << t_simd_stream_omp << " s)\n";

    // Provjera konzistentnosti – svi moraju dati isti optimum
    bool ok = (r_classic2D == r_1d_base) &&
              (r_1d_base == r_simd_stream) &&
              (r_simd_stream == r_simd_stream_omp);

    if (ok)
    {
        std::cout << "\nSvi rezultati su konzistentni.\n";
    }
    else
    {
        std::cout << "\nUpozorenje: Rezultati se razlikuju!\n";
    }

    // Apsolutna vremena
    std::cout << "\n====================================================\n";
    std::cout << "APSOLUTNA VREMENA (sekunde)\n";
    std::cout << "====================================================\n";
    std::cout << "Classic 2D:            " << t_classic2D << " s\n";
    std::cout << "1D baseline (REF):     " << t_1d_base << " s\n";
    std::cout << "SIMD stream:           " << t_simd_stream << " s\n";
    std::cout << "SIMD stream + OpenMP:  " << t_simd_stream_omp << " s\n";

    // UBRZANJA U ODNOSU NA REFERENTNI knapsack1D_baseline()
    std::cout << "\n====================================================\n";
    std::cout << "UBRZANJA U ODNOSU NA knapsack1D_baseline (REFERENTNA)\n";
    std::cout << "====================================================\n";

    // Classic 2D se računa kao usporenje/ubrzanje u odnosu na baseline, ali je uvršten samo za kompletnu tablicu.
    double s_classic2D = t_1d_base / t_classic2D; 
    double s_simd_stream = t_1d_base / t_simd_stream;
    double s_simd_stream_omp = t_1d_base / t_simd_stream_omp;

    std::cout << "SIMD stream:           " << s_simd_stream << "x\n";
    std::cout << "SIMD stream + OpenMP:  " << s_simd_stream_omp << "x\n";

    std::cout << "\n====================================================\n";

    //  Zapis u CSV fajl

    {
        std::ofstream csv("results.csv");
        if (!csv)
        {
            std::cerr << "Greska: ne mogu otvoriti results.csv za pisanje!\n";
        }
        else
        {
            csv << "Method,TimeSeconds,SpeedupVs1DBaseline,Result\n";
            csv << "Classic2D," << t_classic2D << "," << s_classic2D << "," << r_classic2D << "\n";
            csv << "1D_baseline," << t_1d_base << "," << 1.0 << "," << r_1d_base << "\n"; // REFERENTNA
            csv << "SIMD_stream," << t_simd_stream << "," << s_simd_stream << "," << r_simd_stream << "\n";
            csv << "SIMD_stream_OpenMP," << t_simd_stream_omp << "," << s_simd_stream_omp << "," << r_simd_stream_omp << "\n";
        }
    }

    return 0;
}