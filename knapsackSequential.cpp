#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

class ContainerLoader {
private:
    std::vector<int> weights;  // Težine/volumeni dobara
    std::vector<int> values;   // Vrijednosti dobara
    int capacity;              // Kapacitet kontejnera
    int n;                     // Broj dobara

public:
    ContainerLoader(const std::vector<int>& w, const std::vector<int>& v, int cap) 
        : weights(w), values(v), capacity(cap), n(w.size()) {}

    // Klasični dinamički program (2D DP)
    int knapsackClassic() {
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));

        for (int i = 1; i <= n; i++) {
            for (int w = 0; w <= capacity; w++) {
                if (weights[i - 1] <= w) {
                    dp[i][w] = std::max(
                        dp[i - 1][w],
                        dp[i - 1][w - weights[i - 1]] + values[i - 1]
                    );
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        return dp[n][capacity];
    }

    // Optimizovana verzija (1D DP - sekvencijalno)
    int knapsackOptimized() {
        std::vector<int> dp(capacity + 1, 0);

        for (int i = 0; i < n; i++) {
            for (int w = capacity; w >= weights[i]; w--) {
                dp[w] = std::max(dp[w], dp[w - weights[i]] + values[i]);
            }
        }
        return dp[capacity];
    }
};

// Generisanje test podataka
void generateTestData(int n, std::vector<int>& weights, std::vector<int>& values, int& capacity) {
    weights.resize(n);
    values.resize(n);

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    for (int i = 0; i < n; i++) {
        weights[i] = std::rand() % 100 + 1;
        values[i] = std::rand() % 500 + 50;
    }

    capacity = n * 50;  // Razuman kapacitet
}

int main() {
    const int num_items = 1000;

    std::vector<int> weights, values;
    int capacity;

    generateTestData(num_items, weights, values, capacity);

    ContainerLoader loader(weights, values, capacity);

    std::cout << "SEKVENCIJALNA VERZIJA - POREĐENJE KNAPSACK METODA\n";
    std::cout << "Broj predmeta: " << num_items << ", Kapacitet: " << capacity << "\n\n";

    // Mjerenje vremena
    clock_t start, end;

    start = clock();
    int result_classic = loader.knapsackClassic();
    end = clock();
    double time_classic = double(end - start) / CLOCKS_PER_SEC;

    start = clock();
    int result_optimized = loader.knapsackOptimized();
    end = clock();
    double time_optimized = double(end - start) / CLOCKS_PER_SEC;

    // Rezultati
    std::cout << "Rezultati:\n";
    std::cout << "Klasični DP:     " << result_classic << " (Vrijeme: " << time_classic << " s)\n";
    std::cout << "Optimizovani DP: " << result_optimized << " (Vrijeme: " << time_optimized << " s)\n";

    // Provjera konzistentnosti
    if (result_classic == result_optimized) {
        std::cout << "\n✓ Rezultati su konzistentni!\n";
    } else {
        std::cout << "\n⚠ Upozorenje: Rezultati se razlikuju!\n";
    }

    // Uporedi performanse
    std::cout << "\nUbrzanje (Optimized vs Classic): " << time_classic / time_optimized << "x\n";

    return 0;
}
