#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

/* =========================================================================
 * Algorithms
 * ========================================================================= */
void insertion_sort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i], j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

static void merge(int arr[], int temp[], int l, int m, int r) {
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    for (i = l; i <= r; i++) arr[i] = temp[i];
}

static void merge_sort_rec(int arr[], int temp[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        merge_sort_rec(arr, temp, l, m);
        merge_sort_rec(arr, temp, m + 1, r);
        merge(arr, temp, l, m, r);
    }
}

void merge_sort(int arr[], int n) {
    if (n <= 1) return;
    int *temp = (int *)malloc((size_t)n * sizeof(int));
    if (!temp) { perror("malloc failed"); exit(EXIT_FAILURE); }
    merge_sort_rec(arr, temp, 0, n - 1);
    free(temp);
}

/* =========================================================================
 * High-Resolution Monotonic Timing Helper (in Microseconds)
 * ========================================================================= */
static inline double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000000.0 + (double)ts.tv_nsec / 1000.0;
}

/* =========================================================================
 * Data Generators
 * ========================================================================= */
typedef enum {
    DIST_RANDOM,
    DIST_SORTED,
    DIST_REVERSE,
    DIST_NEARLY_SORTED,
    DIST_REPEATED
} Distribution;

const char *dist_names[] = {
    "Random",
    "Sorted",
    "Reverse",
    "Nearly Sorted",
    "Repeated Values"
};

void generate_data(int arr[], int n, Distribution dist) {
    switch (dist) {
        case DIST_RANDOM:
            for (int i = 0; i < n; i++) arr[i] = (rand() % 2000000) - 1000000;
            break;
        case DIST_SORTED:
            for (int i = 0; i < n; i++) arr[i] = i * 2;
            break;
        case DIST_REVERSE:
            for (int i = 0; i < n; i++) arr[i] = (n - i) * 2;
            break;
        case DIST_NEARLY_SORTED:
            for (int i = 0; i < n; i++) arr[i] = i * 2;
            /* Perturb ~5% of elements */
            int swaps = (n * 5) / 100;
            if (swaps < 1 && n > 1) swaps = 1;
            for (int s = 0; s < swaps; s++) {
                int idx1 = rand() % n;
                int idx2 = rand() % n;
                int tmp = arr[idx1];
                arr[idx1] = arr[idx2];
                arr[idx2] = tmp;
            }
            break;
        case DIST_REPEATED:
            for (int i = 0; i < n; i++) arr[i] = (rand() % 7) * 10; /* only 7 distinct keys */
            break;
    }
}

int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(void) {
    srand(42);
    int sizes[] = {10, 20, 50, 100, 200, 500, 1000, 2500, 5000, 10000, 25000, 50000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int reps = 5;

    FILE *fout = fopen("benchmark_results.csv", "w");
    if (!fout) { perror("Failed to open benchmark_results.csv"); return 1; }
    fprintf(fout, "Distribution,InputSize,Algorithm,Rep1,Rep2,Rep3,Rep4,Rep5,Median_us,Mean_us\n");

    printf("Starting CSP502 Benchmarking across 5 Distributions...\n");

    for (int d = 0; d < 5; d++) {
        Distribution dist = (Distribution)d;
        printf("--- Benchmarking Distribution: %s ---\n", dist_names[d]);

        for (int s = 0; s < num_sizes; s++) {
            int n = sizes[s];

            /* Master array */
            int *master = (int *)malloc((size_t)n * sizeof(int));
            int *work = (int *)malloc((size_t)n * sizeof(int));

            generate_data(master, n, dist);

            /* Benchmark Insertion Sort (skip n > 25000 for reverse/random if too slow, but n=50000 takes ~1-2s in C -O3) */
            double times_ins[5];
            bool skip_ins = (n > 50000);

            if (!skip_ins) {
                for (int r = 0; r < reps; r++) {
                    memcpy(work, master, (size_t)n * sizeof(int));
                    double t0 = get_time_us();
                    insertion_sort(work, n);
                    double t1 = get_time_us();
                    times_ins[r] = t1 - t0;
                }
                double sorted_times[5];
                memcpy(sorted_times, times_ins, 5 * sizeof(double));
                qsort(sorted_times, 5, sizeof(double), compare_doubles);
                double med_ins = sorted_times[2];
                double sum_ins = 0;
                for (int r = 0; r < reps; r++) sum_ins += times_ins[r];
                double mean_ins = sum_ins / reps;

                fprintf(fout, "%s,%d,InsertionSort,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                        dist_names[d], n, times_ins[0], times_ins[1], times_ins[2], times_ins[3], times_ins[4],
                        med_ins, mean_ins);
            }

            /* Benchmark Merge Sort */
            double times_merge[5];
            for (int r = 0; r < reps; r++) {
                memcpy(work, master, (size_t)n * sizeof(int));
                double t0 = get_time_us();
                merge_sort(work, n);
                double t1 = get_time_us();
                times_merge[r] = t1 - t0;
            }
            double sorted_merge[5];
            memcpy(sorted_merge, times_merge, 5 * sizeof(double));
            qsort(sorted_merge, 5, sizeof(double), compare_doubles);
            double med_merge = sorted_merge[2];
            double sum_merge = 0;
            for (int r = 0; r < reps; r++) sum_merge += times_merge[r];
            double mean_merge = sum_merge / reps;

            fprintf(fout, "%s,%d,MergeSort,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    dist_names[d], n, times_merge[0], times_merge[1], times_merge[2], times_merge[3], times_merge[4],
                    med_merge, mean_merge);

            free(master);
            free(work);
        }
    }

    fclose(fout);
    printf("Benchmarking completed successfully. Results saved to benchmark_results.csv\n");
    return 0;
}
