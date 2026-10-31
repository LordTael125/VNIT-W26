#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <time.h>

/* =========================================================================
 * Algorithm 1: Insertion Sort (Quadratic Worst-Case: O(n^2))
 * In-place, adaptive, stable comparison-based sorting.
 * ========================================================================= */
void insertion_sort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        /* Move elements of arr[0..i-1], that are greater than key,
           to one position ahead of their current position */
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

/* =========================================================================
 * Algorithm 2: Merge Sort (Log-Linear Worst-Case: O(n log n))
 * Divide-and-conquer, out-of-place, stable comparison-based sorting.
 * ========================================================================= */
static void merge(int arr[], int temp[], int left, int mid, int right) {
    int i = left;     /* Initial index of first subarray */
    int j = mid + 1;  /* Initial index of second subarray */
    int k = left;     /* Initial index of merged temp array */

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    /* Copy merged elements back to original array */
    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

static void merge_sort_recursive(int arr[], int temp[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort_recursive(arr, temp, left, mid);
        merge_sort_recursive(arr, temp, mid + 1, right);
        merge(arr, temp, left, mid, right);
    }
}

/* Wrapper for Merge Sort allocating a single scratch buffer to eliminate
   repeated per-call heap allocation noise from measurement */
void merge_sort(int arr[], int n) {
    if (n <= 1) return;
    int *temp = (int *)malloc((size_t)n * sizeof(int));
    if (!temp) {
        fprintf(stderr, "Error: Memory allocation failed for merge sort buffer.\n");
        exit(EXIT_FAILURE);
    }
    merge_sort_recursive(arr, temp, 0, n - 1);
    free(temp);
}

/* =========================================================================
 * Correctness Testing & Verification Harness
 * ========================================================================= */
static bool is_sorted(const int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return false;
    }
    return true;
}

__attribute__((unused)) static void print_array(const char *label, const int arr[], int n) {
    printf("%s: [", label);
    for (int i = 0; i < n; i++) {
        printf("%d%s", arr[i], (i == n - 1) ? "" : ", ");
    }
    printf("]\n");
}

typedef struct {
    const char *name;
    int input[16];
    int expected[16];
    int size;
} TestCase;

int main(void) {
    printf("=============================================================\n");
    printf("   CSP502 Practical 6: Algorithm Correctness Verification     \n");
    printf("=============================================================\n\n");

    TestCase tests[] = {
        {
            .name = "TC1: Empty & Single-Element Array",
            .input = {42},
            .expected = {42},
            .size = 1
        },
        {
            .name = "TC2: Already Sorted Array",
            .input = {1, 3, 5, 7, 9, 11, 13, 15},
            .expected = {1, 3, 5, 7, 9, 11, 13, 15},
            .size = 8
        },
        {
            .name = "TC3: Reverse Sorted Array",
            .input = {50, 40, 30, 20, 10, 0, -10},
            .expected = {-10, 0, 10, 20, 30, 40, 50},
            .size = 7
        },
        {
            .name = "TC4: Array with Duplicate Elements",
            .input = {5, 2, 8, 5, 2, 8, 5, 5, 1},
            .expected = {1, 2, 2, 5, 5, 5, 5, 8, 8},
            .size = 9
        },
        {
            .name = "TC5: Mixed Positive and Negative Integers",
            .input = {-12, 45, 0, -3, 89, -100, 24, 7},
            .expected = {-100, -12, -3, 0, 7, 24, 45, 89},
            .size = 8
        },
        {
            .name = "TC6: All Identical Elements",
            .input = {7, 7, 7, 7, 7, 7},
            .expected = {7, 7, 7, 7, 7, 7},
            .size = 6
        }
    };

    int num_tests = sizeof(tests) / sizeof(tests[0]);
    bool all_passed = true;

    for (int i = 0; i < num_tests; i++) {
        TestCase tc = tests[i];
        int arr_ins[16];
        int arr_merge[16];
        memcpy(arr_ins, tc.input, tc.size * sizeof(int));
        memcpy(arr_merge, tc.input, tc.size * sizeof(int));

        insertion_sort(arr_ins, tc.size);
        merge_sort(arr_merge, tc.size);

        bool ins_ok = (memcmp(arr_ins, tc.expected, tc.size * sizeof(int)) == 0) && is_sorted(arr_ins, tc.size);
        bool merge_ok = (memcmp(arr_merge, tc.expected, tc.size * sizeof(int)) == 0) && is_sorted(arr_merge, tc.size);

        printf("Test %d: %-42s | InsertionSort: %s | MergeSort: %s\n",
               i + 1, tc.name,
               ins_ok ? "PASSED" : "FAILED",
               merge_ok ? "PASSED" : "FAILED");

        if (!ins_ok || !merge_ok) {
            all_passed = false;
        }
    }

    /* Verification on a larger pseudo-random array */
    int large_n = 1000;
    int *rand_master = (int *)malloc(large_n * sizeof(int));
    int *rand_ins = (int *)malloc(large_n * sizeof(int));
    int *rand_merge = (int *)malloc(large_n * sizeof(int));

    srand(12345);
    for (int i = 0; i < large_n; i++) {
        rand_master[i] = (rand() % 20001) - 10000;
    }
    memcpy(rand_ins, rand_master, large_n * sizeof(int));
    memcpy(rand_merge, rand_master, large_n * sizeof(int));

    insertion_sort(rand_ins, large_n);
    merge_sort(rand_merge, large_n);

    bool large_ins_ok = is_sorted(rand_ins, large_n);
    bool large_merge_ok = is_sorted(rand_merge, large_n);
    bool arrays_identical = (memcmp(rand_ins, rand_merge, large_n * sizeof(int)) == 0);

    printf("Test 7: %-42s | InsertionSort: %s | MergeSort: %s\n",
           "Large Random Array (n=1000, [-10k, 10k])",
           large_ins_ok ? "PASSED" : "FAILED",
           large_merge_ok ? "PASSED" : "FAILED");
    printf("        Identical Output Verification: %s\n",
           arrays_identical ? "PASSED" : "FAILED");

    free(rand_master);
    free(rand_ins);
    free(rand_merge);

    printf("\nOverall Correctness Status: %s\n",
           (all_passed && large_ins_ok && large_merge_ok && arrays_identical) ? "ALL TESTS PASSED" : "TEST FAILURES DETECTED");

    return 0;
}
