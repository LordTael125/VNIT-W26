package main

import (
	"math/rand"
	"sort"
	"testing"
)

func TestSequentialStats(t *testing.T) {
	data := []int{3, 6, 9, 12, 1, 2, 4, 5, -3, -6}
	divisor := 3
	stats := SequentialStats(data, divisor)

	expectedCount := int64(6)
	expectedSum := int64(3 + 6 + 9 + 12 + (-3) + (-6)) // 21
	expectedMin := -6
	expectedMax := 12

	if stats.Count != expectedCount || stats.Sum != expectedSum || stats.Min != expectedMin || stats.Max != expectedMax {
		t.Errorf("SequentialStats failed: got %+v, want count=%d, sum=%d, min=%d, max=%d",
			stats, expectedCount, expectedSum, expectedMin, expectedMax)
	}
}

func TestParallelStatsEquivalence(t *testing.T) {
	sizes := []int{0, 1, 7, 100, 1000, 25000}
	workerCounts := []int{1, 2, 3, 4, 7, 8, 16, 32}
	divisor := 3

	for _, sz := range sizes {
		data := GenerateDataset(sz, 12345)
		seq := SequentialStats(data, divisor)

		for _, w := range workerCounts {
			par := ParallelStats(data, divisor, w)
			if seq != par {
				t.Fatalf("Mismatch for size %d, workers %d: seq=%+v, par=%+v", sz, w, seq, par)
			}
		}
	}
}

func TestStatsNoMatch(t *testing.T) {
	data := []int{1, 2, 4, 5, 7, 8, 10}
	divisor := 3
	seq := SequentialStats(data, divisor)
	par := ParallelStats(data, divisor, 4)

	if seq.Found || par.Found {
		t.Errorf("Expected Found=false when no numbers match divisor")
	}
	if seq != par {
		t.Errorf("Mismatch when no matches: seq=%+v, par=%+v", seq, par)
	}
}

func TestSequentialMergeSort(t *testing.T) {
	sizes := []int{0, 1, 2, 15, 100, 1000}
	for _, sz := range sizes {
		data := GenerateDataset(sz, int64(sz))
		expected := DeepCopySlice(data)
		sort.Ints(expected)

		SequentialMergeSort(data)
		if !IsSorted(data) {
			t.Fatalf("SequentialMergeSort failed to sort slice of size %d", sz)
		}
		for i := range data {
			if data[i] != expected[i] {
				t.Fatalf("SequentialMergeSort mismatch at index %d for size %d", i, sz)
			}
		}
	}
}

func TestParallelMergeSortEquivalence(t *testing.T) {
	sizes := []int{0, 1, 2, 17, 100, 1000, 50000}
	workerCounts := []int{1, 2, 3, 4, 5, 8, 16}

	for _, sz := range sizes {
		data := GenerateDataset(sz, int64(sz*31))
		expected := DeepCopySlice(data)
		sort.Ints(expected)

		for _, w := range workerCounts {
			copyData := DeepCopySlice(data)
			ParallelMergeSort(copyData, w)

			if !IsSorted(copyData) {
				t.Fatalf("ParallelMergeSort produced unsorted output for size %d, workers %d", sz, w)
			}
			for i := range copyData {
				if copyData[i] != expected[i] {
					t.Fatalf("ParallelMergeSort mismatch at index %d for size %d, workers %d", i, sz, w)
				}
			}
		}
	}
}

func TestSortCornerCases(t *testing.T) {
	// Already sorted
	sortedData := []int{-10, -5, 0, 1, 2, 3, 10, 100}
	c1 := DeepCopySlice(sortedData)
	ParallelMergeSort(c1, 4)
	if !IsSorted(c1) {
		t.Errorf("Failed on already sorted data")
	}

	// Reverse sorted
	revData := []int{100, 50, 20, 10, 0, -5, -20}
	c2 := DeepCopySlice(revData)
	ParallelMergeSort(c2, 4)
	if !IsSorted(c2) {
		t.Errorf("Failed on reverse sorted data")
	}

	// All identical elements
	identData := []int{7, 7, 7, 7, 7, 7, 7, 7, 7}
	c3 := DeepCopySlice(identData)
	ParallelMergeSort(c3, 3)
	if !IsSorted(c3) {
		t.Errorf("Failed on identical elements data")
	}
}

// Ensure race detector tests concurrent goroutines thoroughly
func TestConcurrencyRaces(t *testing.T) {
	const n = 100000
	const workers = 16
	r := rand.New(rand.NewSource(999))
	data := make([]int, n)
	for i := range data {
		data[i] = r.Intn(10000) - 5000
	}

	// Concurrently invoke parallel operations
	t.Run("ParallelStatsRaceCheck", func(t *testing.T) {
		res := ParallelStats(data, 5, workers)
		if !res.Found || res.Count == 0 {
			t.Errorf("Unexpected empty stats result")
		}
	})

	t.Run("ParallelSortRaceCheck", func(t *testing.T) {
		c := DeepCopySlice(data)
		ParallelMergeSort(c, workers)
		if !IsSorted(c) {
			t.Errorf("Parallel sort race check resulted in unsorted slice")
		}
	})
}
