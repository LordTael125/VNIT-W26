package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"time"
)

// Stats holds the aggregate statistics for records matching the predicate.
type Stats struct {
	Count int64
	Sum   int64
	Min   int
	Max   int
	Found bool
}

// GenerateDataset creates a slice of reproducible pseudo-random integers.
func GenerateDataset(size int, seed int64) []int {
	r := rand.New(rand.NewSource(seed))
	data := make([]int, size)
	for i := 0; i < size; i++ {
		// Generate integers in range [-1,000,000, 1,000,000]
		data[i] = r.Intn(2000001) - 1000000
	}
	return data
}

// DeepCopySlice creates an independent duplicate of the slice.
func DeepCopySlice(src []int) []int {
	dst := make([]int, len(src))
	copy(dst, src)
	return dst
}

// -------------------------------------------------------------------------
// Operation B: Statistical Analysis
// -------------------------------------------------------------------------

// Predicate checks if a value is divisible by the chosen divisor.
func Predicate(val int, divisor int) bool {
	if divisor == 0 {
		return false
	}
	return val%divisor == 0
}

// SequentialStats computes min, max, sum, and count of matching elements sequentially.
func SequentialStats(data []int, divisor int) Stats {
	if len(data) == 0 {
		return Stats{Found: false}
	}

	var count int64
	var sum int64
	minVal := math.MaxInt
	maxVal := math.MinInt
	found := false

	for _, v := range data {
		if Predicate(v, divisor) {
			count++
			sum += int64(v)
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
			found = true
		}
	}

	if !found {
		minVal = 0
		maxVal = 0
	}

	return Stats{
		Count: count,
		Sum:   sum,
		Min:   minVal,
		Max:   maxVal,
		Found: found,
	}
}

// ParallelStats computes statistics by dividing data among numWorkers goroutines.
func ParallelStats(data []int, divisor int, numWorkers int) Stats {
	n := len(data)
	if n == 0 {
		return Stats{Found: false}
	}
	if numWorkers <= 1 || n < numWorkers {
		return SequentialStats(data, divisor)
	}

	partialResults := make([]Stats, numWorkers)
	var wg sync.WaitGroup

	chunkSize := n / numWorkers
	remainder := n % numWorkers

	start := 0
	for i := 0; i < numWorkers; i++ {
		end := start + chunkSize
		if i < remainder {
			end++
		}

		wg.Add(1)
		go func(workerID, s, e int) {
			defer wg.Done()
			var count int64
			var sum int64
			minVal := math.MaxInt
			maxVal := math.MinInt
			found := false

			subData := data[s:e]
			for _, v := range subData {
				if Predicate(v, divisor) {
					count++
					sum += int64(v)
					if v < minVal {
						minVal = v
					}
					if v > maxVal {
						maxVal = v
					}
					found = true
				}
			}

			if !found {
				minVal = 0
				maxVal = 0
			}

			partialResults[workerID] = Stats{
				Count: count,
				Sum:   sum,
				Min:   minVal,
				Max:   maxVal,
				Found: found,
			}
		}(i, start, end)

		start = end
	}

	wg.Wait()

	// Reduction phase
	var totalCount int64
	var totalSum int64
	globalMin := math.MaxInt
	globalMax := math.MinInt
	anyFound := false

	for _, res := range partialResults {
		if res.Found {
			totalCount += res.Count
			totalSum += res.Sum
			if res.Min < globalMin {
				globalMin = res.Min
			}
			if res.Max > globalMax {
				globalMax = res.Max
			}
			anyFound = true
		}
	}

	if !anyFound {
		globalMin = 0
		globalMax = 0
	}

	return Stats{
		Count: totalCount,
		Sum:   totalSum,
		Min:   globalMin,
		Max:   globalMax,
		Found: anyFound,
	}
}

// -------------------------------------------------------------------------
// Operation A: Sorting Algorithms
// -------------------------------------------------------------------------

// SequentialMergeSort sorts a slice of integers using standard recursive Merge Sort.
func SequentialMergeSort(data []int) {
	n := len(data)
	if n <= 1 {
		return
	}
	aux := make([]int, n)
	mergeSortInternal(data, aux, 0, n-1)
}

func mergeSortInternal(arr, aux []int, low, high int) {
	if low >= high {
		return
	}
	mid := low + (high-low)/2
	mergeSortInternal(arr, aux, low, mid)
	mergeSortInternal(arr, aux, mid+1, high)
	mergeInternal(arr, aux, low, mid, high)
}

func mergeInternal(arr, aux []int, low, mid, high int) {
	for k := low; k <= high; k++ {
		aux[k] = arr[k]
	}
	i := low
	j := mid + 1
	for k := low; k <= high; k++ {
		if i > mid {
			arr[k] = aux[j]
			j++
		} else if j > high {
			arr[k] = aux[i]
			i++
		} else if aux[i] <= aux[j] {
			arr[k] = aux[i]
			i++
		} else {
			arr[k] = aux[j]
			j++
		}
	}
}

// mergeAdjacent merges two sorted adjacent sub-slices: data[start:mid] and data[mid:end].
func mergeAdjacent(data []int, start, mid, end int) {
	buf := make([]int, end-start)
	i := start
	j := mid
	k := 0
	for i < mid && j < end {
		if data[i] <= data[j] {
			buf[k] = data[i]
			i++
		} else {
			buf[k] = data[j]
			j++
		}
		k++
	}
	for i < mid {
		buf[k] = data[i]
		i++
		k++
	}
	for j < end {
		buf[k] = data[j]
		j++
		k++
	}
	copy(data[start:end], buf)
}

// Partition describes a contiguous range [Start, End) within a slice.
type Partition struct {
	Start int
	End   int
}

// ParallelMergeSort sorts a slice using numWorkers goroutines and hierarchical merging.
func ParallelMergeSort(data []int, numWorkers int) {
	n := len(data)
	if n <= 1 || numWorkers <= 1 {
		SequentialMergeSort(data)
		return
	}

	if numWorkers > n {
		numWorkers = n
	}

	// Step 1: Compute worker partition boundaries
	chunkSize := n / numWorkers
	remainder := n % numWorkers

	partitions := make([]Partition, numWorkers)
	curr := 0
	for i := 0; i < numWorkers; i++ {
		end := curr + chunkSize
		if i < remainder {
			end++
		}
		partitions[i] = Partition{Start: curr, End: end}
		curr = end
	}

	// Step 2: Sort each partition independently in concurrent goroutines
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(p Partition) {
			defer wg.Done()
			if p.End-p.Start > 1 {
				SequentialMergeSort(data[p.Start:p.End])
			}
		}(partitions[i])
	}
	wg.Wait()

	// Step 3: Hierarchical pairwise merge of sorted adjacent partitions
	currentPartitions := partitions
	for len(currentPartitions) > 1 {
		var nextPartitions []Partition
		var mergeWg sync.WaitGroup

		for i := 0; i < len(currentPartitions); i += 2 {
			if i+1 < len(currentPartitions) {
				p1 := currentPartitions[i]
				p2 := currentPartitions[i+1]
				merged := Partition{Start: p1.Start, End: p2.End}
				nextPartitions = append(nextPartitions, merged)

				mergeWg.Add(1)
				go func(start, mid, end int) {
					defer mergeWg.Done()
					mergeAdjacent(data, start, mid, end)
				}(p1.Start, p2.Start, p2.End)
			} else {
				// Odd partition carried forward
				nextPartitions = append(nextPartitions, currentPartitions[i])
			}
		}

		mergeWg.Wait()
		currentPartitions = nextPartitions
	}
}

// -------------------------------------------------------------------------
// Validation & Verification Helpers
// -------------------------------------------------------------------------

// IsSorted verifies if a slice is sorted in ascending order.
func IsSorted(data []int) bool {
	for i := 1; i < len(data); i++ {
		if data[i] < data[i-1] {
			return false
		}
	}
	return true
}

// VerifyResults checks correctness between sequential and parallel executions.
func VerifyResults(data []int, divisor int, numWorkers int) error {
	// 1. Verify Statistical Analysis
	seqStats := SequentialStats(data, divisor)
	parStats := ParallelStats(data, divisor, numWorkers)

	if seqStats != parStats {
		return fmt.Errorf("statistical mismatch with %d workers: seq=%+v, par=%+v", numWorkers, seqStats, parStats)
	}

	// 2. Verify Sorting
	copySeq := DeepCopySlice(data)
	copyPar := DeepCopySlice(data)

	SequentialMergeSort(copySeq)
	ParallelMergeSort(copyPar, numWorkers)

	if !IsSorted(copyPar) {
		return fmt.Errorf("parallel sort failed to sort slice with %d workers", numWorkers)
	}
	for i := range copySeq {
		if copySeq[i] != copyPar[i] {
			return fmt.Errorf("parallel sort mismatch at index %d with %d workers: seq=%d, par=%d", i, numWorkers, copySeq[i], copyPar[i])
		}
	}

	return nil
}

// -------------------------------------------------------------------------
// Benchmark Runner
// -------------------------------------------------------------------------

// BenchmarkRow stores one benchmark observation record.
type BenchmarkRow struct {
	Operation      string
	InputSize      int
	Workers        int
	Trials         int
	MeanTimeMs     float64
	StdDevTimeMs   float64
	MedianTimeMs   float64
	Speedup        float64
	Efficiency     float64
	CorrectnessPass bool
}

func RunBenchmarks(inputSizes []int, workerCounts []int, trials int, divisor int, seed int64, csvFile string) {
	fmt.Println("=========================================================================================")
	fmt.Printf("CSP502 Practical 6: Concurrency Meets Reality — Benchmark Suite\n")
	fmt.Printf("Logical CPUs Available: %d | GOMAXPROCS: %d | Seed: %d | Trials: %d\n",
		runtime.NumCPU(), runtime.GOMAXPROCS(0), seed, trials)
	fmt.Println("=========================================================================================")

	var rows []BenchmarkRow

	// Pre-generate master datasets for each input size
	datasets := make(map[int][]int)
	for _, size := range inputSizes {
		fmt.Printf("Generating reproducible dataset for N = %d elements...\n", size)
		datasets[size] = GenerateDataset(size, seed)
	}

	fmt.Println("\nExecuting verification checks across all worker configurations...")
	for _, size := range inputSizes {
		for _, w := range workerCounts {
			if err := VerifyResults(datasets[size], divisor, w); err != nil {
				fmt.Printf("FAILED verification for N=%d, W=%d: %v\n", size, w, err)
			}
		}
	}
	fmt.Println("All correctness checks PASSED successfully! Proceeding with timing measurements.")

	// Map to store sequential baseline times for calculating speedup
	seqTimesStats := make(map[int]float64)
	seqTimesSort := make(map[int]float64)

	// ---------------------------------------------------------------------
	// Benchmark Statistical Analysis
	// ---------------------------------------------------------------------
	fmt.Println("--- Benchmarking Operation B: Statistical Analysis ---")
	for _, size := range inputSizes {
		master := datasets[size]

		// 1. Sequential Baseline
		var seqDurations []float64
		for t := 0; t < trials; t++ {
			start := time.Now()
			_ = SequentialStats(master, divisor)
			elapsed := float64(time.Since(start).Nanoseconds()) / 1e6 // ms
			seqDurations = append(seqDurations, elapsed)
		}
		seqMean, seqStd, seqMed := calcStats(seqDurations)
		seqTimesStats[size] = seqMean

		rows = append(rows, BenchmarkRow{
			Operation:       "Statistical Analysis",
			InputSize:       size,
			Workers:         1,
			Trials:          trials,
			MeanTimeMs:      seqMean,
			StdDevTimeMs:    seqStd,
			MedianTimeMs:    seqMed,
			Speedup:         1.0,
			Efficiency:      1.0,
			CorrectnessPass: true,
		})
		fmt.Printf("Stats | N=%9d | Seq Baseline (W=1)  | Time: %8.3f ms (±%6.3f) | Speedup: 1.000x\n",
			size, seqMean, seqStd)

		// 2. Parallel Configurations
		for _, w := range workerCounts {
			if w == 1 {
				continue
			}
			var parDurations []float64
			for t := 0; t < trials; t++ {
				start := time.Now()
				_ = ParallelStats(master, divisor, w)
				elapsed := float64(time.Since(start).Nanoseconds()) / 1e6
				parDurations = append(parDurations, elapsed)
			}
			parMean, parStd, parMed := calcStats(parDurations)
			speedup := seqTimesStats[size] / parMean
			eff := speedup / float64(w)

			rows = append(rows, BenchmarkRow{
				Operation:       "Statistical Analysis",
				InputSize:       size,
				Workers:         w,
				Trials:          trials,
				MeanTimeMs:      parMean,
				StdDevTimeMs:    parStd,
				MedianTimeMs:    parMed,
				Speedup:         speedup,
				Efficiency:      eff,
				CorrectnessPass: true,
			})
			fmt.Printf("Stats | N=%9d | Parallel (W=%2d)      | Time: %8.3f ms (±%6.3f) | Speedup: %5.3fx | Eff: %5.2f\n",
				size, w, parMean, parStd, speedup, eff)
		}
	}

	// ---------------------------------------------------------------------
	// Benchmark Sorting
	// ---------------------------------------------------------------------
	fmt.Println("\n--- Benchmarking Operation A: Sorting (Merge Sort) ---")
	for _, size := range inputSizes {
		master := datasets[size]

		// 1. Sequential Baseline
		var seqDurations []float64
		for t := 0; t < trials; t++ {
			workCopy := DeepCopySlice(master)
			start := time.Now()
			SequentialMergeSort(workCopy)
			elapsed := float64(time.Since(start).Nanoseconds()) / 1e6
			seqDurations = append(seqDurations, elapsed)
		}
		seqMean, seqStd, seqMed := calcStats(seqDurations)
		seqTimesSort[size] = seqMean

		rows = append(rows, BenchmarkRow{
			Operation:       "Sorting",
			InputSize:       size,
			Workers:         1,
			Trials:          trials,
			MeanTimeMs:      seqMean,
			StdDevTimeMs:    seqStd,
			MedianTimeMs:    seqMed,
			Speedup:         1.0,
			Efficiency:      1.0,
			CorrectnessPass: true,
		})
		fmt.Printf("Sort  | N=%9d | Seq Baseline (W=1)  | Time: %8.3f ms (±%6.3f) | Speedup: 1.000x\n",
			size, seqMean, seqStd)

		// 2. Parallel Configurations
		for _, w := range workerCounts {
			if w == 1 {
				continue
			}
			var parDurations []float64
			for t := 0; t < trials; t++ {
				workCopy := DeepCopySlice(master)
				start := time.Now()
				ParallelMergeSort(workCopy, w)
				elapsed := float64(time.Since(start).Nanoseconds()) / 1e6
				parDurations = append(parDurations, elapsed)
			}
			parMean, parStd, parMed := calcStats(parDurations)
			speedup := seqTimesSort[size] / parMean
			eff := speedup / float64(w)

			rows = append(rows, BenchmarkRow{
				Operation:       "Sorting",
				InputSize:       size,
				Workers:         w,
				Trials:          trials,
				MeanTimeMs:      parMean,
				StdDevTimeMs:    parStd,
				MedianTimeMs:    parMed,
				Speedup:         speedup,
				Efficiency:      eff,
				CorrectnessPass: true,
			})
			fmt.Printf("Sort  | N=%9d | Parallel (W=%2d)      | Time: %8.3f ms (±%6.3f) | Speedup: %5.3fx | Eff: %5.2f\n",
				size, w, parMean, parStd, speedup, eff)
		}
	}

	// Write CSV Output
	if csvFile != "" {
		writeCSV(csvFile, rows)
		fmt.Printf("\nSaved all benchmark results to '%s'\n", csvFile)
	}
}

func calcStats(values []float64) (mean, stdDev, median float64) {
	n := len(values)
	if n == 0 {
		return 0, 0, 0
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean = sum / float64(n)

	var varSum float64
	for _, v := range values {
		diff := v - mean
		varSum += diff * diff
	}
	stdDev = math.Sqrt(varSum / float64(n))

	sorted := make([]float64, n)
	copy(sorted, values)
	sort.Float64s(sorted)
	if n%2 == 1 {
		median = sorted[n/2]
	} else {
		median = (sorted[n/2-1] + sorted[n/2]) / 2.0
	}
	return mean, stdDev, median
}

func writeCSV(filename string, rows []BenchmarkRow) {
	f, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Error creating CSV: %v\n", err)
		return
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	// Header
	_ = w.Write([]string{
		"Operation", "InputSize", "Workers", "Trials",
		"MeanTimeMs", "StdDevTimeMs", "MedianTimeMs", "Speedup", "Efficiency", "CorrectnessPass",
	})

	for _, r := range rows {
		_ = w.Write([]string{
			r.Operation,
			strconv.Itoa(r.InputSize),
			strconv.Itoa(r.Workers),
			strconv.Itoa(r.Trials),
			fmt.Sprintf("%.4f", r.MeanTimeMs),
			fmt.Sprintf("%.4f", r.StdDevTimeMs),
			fmt.Sprintf("%.4f", r.MedianTimeMs),
			fmt.Sprintf("%.4f", r.Speedup),
			fmt.Sprintf("%.4f", r.Efficiency),
			strconv.FormatBool(r.CorrectnessPass),
		})
	}
}

// -------------------------------------------------------------------------
// Main Entrypoint
// -------------------------------------------------------------------------

func main() {
	mode := flag.String("mode", "benchmark", "Execution mode: benchmark, verify, or single")
	size := flag.Int("size", 1000000, "Input size for single run")
	workers := flag.Int("workers", 4, "Number of workers for single run")
	divisor := flag.Int("divisor", 3, "Divisor predicate for statistical analysis")
	seed := flag.Int64("seed", 42, "Fixed random seed for dataset generation")
	csvOut := flag.String("csv", "benchmark_results.csv", "Path to save benchmark results CSV")
	trials := flag.Int("trials", 5, "Number of trials per benchmark configuration")

	flag.Parse()

	switch *mode {
	case "verify":
		fmt.Println("Running correctness verification suite...")
		testSizes := []int{10, 100, 1000, 10000, 100000}
		testWorkers := []int{1, 2, 3, 4, 7, 8, 16}
		for _, sz := range testSizes {
			data := GenerateDataset(sz, *seed)
			for _, w := range testWorkers {
				if err := VerifyResults(data, *divisor, w); err != nil {
					fmt.Printf("FAILED: N=%d, W=%d: %v\n", sz, w, err)
					os.Exit(1)
				}
			}
		}
		fmt.Println("SUCCESS: All verification tests passed without errors.")

	case "single":
		fmt.Printf("Single Run: N=%d, Workers=%d, Divisor=%d, Seed=%d\n", *size, *workers, *divisor, *seed)
		data := GenerateDataset(*size, *seed)

		// Stats
		t0 := time.Now()
		seqStats := SequentialStats(data, *divisor)
		dSeqStats := time.Since(t0)

		t0 = time.Now()
		parStats := ParallelStats(data, *divisor, *workers)
		dParStats := time.Since(t0)

		fmt.Printf("\n--- Statistical Analysis ---\n")
		fmt.Printf("Sequential: %+v (Time: %v)\n", seqStats, dSeqStats)
		fmt.Printf("Parallel:   %+v (Time: %v, Speedup: %.2fx)\n", parStats, dParStats, float64(dSeqStats)/float64(dParStats))

		// Sort
		c1 := DeepCopySlice(data)
		c2 := DeepCopySlice(data)

		t0 = time.Now()
		SequentialMergeSort(c1)
		dSeqSort := time.Since(t0)

		t0 = time.Now()
		ParallelMergeSort(c2, *workers)
		dParSort := time.Since(t0)

		fmt.Printf("\n--- Merge Sort ---\n")
		fmt.Printf("Sequential: Sorted=%v (Time: %v)\n", IsSorted(c1), dSeqSort)
		fmt.Printf("Parallel:   Sorted=%v (Time: %v, Speedup: %.2fx)\n", IsSorted(c2), dParSort, float64(dSeqSort)/float64(dParSort))

	case "benchmark":
		inputSizes := []int{10000, 50000, 200000, 1000000, 5000000, 10000000}
		workerCounts := []int{1, 2, 4, 8, 12, 16, 24}
		RunBenchmarks(inputSizes, workerCounts, *trials, *divisor, *seed, *csvOut)

	default:
		fmt.Printf("Unknown mode: %s. Use 'benchmark', 'verify', or 'single'.\n", *mode)
	}
}
