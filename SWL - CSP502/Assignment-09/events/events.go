package events

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
)

// Priority classifies event scheduling priority.
type Priority int

const (
	PriorityNormal Priority = 0
	PriorityHigh   Priority = 1
)

func (p Priority) String() string {
	if p == PriorityHigh {
		return "HIGH"
	}
	return "NORMAL"
}

// EventClass represents the three functional processing classes from Lab 9.
type EventClass string

const (
	ClassLoginFailed EventClass = "LOGIN_FAILED"
	ClassFileAccess  EventClass = "FILE_ACCESS"
	ClassOther       EventClass = "OTHER"
)

// Record preserves the six original Lab 4A fields plus unique EventID and validation metadata.
type Record struct {
	EventID   uint64     `json:"event_id"`
	Date      string     `json:"date"`
	TimeStr   string     `json:"time"`
	EventType string     `json:"event_type"`
	User      string     `json:"user"`
	IP        string     `json:"ip"`
	Resource  string     `json:"resource"`
	Valid     bool       `json:"valid"`
	Class     EventClass `json:"class"`
	Priority  Priority   `json:"priority"`
}

// DatasetStats summarizes dataset validation output for Part B.
type DatasetStats struct {
	Name          string         `json:"name"`
	Path          string         `json:"path"`
	Seed          int64          `json:"seed"`
	SHA256        string         `json:"sha256"`
	TotalLines    int            `json:"total_lines"`
	ValidCount    int            `json:"valid_count"`
	Malformed     int            `json:"malformed"`
	DuplicateRows int            `json:"duplicate_rows"`
	UniqueUsers   int            `json:"unique_users"`
	UniqueIPs     int            `json:"unique_ips"`
	EventCounts   map[string]int `json:"event_counts"`
}

// ParseAndValidate reads a Lab 4A log file, assigning each line a unique EventID (1..N).
func ParseAndValidate(name, path string, seed int64) ([]Record, DatasetStats, error) {
	rawBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, DatasetStats{}, err
	}
	sum := sha256.Sum256(rawBytes)

	stats := DatasetStats{
		Name:        name,
		Path:        path,
		Seed:        seed,
		SHA256:      hex.EncodeToString(sum[:]),
		EventCounts: make(map[string]int),
	}

	seenLines := make(map[string]struct{})
	uniqueUsers := make(map[string]struct{})
	uniqueIPs := make(map[string]struct{})
	var records []Record

	scanner := bufio.NewScanner(strings.NewReader(string(rawBytes)))
	var id uint64
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		id++
		stats.TotalLines++
		if _, dup := seenLines[line]; dup {
			stats.DuplicateRows++
		} else {
			seenLines[line] = struct{}{}
		}

		fields := strings.Fields(line)
		if len(fields) != 6 {
			stats.Malformed++
			records = append(records, Record{
				EventID: id,
				Valid:   false,
			})
			continue
		}

		rec := Record{
			EventID:   id,
			Date:      fields[0],
			TimeStr:   fields[1],
			EventType: fields[2],
			User:      fields[3],
			IP:        fields[4],
			Resource:  fields[5],
			Valid:     true,
		}
		switch rec.EventType {
		case "LOGIN_FAILED":
			rec.Class = ClassLoginFailed
			rec.Priority = PriorityHigh
		case "FILE_ACCESS":
			rec.Class = ClassFileAccess
			rec.Priority = PriorityNormal
		default:
			rec.Class = ClassOther
			rec.Priority = PriorityNormal
		}

		stats.ValidCount++
		stats.EventCounts[rec.EventType]++
		uniqueUsers[rec.User] = struct{}{}
		uniqueIPs[rec.IP] = struct{}{}
		records = append(records, rec)
	}
	if err := scanner.Err(); err != nil {
		return nil, stats, err
	}
	stats.UniqueUsers = len(uniqueUsers)
	stats.UniqueIPs = len(uniqueIPs)
	return records, stats, nil
}

// ReplayItem pairs an event record with deterministic arrival offset and disturbance flags.
type ReplayItem struct {
	Record           Record        `json:"record"`
	ArrivalOffsetNs  int64         `json:"arrival_offset_ns"`
	BaseCostNs       int64         `json:"base_cost_ns"`
	IsSlow           bool          `json:"is_slow"`
	FailFirstAttempt bool          `json:"fail_first_attempt"`
}

// TraceSpec configures deterministic replay trace generation for D0-D4 and paced runs.
type TraceSpec struct {
	Scenario         string  `json:"scenario"`
	Seed             int64   `json:"seed"`
	NumEvents        int     `json:"num_events"`
	NominalRateRPS   float64 `json:"nominal_rate_rps"`
	BaseCostUs       int64   `json:"base_cost_us"`
	BurstMiddle3x    bool    `json:"burst_middle_3x"`
	SlowRatio        float64 `json:"slow_ratio"`
	SlowMultiplier   int64   `json:"slow_multiplier"`
	TransientFailRat float64 `json:"transient_fail_ratio"`
	HighPriorityFrac float64 `json:"high_priority_fraction"` // >0 forces target fraction of LOGIN_FAILED
}

// BuildReplayTrace constructs an identical deterministic trace for competing designs.
func BuildReplayTrace(baseRecords []Record, spec TraceSpec) []ReplayItem {
	rng := rand.New(rand.NewSource(spec.Seed))

	// Collect valid pool and high/normal pools for D4 priority-pressure traces.
	var validPool, highPool, normPool []Record
	for _, r := range baseRecords {
		if r.Valid {
			validPool = append(validPool, r)
			if r.Priority == PriorityHigh {
				highPool = append(highPool, r)
			} else {
				normPool = append(normPool, r)
			}
		}
	}

	n := spec.NumEvents
	if n <= 0 || n > len(validPool) {
		n = len(validPool)
	}

	items := make([]ReplayItem, 0, n)
	var elapsedNs int64
	nominalGapNs := float64(time.Second) / spec.NominalRateRPS
	baseCostNs := spec.BaseCostUs * 1000
	if baseCostNs <= 0 {
		baseCostNs = int64(300 * time.Microsecond)
	}
	mult := spec.SlowMultiplier
	if mult <= 1 {
		mult = 5
	}

	for i := 0; i < n; i++ {
		var rec Record
		if spec.HighPriorityFrac > 0 && len(highPool) > 0 && len(normPool) > 0 {
			if rng.Float64() < spec.HighPriorityFrac {
				rec = highPool[i%len(highPool)]
			} else {
				rec = normPool[i%len(normPool)]
			}
		} else {
			rec = validPool[i%len(validPool)]
		}
		rec.EventID = uint64(i + 1) // unique sequential EventID independent of contents

		gap := nominalGapNs
		if spec.BurstMiddle3x && i >= n/3 && i < (2*n)/3 {
			gap = nominalGapNs / 3.0
		}
		// Small deterministic jitter in [0.9x, 1.1x]
		gap *= (0.9 + 0.2*rng.Float64())
		if i > 0 {
			elapsedNs += int64(gap)
		}

		// Deterministic event-ID-based disturbance injection
		isSlow := spec.SlowRatio > 0 && rng.Float64() < spec.SlowRatio
		failFirst := spec.TransientFailRat > 0 && rng.Float64() < spec.TransientFailRat
		cost := baseCostNs + int64(rng.Int63n(baseCostNs/2+1))
		if isSlow {
			cost *= mult
		}

		items = append(items, ReplayItem{
			Record:           rec,
			ArrivalOffsetNs:  elapsedNs,
			BaseCostNs:       cost,
			IsSlow:           isSlow,
			FailFirstAttempt: failFirst,
		})
	}
	return items
}

// SaveReplayTrace writes a replay trace to JSON for auditability & exact reuse.
func SaveReplayTrace(path string, items []ReplayItem) error {
	b, err := json.MarshalIndent(items, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0644)
}

// SortedKeys returns sorted keys of a string->int map for deterministic comparison.
func SortedKeys(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// FormatStats returns a concise human-readable dataset validation summary.
func FormatStats(s DatasetStats) string {
	return fmt.Sprintf(
		"%s: lines=%d valid=%d malformed=%d duplicates=%d users=%d ips=%d sha256=%s...",
		s.Name, s.TotalLines, s.ValidCount, s.Malformed, s.DuplicateRows, s.UniqueUsers, s.UniqueIPs, s.SHA256[:12],
	)
}
