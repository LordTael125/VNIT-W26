import sys
import heapq
import bisect
from collections import defaultdict
from datetime import datetime

class LogRecord:
    """A simple class to hold parsed log data for sequential sorting and searching."""
    __slots__ = ['timestamp', 'event_type', 'user', 'ip', 'resource', 'raw_line']

    def __init__(self, timestamp, event_type, user, ip, resource, raw_line):
        self.timestamp = timestamp
        self.event_type = event_type
        self.user = user
        self.ip = ip
        self.resource = resource
        self.raw_line = raw_line

    # Define comparison operators so bisect (Binary Search) can compare LogRecords to datetimes
    def __lt__(self, other):
        if isinstance(other, datetime):
            return self.timestamp < other
        return self.timestamp < other.timestamp


class LogAnalyzer:
    def __init__(self):
        # Data Structures as per Initial Decision Table
        self.logs = []                              # Sorted Array for Time-Range Searches
        self.user_events = defaultdict(list)        # Hash Map: User -> List of LogRecords
        self.unique_users = set()                   # Hash Set: Unique Users
        self.unique_ips = set()                     # Hash Set: Unique IPs
        self.user_event_counts = defaultdict(int)   # Hash Map: User -> Total Event Count
        self.file_access_counts = defaultdict(int)  # Hash Map: File -> Access Count
        
        # State tracking for anomalies
        self.duplicate_count = 0
        self.malformed_count = 0

    def load_logs(self, file_path):
        """Loads, sanitizes, and indexes the log file."""
        seen_lines = set() # Temporary Hash Set for exact duplicate detection
        
        print(f"Loading and indexing {file_path}...")
        
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                raw_line = line.strip()
                if not raw_line: continue
                
                # Handling Duplicates
                if raw_line in seen_lines:
                    self.duplicate_count += 1
                    continue
                seen_lines.add(raw_line)

                parts = raw_line.split(' ')
                
                # Handling Malformed Records (Expecting exactly 6 parts based on Lab 4A schema)
                if len(parts) != 6:
                    self.malformed_count += 1
                    continue
                
                date_str, time_str, event_type, user, ip, resource = parts
                
                try:
                    timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    self.malformed_count += 1
                    continue

                record = LogRecord(timestamp, event_type, user, ip, resource, raw_line)
                
                # Populate Data Structures
                self.logs.append(record)
                self.unique_users.add(user)
                self.unique_ips.add(ip)
                self.user_event_counts[user] += 1
                self.user_events[user].append(record)
                
                if event_type == "FILE_ACCESS":
                    self.file_access_counts[resource] += 1

        # Guarantee the array is sorted chronologically for Binary Search
        self.logs.sort(key=lambda r: r.timestamp)
        print(f"Loaded {len(self.logs)} valid records. (Duplicates ignored: {self.duplicate_count}, Malformed ignored: {self.malformed_count})")

    # --- Query 1: Which users generated the most events? ---
    def get_top_k_users(self, k=5):
        """Uses a Min-Heap (via heapq) on a Hash Map to find Top K in O(N + k log N)"""
        # heapq.nlargest builds a min-heap of size K under the hood
        return heapq.nlargest(k, self.user_event_counts.items(), key=lambda x: x[1])

    # --- Query 2: How many unique users/IP addresses occurred? ---
    def get_unique_counts(self):
        """Uses Hash Set cardinality in O(1) time"""
        return len(self.unique_users), len(self.unique_ips)

    # --- Query 3: Find all events belonging to a particular user ---
    def get_events_for_user(self, user):
        """Uses Hash Map lookup for O(1) access time"""
        return self.user_events.get(user, [])

    # --- Query 4: Find events occurring within a given time interval ---
    def get_events_in_time_range(self, start_dt, end_dt):
        """Uses Binary Search (bisect) on a Sorted Array for O(log N + R) time"""
        # Find start index
        start_idx = bisect.bisect_left(self.logs, start_dt)
        
        # Sequentially collect events until the end boundary
        results = []
        for i in range(start_idx, len(self.logs)):
            if self.logs[i].timestamp > end_dt:
                break
            results.append(self.logs[i])
        return results

    # --- Query 5: Identify repeated failed logins ---
    def identify_brute_force_ips(self, threshold=5):
        """Uses a Hash Map state tracker to find consecutive failures"""
        failed_streaks = defaultdict(int)
        flagged_ips = set()
        
        for record in self.logs:
            if record.event_type == "LOGIN_FAILED":
                failed_streaks[record.ip] += 1
                if failed_streaks[record.ip] >= threshold:
                    flagged_ips.add(record.ip)
            elif record.event_type == "LOGIN":
                # Reset streak on successful login
                failed_streaks[record.ip] = 0
                
        return flagged_ips

    # --- Query 6: Find the top k most frequently accessed files ---
    def get_top_k_files(self, k=5):
        """Uses a Min-Heap (via heapq) on a Hash Map to find Top K in O(M + k log M)"""
        return heapq.nlargest(k, self.file_access_counts.items(), key=lambda x: x[1])

    # --- Query 7: Identify users who were active during a specified interval ---
    def get_active_users_in_range(self, start_dt, end_dt):
        """Composite operation: Binary Search + Hash Set"""
        events = self.get_events_in_time_range(start_dt, end_dt)
        active_users = set()
        for event in events:
            active_users.add(event.user)
        return active_users


# --- Execution Example ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Gr05_log_analyzer.py <logfile>")
        sys.exit(1)

    analyzer = LogAnalyzer()
    analyzer.load_logs(sys.argv[1])

    print("\n--- 1. Top 3 Users ---")
    for u, count in analyzer.get_top_k_users(3):
        print(f"User: {u}, Events: {count}")

    print("\n--- 2. Unique Entities ---")
    u_count, ip_count = analyzer.get_unique_counts()
    print(f"Unique Users: {u_count}, Unique IPs: {ip_count}")

    print("\n--- 5. Repeated Failed Logins (Brute Force Detection) ---")
    threats = analyzer.identify_brute_force_ips(threshold=4)
    print(f"Flagged IPs with >= 4 consecutive failures: {threats}")

    print("\n--- 6. Top 3 Accessed Files ---")
    for f, count in analyzer.get_top_k_files(3):
        print(f"File: {f}, Accesses: {count}")

    print("\n--- 4/7. Time Range Search Example ---")
    start = datetime(2026, 9, 9, 8, 30, 0)
    end = datetime(2026, 9, 9, 8, 35, 0)
    events = analyzer.get_events_in_time_range(start, end)
    active = analyzer.get_active_users_in_range(start, end)
    print(f"Found {len(events)} events between {start.time()} and {end.time()}.")
    print(f"Number of unique active users in this window: {len(active)}")