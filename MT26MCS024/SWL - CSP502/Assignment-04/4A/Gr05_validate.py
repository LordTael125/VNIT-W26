import sys
from collections import Counter

def validate_log(filepath):
    print(f"--- Validating {filepath} ---")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    total_records = len(lines)
    unique_users = set()
    unique_ips = set()
    event_counts = Counter()
    resource_counts = Counter()
    user_counts = Counter()
    
    malformed_count = 0
    duplicate_count = total_records - len(set(lines))
    
    # Rare user tracking
    rare_user_count = 0
    
    # Tie tracking
    user098_events = 0
    user099_events = 0
    
    # Failed login clusters
    failed_login_cluster_user15 = 0
    
    earliest_time = None
    latest_time = None
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            malformed_count += 1
            continue
            
        timestamp = f"{parts[0]} {parts[1]}"
        if earliest_time is None or timestamp < earliest_time:
            earliest_time = timestamp
        if latest_time is None or timestamp > latest_time:
            latest_time = timestamp
            
        if len(parts) >= 3:
            event = parts[2]
            event_counts[event] += 1
            
        if len(parts) >= 4:
            user = parts[3]
            unique_users.add(user)
            user_counts[user] += 1
            
            if user == "user100":
                rare_user_count += 1
            if user == "user098":
                user098_events += 1
            if user == "user099":
                user099_events += 1
                
        if len(parts) == 6:
            ip = parts[4]
            resource = parts[5]
            unique_ips.add(ip)
            if resource != "-":
                resource_counts[resource] += 1
                
            if event == "LOGIN_FAILED" and user == "user015" and ip == "10.10.10.10":
                failed_login_cluster_user15 += 1
        elif len(parts) == 5: # No resource provided or space issues? The generator provides 6 parts: DATE TIME EVENT USER IP RESOURCE
            pass

    print(f"Total Records: {total_records}")
    print(f"Earliest Time: {earliest_time}")
    print(f"Latest Time: {latest_time}")
    print(f"Unique Users: {len(unique_users)}")
    print(f"Unique IPs: {len(unique_ips)}")
    
    print("\nEvent Frequencies:")
    for event, count in event_counts.most_common():
        print(f"  {event}: {count}")
        
    print("\nTop 5 Active Users:")
    for user, count in user_counts.most_common(5):
        print(f"  {user}: {count}")
        
    print("\nTop 5 Hot Resources:")
    for res, count in resource_counts.most_common(5):
        print(f"  {res}: {count}")
        
    print(f"\nDuplicates: {duplicate_count} ({duplicate_count/total_records*100:.2f}%)")
    print(f"Malformed: {malformed_count} ({malformed_count/total_records*100:.2f}%)")
    
    print(f"Rare user 'user100' events (expect 1): {rare_user_count}")
    print(f"Tie check: user098 events: {user098_events}, user099 events: {user099_events} (Are they equal? {user098_events == user099_events})")
    print(f"Failed login clusters for user015 at 10.10.10.10: {failed_login_cluster_user15}")
    print("\n")

if __name__ == "__main__":
    validate_log("Gr05_events_small.log")
    validate_log("Gr05_events_medium.log")
    validate_log("Gr05_events_large.log")
    validate_log("Gr05_events_large_variant.log")
