import random
import datetime
import sys

# Fixed seed for reproducibility
random.seed(42)

def generate_logs(num_records, filename, is_variant=False):
    # Setup users (user001 to user100)
    users = [f"user{i:03d}" for i in range(1, 101)]
    
    # Skewed user activity: 20% of users do 80% of activity
    active_users = users[:20]
    inactive_users = users[20:]
    user_weights = [80.0 / len(active_users)] * len(active_users) + [20.0 / len(inactive_users)] * len(inactive_users)
    
    if is_variant:
        # Invert the skew for the variant
        user_weights = [20.0 / len(active_users)] * len(active_users) + [80.0 / len(inactive_users)] * len(inactive_users)
        
    # Setup IP addresses
    ips = [f"10.10.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(50)]
    
    # Setup Resources
    resources = [f"/data/report{i:03d}.txt" for i in range(1, 51)] + ["gcc", "python3", "bash", "-"]
    # Hot resources: a few resources accessed heavily
    hot_resources = resources[:5]
    cold_resources = resources[5:]
    resource_weights = [70.0 / len(hot_resources)] * len(hot_resources) + [30.0 / len(cold_resources)] * len(cold_resources)

    # Event types
    events = ["LOGIN", "LOGOUT", "LOGIN_FAILED", "FILE_ACCESS", "PROCESS_START", "PROCESS_END"]

    # Setup time bursts
    start_time = datetime.datetime(2026, 9, 9, 8, 0, 0) # Base time
    
    records = []
    
    # Rare user/event combination (Ensure exactly 1)
    rare_user = "user100"
    rare_event = "LOGOUT"
    rare_resource = "/data/super_secret_file.txt"
    records.append((start_time, rare_event, rare_user, "10.10.99.99", rare_resource))
    
    # Ties: Ensure user098 and user099 have exactly the same number of events. We will do this after generation.
    user098_count = 0
    
    # Generate bulk
    # Leave room for duplicates, malformed, and tie forcing
    bulk_count = int(num_records * 0.95)
    
    current_time = start_time
    for i in range(bulk_count):
        # Time progression
        if is_variant and 1000 <= i <= 2000:
            # Time burst shifted
            current_time += datetime.timedelta(seconds=random.randint(0, 1))
        elif not is_variant and 5000 <= i <= 6000:
             # Standard time burst
             current_time += datetime.timedelta(seconds=random.randint(0, 1))
        else:
             current_time += datetime.timedelta(seconds=random.randint(1, 60))
        
        # Select fields
        user = random.choices(users, weights=user_weights, k=1)[0]
        ip = random.choice(ips)
        event = random.choice(events)
        
        # Tie condition user: we skip user098 and user099 here to manually inject them later
        while user in ["user098", "user099", rare_user]:
            user = random.choices(users, weights=user_weights, k=1)[0]
            
        # Repeated failed logins cluster (IP: 10.10.10.10)
        if i > 0 and i % 100 == 0 and i % 200 != 0:
            event = "LOGIN_FAILED"
            user = "user015"
            ip = "10.10.10.10"
            resource = "-"
        else:
            resource = random.choices(resources, weights=resource_weights, k=1)[0]
            if event in ["LOGIN", "LOGOUT", "LOGIN_FAILED"]:
                resource = "-"
                
        records.append((current_time, event, user, ip, resource))

    # Tie Generation
    tie_count = int(num_records * 0.02) // 2
    for _ in range(tie_count):
        current_time += datetime.timedelta(seconds=random.randint(1, 60))
        records.append((current_time, "FILE_ACCESS", "user098", "10.10.1.1", "/data/tie_file.txt"))
        current_time += datetime.timedelta(seconds=random.randint(1, 60))
        records.append((current_time, "FILE_ACCESS", "user099", "10.10.1.2", "/data/tie_file.txt"))

    # Duplicates (1%)
    duplicate_count = int(num_records * 0.01)
    if records:
        for _ in range(duplicate_count):
            idx = random.randint(0, len(records) - 1)
            records.append(records[idx])
            
    # Malformed (0.1%)
    malformed_count = int(num_records * 0.001)
    for _ in range(malformed_count):
        current_time += datetime.timedelta(seconds=random.randint(1, 60))
        # Missing resource and IP
        records.append((current_time, "LOGIN", "user001", "", ""))

    # Pad to exact num_records if needed
    while len(records) < num_records:
        current_time += datetime.timedelta(seconds=random.randint(1, 60))
        records.append((current_time, "LOGIN", "user002", "10.10.5.5", "-"))
        
    records = records[:num_records]

    # Sort records by time to ensure log realism (even duplicates/malformed might get sorted slightly, but time is mostly increasing)
    records.sort(key=lambda x: x[0])

    with open(filename, 'w') as f:
        for rec in records:
            dt = rec[0]
            # Handle malformed explicitly by omitting fields if they are empty
            if rec[3] == "" and rec[4] == "":
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {rec[1]} {rec[2]}\n")
            else:
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {rec[1]} {rec[2]} {rec[3]} {rec[4]}\n")

if __name__ == "__main__":
    print("Generating events_small.log...")
    generate_logs(1000, "Gr05_events_small.log")
    
    print("Generating events_medium.log...")
    generate_logs(10000, "Gr05_events_medium.log")
    
    print("Generating events_large.log...")
    generate_logs(100000, "Gr05_events_large.log")
    
    print("Generating events_large_variant.log...")
    generate_logs(100000, "Gr05_events_large_variant.log", is_variant=True)
    print("Done!")
