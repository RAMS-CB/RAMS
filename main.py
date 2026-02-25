import sys
import os
import time
import schedule
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.mongo_connection import get_db_connection
from database.mongo import process_pipeline, start_scheduler

def main():
    print("Initializing RAMS Document Processing Pipeline...")
    
    # Verify database connection early
    try:
        print("Checking MongoDB connection...")
        client = get_db_connection()
        print("MongoDB connection successful!")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not connect to MongoDB: {e}")
        print("Exiting pipeline setup.")
        sys.exit(1)

    print("\nExecuting initial pipeline run...")
    # Run the pipeline once immediately to process any outstanding updates
    process_pipeline()
    
    print("\nStarting periodic scheduler...")
    # Set up the schedule
    schedule.every(1).hours.do(process_pipeline)
    print("Scheduler configured to run every 1 hour. Waiting for next execution...")
    
    # Keep the script running to execute scheduled tasks
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nScheduler stopped by user. Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
