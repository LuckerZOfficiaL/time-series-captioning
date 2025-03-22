import subprocess
import time

def run_script(script_path):
    while True:
        try:
            subprocess.run(['python', script_path], check=True)
            print("\nScript completed successfully!")
            break  # Exit the loop if the script has succesfully terminated
        except subprocess.CalledProcessError as e:
            print(f"\nError occurred: {e}")
            print("Restarting script in 5 seconds...")
            time.sleep(5)
        except Exception as e: # Catch any other exceptions
            print(f"\nUnexpected error: {e}")
            print("Restarting script in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    run_script(script_path='/home/ubuntu/thesis/source/factcheck/factcheck_detection.py')