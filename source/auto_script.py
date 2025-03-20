import subprocess
import time

def run_script(script_path):
    while True:
        try:
            # Replace 'your_script.py' with the actual name of your script
            subprocess.run(['python', script_path], check=True)
            print("Script completed successfully!")
            break  # Exit the loop if the script runs without errors
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            print("Restarting script in 5 seconds...")
            time.sleep(5)  # Wait for 5 seconds before restarting
        except Exception as e: # Catch any other exceptions
            print(f"Unexpected error: {e}")
            print("Restarting script in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    run_script(script_path='/home/ubuntu/thesis/source/factcheck/factcheck.py')