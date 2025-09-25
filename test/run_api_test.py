import requests
import json
import os

def run_evaluation_test():
    """
    Reads a custom kernel file, sends it to the KernelBench API for evaluation,
    and prints the result.
    """
    # --- Configuration ---
    # The host port you mapped in your `docker run` command
    API_PORT = 30000
    API_URL = f"http://localhost:{API_PORT}/evaluate"
    
    # The file containing the custom code to be tested
    TEST_FILE_PATH = "test/test.py"
    
    # The problem this code is intended for
    LEVEL = 1
    PROBLEM_ID = 2

    print(f"--- KernelBench API Test ---")
    print(f"Targeting API at: {API_URL}")

    # 1. Read the custom kernel code from the file
    try:
        with open(TEST_FILE_PATH, 'r', encoding='utf-8') as f:
            custom_code_string = f.read()
        print(f"Successfully read custom code from '{TEST_FILE_PATH}'")
    except FileNotFoundError:
        print(f"Error: Test file not found at '{os.path.abspath(TEST_FILE_PATH)}'.")
        print("Please make sure the file exists and you are running this script from the project root.")
        return

    # 2. Construct the JSON payload for the POST request
    payload = {
        "level": LEVEL,
        "problem_id": PROBLEM_ID,
        "custom_code": custom_code_string,
        # You can also add optional eval_params here if needed
        "eval_params": {
            "num_correct_trials": 3,
            "num_perf_trials": 10
        },
        "device": "cuda:0"
    }

    # 3. Send the request to the API
    try:
        print("Sending request to the evaluation server...")
        response = requests.post(API_URL, json=payload, timeout=300) # 5-minute timeout for compilation and eval

        # 4. Process the response
        if response.status_code == 200:
            print("\n✅ Evaluation successful! Server response:")
            # Pretty-print the JSON response
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"\n❌ Evaluation failed. Server returned status code: {response.status_code}")
            print("Server error message:")
            # Try to print the JSON error detail, or fall back to raw text
            try:
                print(json.dumps(response.json(), indent=2))
            except json.JSONDecodeError:
                print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"\n❌ An error occurred while trying to connect to the server.")
        print(f"Error details: {e}")
        print(f"Please ensure the Docker container is running and port {API_PORT} is correctly mapped.")

if __name__ == "__main__":
    run_evaluation_test()