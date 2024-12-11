import requests

# URL of the Flask application
BASE_URL = "http://127.0.0.1:5000"  # Update if running on a different host or port

# Test data for the route
test_payload = {
    "prices": [100000, 102000, 105000, 103000, 106000, 108000, 110000, 112000, 23123123123, 118000]
}

def test_predict_route():
    # Full endpoint URL
    endpoint = f"{BASE_URL}/predict"
    
    try:
        # Send POST request to the predict endpoint
        response = requests.post(endpoint, json=test_payload)
        
        # Check if the response is successful
        if response.status_code == 200:
            print("Test Passed!")
            print("Response Data:", response.json())
        else:
            print(f"Test Failed with status code {response.status_code}")
            print("Response:", response.text)
            
    except Exception as e:
        print("Test Failed!")
        print("Error:", str(e))

if __name__ == "__main__":
    test_predict_route()
