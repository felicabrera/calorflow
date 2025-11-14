import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health_check():
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_root():
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_list_models():
    print("\n=== Testing List Models ===")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_process_info():
    print("\n=== Testing Process Info (FCC) ===")
    response = requests.get(f"{BASE_URL}/process-info/FCC")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    print("\n=== Testing Process Info (CCR) ===")
    response = requests.get(f"{BASE_URL}/process-info/CCR")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_data_quality():
    print("\n=== Testing Data Quality Check ===")
    
    data_path = "data/FCC - Cracking Catalítico/Predictoras_202406_202502_FCC.csv"
    if not Path(data_path).exists():
        print(f"Skipping test - file not found: {data_path}")
        return True
    
    payload = {
        "data_path": data_path,
        "target_cols": ["PCI", "H2"]
    }
    
    response = requests.post(f"{BASE_URL}/data-quality", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Quality Score: {result.get('quality_score')}")
        print(f"Samples: {result.get('n_samples')}")
        print(f"Features: {result.get('n_features')}")
        print(f"Recommendations: {result.get('recommendations')}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_prediction():
    print("\n=== Testing Prediction ===")
    
    data_path = "data/FCC - Cracking Catalítico/Predictoras_202503_202508_FCC.csv"
    if not Path(data_path).exists():
        print(f"Skipping test - file not found: {data_path}")
        return True
    
    payload = {
        "process_name": "FCC",
        "data_path": data_path,
        "model_dir": "models"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Samples predicted: {result.get('n_samples')}")
        print(f"Statistics: {json.dumps(result.get('statistics'), indent=2)}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_submission():
    print("\n=== Testing Submission Generation ===")
    
    payload = {
        "process_name": "FCC",
        "model_dir": "models",
        "data_dir": "data",
        "output_dir": "predictions"
    }
    
    response = requests.post(f"{BASE_URL}/submission", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Submission file: {result.get('submission_file')}")
        print(f"Samples: {result.get('n_samples')}")
        print(f"PCI range: {result.get('pci_range')}")
        print(f"H2 range: {result.get('h2_range')}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def run_all_tests():
    print("="*60)
    print("CALORFLOW API TEST SUITE")
    print("="*60)
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health_check),
        ("List Models", test_list_models),
        ("Process Info", test_process_info),
        ("Data Quality", test_data_quality),
        ("Prediction", test_prediction),
        ("Submission", test_submission),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results[name] = False
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to API")
        print("Make sure the API is running: python run_api.py")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
