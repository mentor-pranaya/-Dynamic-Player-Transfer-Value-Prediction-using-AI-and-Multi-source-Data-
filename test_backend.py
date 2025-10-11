from fastapi.testclient import TestClient
import backend

def run_test():
    # explicit load of model for TestClient environment
    try:
        backend.load_model()
    except Exception:
        pass
    client = TestClient(backend.app)
    payload={'features':{'age':25,'appearance':50,'goals':5,'assists':3,'minutes played':2000,'days_injured':10,'games_injured':2,'current_value':5000000,'highest_value':10000000,'position_encoded':3,'winger':0}}
    resp = client.post('/predict', json=payload)
    print(resp.status_code)
    print(resp.json())

if __name__ == '__main__':
    run_test()
