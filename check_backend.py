import requests
import sys

def check():
    url = 'http://127.0.0.1:8000/predict'
    payload={'features':{'age':22,'appearance':30,'goals':8,'assists':4,'minutes played':1800,'days_injured':0,'games_injured':0,'current_value':15000000,'highest_value':25000000,'position_encoded':4,'winger':1}}
    try:
        r = requests.post(url, json=payload, timeout=5)
        print('status', r.status_code)
        print(r.json())
    except Exception as e:
        print('error connecting to', url)
        print(e)
        sys.exit(2)

if __name__ == '__main__':
    check()
