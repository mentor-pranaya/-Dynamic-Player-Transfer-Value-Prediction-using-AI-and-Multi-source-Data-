import os
import requests
import json

backend = os.environ.get('BACKEND_URL','http://127.0.0.1:8000')
url = backend.rstrip('/') + '/predict_ts'
payload = {'name':'test_player','current_value':1200000,'highest_value':2000000}
try:
    r = requests.post(url, json=payload, timeout=10)
    print('status', r.status_code)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print('error connecting to', url)
    print(e)
