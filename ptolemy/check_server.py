import requests
try:
    r = requests.get('http://localhost:8000/', timeout=2)
    print(f'API Status: {r.status_code}')
except Exception as e:
    print(f'API not accessible: {e}')
