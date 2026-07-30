import requests
url = 'https://power.larc.nasa.gov/api/temporal/daily/point'
params = {
    'parameters': 'T2M,T2M_MAX,T2M_MIN,PRECTOTCORR',
    'community': 'AG',
    'longitude': -55.71,
    'latitude': -12.55,
    'start': '19800101',
    'end': '20241231',
    'format': 'JSON'
}
try:
    resp = requests.get(url, params=params, timeout=90, headers={'Accept': 'application/json'})
    print(f'Status: {resp.status_code}')
    if resp.status_code == 200:
        data = resp.json()
        props = data['properties']['parameter']
        print(f'Keys: {list(props.keys())}')
        t2m = props['T2M']
        print(f'T2M entries: {len(t2m)}')
        print(f'Sample: {list(t2m.items())[:3]}')
    else:
        print(f'Response: {resp.text[:500]}')
except Exception as e:
    import traceback
    print(f'Error: {type(e).__name__}: {e}')
    traceback.print_exc()
