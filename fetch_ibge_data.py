"""Fetch IBGE SIDRA soybean yield data for three municipalities."""
import requests, json, sys

MUNICIPIOS = {
    'Sorriso_MT': '5107925',
    'Londrina_PR': '4113700',
    'Rio Verde_GO': '5218805',
}

VARS = {
    'rendimento': 214,   # kg/ha
    'producao': 216,     # toneladas
}

results = {}
for nome, codigo in MUNICIPIOS.items():
    url = f'https://apisidra.ibge.gov.br/values/t/1612/n6/{codigo}/p/all/v/214?formato=json'
    print(f'\n=== {nome} ({codigo}) ===')
    print(f'URL: {url}')
    try:
        r = requests.get(url, timeout=60, headers={'Accept': 'application/json'})
        print(f'Status: {r.status_code}')
        print(f'Content-Type: {r.headers.get("Content-Type", "N/A")}')
        print(f'Length: {len(r.text)} chars')
        text_preview = r.text[:500]
        print(f'Preview: {text_preview}')

        if r.status_code == 200 and len(r.text) > 10:
            try:
                data = r.json()
                if isinstance(data, list):
                    print(f'Entries: {len(data)}')
                    if len(data) > 0:
                        print(f'Keys: {list(data[0].keys())}')
                        for i, entry in enumerate(data[:5]):
                            print(f'  [{i}] {json.dumps(entry, ensure_ascii=False)[:200]}')
                        # Print all data as table
                        print('\n--- ALL DATA TABLE ---')
                        for entry in data:
                            ano = entry.get('D3N', entry.get('D3C', '?'))
                            valor = entry.get('V', '?')
                            print(f'  Ano: {ano}, Valor: {valor}')
                else:
                    print(f'Not a list, type: {type(data)}')
                    print(json.dumps(data, ensure_ascii=False)[:500])
            except json.JSONDecodeError as e:
                print(f'JSON parse error: {e}')
        else:
            print(f'Non-200 or empty response')
    except Exception as e:
        print(f'Error: {e}')

    # Try alternative format
    if r.status_code != 200 or len(r.text) < 10:
        print(f'\nTrying alternative URL format...')
        url2 = f'https://apisidra.ibge.gov.br/values/t/1612/n6/{codigo}/p/all/v/214'
        try:
            r2 = requests.get(url2, timeout=60, headers={'Accept': 'application/json'})
            print(f'Status: {r2.status_code}')
            if r2.status_code == 200:
                text_preview2 = r2.text[:500]
                print(f'Preview: {text_preview2}')
        except Exception as e2:
            print(f'Error: {e2}')

print('\n\nDONE')
