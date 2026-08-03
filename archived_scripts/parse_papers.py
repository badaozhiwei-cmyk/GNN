import json
with open('papers1.json', encoding='utf-16') as f:
    d = json.load(f)
for w in d.get('results', []):
    print(f"{w.get('id')}: {w.get('display_name')} ({w.get('publication_year')}) - Citations: {w.get('cited_by_count')}")
    print(f"URL: {w.get('doi')}")
    print(f"Abstract: {w.get('abstract', 'No abstract')}")
    print("-" * 50)
