import requests
import urllib.parse

def search_openalex(query):
    url = f"https://api.openalex.org/works?search={urllib.parse.quote(query)}&per-page=10&sort=relevance_score:desc"
    r = requests.get(url)
    data = r.json()
    for w in data.get('results', []):
        print(f"Title: {w.get('display_name')}")
        print(f"Year: {w.get('publication_year')}")
        print(f"Citations: {w.get('cited_by_count')}")
        print(f"DOI: {w.get('doi')}")
        print("-" * 50)

search_openalex('"explainable" "graph neural network" "property prediction"')
search_openalex('("explainable" OR "interpretable") "graph neural network" "ionic liquid"')
