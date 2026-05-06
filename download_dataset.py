import requests
import os
import json
import time
from tqdm import tqdm

# PubMed E-utilities API - completely free, no key needed for basic use
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

SEARCH_TERMS = [
    "cancer BRCA1 drug treatment",
    "lung cancer targeted therapy gene mutation",
    "breast cancer chemotherapy drug resistance gene",
    "leukemia drug gene expression",
    "colorectal cancer immunotherapy biomarker",
]


def search_pubmed(query, max_results=200):
    """Search PubMed and return list of PMC IDs"""
    url = f"{BASE_URL}/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": query + " AND open access[filter]",
        "retmax": max_results,
        "retmode": "json",
        "usehistory": "y"
    }
    response = requests.get(url, params=params)
    data = response.json()
    ids = data["esearchresult"]["idlist"]
    print(f"Found {len(ids)} papers for: {query}")
    return ids


def fetch_full_text(pmc_id):
    """Fetch full text of a paper by PMC ID"""
    url = f"{BASE_URL}/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmc_id,
        "rettype": "text",
        "retmode": "text"
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200 and len(response.text) > 500:
            return response.text
    except Exception as e:
        print(f"Error fetching {pmc_id}: {e}")
    return None


def count_tokens_approx(text):
    """Rough token count: ~4 chars per token"""
    return len(text) // 4


def main():
    all_ids = set()

    print("🔍 Searching PubMed...")
    for term in SEARCH_TERMS:
        ids = search_pubmed(term, max_results=200)
        all_ids.update(ids)
        time.sleep(0.5)  # Be polite to the API

    print(f"\n📚 Total unique papers found: {len(all_ids)}")

    total_tokens = 0
    saved_count = 0
    all_papers = []

    print("\n⬇️  Downloading full texts...")
    for pmc_id in tqdm(list(all_ids)):
        text = fetch_full_text(pmc_id)
        if text:
            tokens = count_tokens_approx(text)
            total_tokens += tokens
            paper = {
                "id": pmc_id,
                "text": text,
                "tokens": tokens
            }
            all_papers.append(paper)
            saved_count += 1

            # Save progress every 50 papers
            if saved_count % 50 == 0:
                print(f"\n💾 Saved {saved_count} papers | ~{total_tokens:,} tokens so far")

        time.sleep(0.4)  # Rate limit: max 3 requests/sec without API key

        # Stop early if we have well over 2M tokens
        if total_tokens >= 2_500_000:
            print(f"\n✅ Token target reached!")
            break

    # Save everything to one JSON file
    output_path = "data/raw/pubmed_cancer_papers.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Done!")
    print(f"   Papers downloaded: {saved_count}")
    print(f"   Approximate tokens: {total_tokens:,}")
    print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    main()