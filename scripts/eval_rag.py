"""
RAG retrieval quality evaluation for resume-supporting figures.

Runs a small set of e-commerce-style queries against the vector store and
reports hit metrics to back up the "RAG 检索效果" claim on the resume.
"""
import time

from src.tools import get_vector_store

# (query, expected_substrings) — expected_substrings are tokens that a good
# hit for this query should contain in the product_name.
QUERIES = [
    ("organic avocado", ["avocado", "avocados"]),
    ("low fat greek yogurt", ["yogurt"]),
    ("whole wheat bread", ["bread", "wheat"]),
    ("chocolate milk", ["chocolate", "milk"]),
    ("frozen pizza", ["pizza"]),
    ("gala apples", ["apple"]),
    ("2 percent milk gallon", ["milk"]),
    ("paper towels", ["towel", "towel", "paper"]),
    ("sparkling water", ["water", "carbonated"]),
    ("cage free eggs", ["egg"]),
]


def main():
    vs = get_vector_store()
    print(f"{'query':<26} | {'top3_hit':<9} | {'top5_hit':<9} | avg_latency_ms")
    print("-" * 70)
    total_top3 = 0
    total_top5 = 0
    latencies = []
    for q, keys in QUERIES:
        t0 = time.time()
        top3 = vs.similarity_search(q, k=3)
        top5 = vs.similarity_search(q, k=5)
        lat = (time.time() - t0) / 2 * 1000
        latencies.append(lat)

        def hit(docs, keys):
            names = [d.metadata.get("product_name", "").lower() for d in docs]
            return any(any(k in n for k in keys) for n in names)

        h3 = hit(top3, keys)
        h5 = hit(top5, keys)
        total_top3 += h3
        total_top5 += h5
        print(
            f"{q:<26} | {str(h3):<9} | {str(h5):<9} | {lat:.0f}"
        )

    n = len(QUERIES)
    print("-" * 70)
    print(f"\nSUMMARY over {n} queries")
    print(f"  Top-3 relevance hit rate : {total_top3}/{n} = {total_top3/n*100:.0f}%")
    print(f"  Top-5 relevance hit rate : {total_top5}/{n} = {total_top5/n*100:.0f}%")
    print(f"  Avg retrieve latency      : {sum(latencies)/len(latencies):.0f} ms")
    print(f"  Catalog size              : {vs._collection.count()} chunks")


if __name__ == "__main__":
    main()