import csv, json, math, argparse, collections


def dcg(rels):
    return sum((2**r - 1)/math.log2(i+2) for i, r in enumerate(rels))


def ndcg_at_k(ground, ranked, k=10):
    rels = [ground.get(doc, 0) for doc in ranked[:k]]
    ideal = sorted(ground.values(), reverse=True)[:k]
    return dcg(rels) / (dcg(ideal) or 1.0)


def recall_at_k(ground, ranked, k=50):
    positives = {d for d, r in ground.items() if r > 0}
    hits = sum(1 for d in ranked[:k] if d in positives)
    return hits / (len(positives) or 1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True, help="jd_id,resume_id,label")
    ap.add_argument("--runs_jsonl", required=True, help="one JSON per line: {jd_id:..., ranked_ids:[...]}" )
    args = ap.parse_args()

    by_jd = collections.defaultdict(dict)
    with open(args.labels_csv) as f:
        for row in csv.DictReader(f):
            by_jd[row["jd_id"]][row["resume_id"]] = int(row["label"]) 

    ndcgs, recalls = [], []
    with open(args.runs_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            jd = rec["jd_id"]
            ranked = rec["ranked_ids"]
            ground = by_jd.get(jd, {})
            if not ground:
                continue
            ndcgs.append(ndcg_at_k(ground, ranked, 10))
            recalls.append(recall_at_k(ground, ranked, 50))
    if ndcgs:
        print(f"nDCG@10: {sum(ndcgs)/len(ndcgs):.3f}  Recall@50: {sum(recalls)/len(recalls):.3f}  (JDs={len(ndcgs)})")
    else:
        print("No matching JDs found in runs for provided labels")


