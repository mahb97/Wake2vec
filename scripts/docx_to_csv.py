import re, csv, argparse
from docx import Document

def parse(path):
    doc = Document(path)
    cat = None
    rows = []
    cat_re = re.compile(r'^(prefix|suffix)\s*[-–]?\s*([A-Za-z()\'\.]+)', re.I)

    def norm_line(s):
        s = s.strip()
        s = re.sub(r'\b\d+\b$', '', s)              # drop trailing counts
        s = re.sub(r'\((sic|decap)\)', '', s, flags=re.I)
        return s.strip()

    for p in doc.paragraphs:
        line = (p.text or "").strip()
        if not line: continue
        m = cat_re.match(line)
        if m:
            cat = (m.group(1).lower(), m.group(2))
            continue
        if len(line) <= 2 and line.isalpha():  # A/B/C dividers
            continue
        for chunk in re.split(r'[,\t]+|\s{2,}', norm_line(line)):
            if not chunk or not re.search(r'[A-Za-z]', chunk): 
                continue
            rec = {"term": chunk, "type": "", "prefix":"", "suffix":"", "source":"FW"}
            if cat:
                kind, aff = cat
                rec["type"] = kind
                rec[kind] = aff
            rows.append(rec)
    # de-dupe preferring labeled rows
    by_term = {}
    for r in rows:
        t = r["term"]
        if t not in by_term: by_term[t] = r
        else:
            for k in ("type","prefix","suffix"):
                if r.get(k) and not by_term[t].get(k):
                    by_term[t][k] = r[k]
    return list(by_term.values())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True)
    ap.add_argument("--out", default="data/affixes.csv")
    args = ap.parse_args()
    rows = parse(args.docx)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["term","type","prefix","suffix","source","count","notes"])
        w.writeheader()
        for r in rows:
            r.setdefault("count",""); r.setdefault("notes","")
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {args.out}")
