"""
pipeline/ingest.py
──────────────────
Day 1 — Step 2: Load raw contracts, chunk with LangChain RecursiveCharacterTextSplitter,
and save processed chunks with metadata to data/processed/.

Design notes:
  - Each chunk gets metadata: {source, contract_type, chunk_id, char_start, char_end}
  - contract_type is inferred from filename keywords (nda/lease/employment)
  - Chunks are saved as a single JSONL file for easy loading by embed.py

Run: python pipeline/ingest.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW_DIR, DATA_PROC_DIR, CHUNK_SIZE, CHUNK_OVERLAP

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("Installing langchain...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "langchain", "-q"])
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# ── Contract type inference ────────────────────────────────────────────────────
TYPE_KEYWORDS = {
    "nda": ["non-disclosure", "nda", "confidentiality", "confidential"],
    "lease": ["lease", "landlord", "tenant", "rent", "premises"],
    "employment": ["employment", "employee", "salary", "compensation", "severance"],
    "service": ["service agreement", "statement of work", "sow", "contractor"],
    "manufacturing": ["manufacturing", "manufacture", "supply agreement", "supplier", "production"],
    "development": ["development agreement", "developer", "software development", "product development"],
    "transportation": ["transportation", "pipeline", "freight", "carrier", "shipment", "delivery"],
    "marketing": ["marketing agreement", "co-branding", "reseller", "distribution", "advertising"],
    "outsourcing": ["outsourcing", "outsource", "managed services", "business process"],
    "ip_license": ["license agreement", "intellectual property", "royalty", "patent", "trademark", "copyright"],
    "consulting": ["consulting", "consultant", "advisory", "professional services"],
}

def infer_contract_type(text: str, filename: str) -> str:
    combined = (filename + " " + text[:500]).lower()
    scores = {ctype: 0 for ctype in TYPE_KEYWORDS}
    for ctype, keywords in TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                scores[ctype] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters common in SEC filings."""
    # Remove SEC EDGAR header boilerplate
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML/XML tags
    text = re.sub(r"&[a-z]+;", " ", text)          # HTML entities
    text = re.sub(r"={3,}", "\n", text)             # === dividers
    text = re.sub(r"-{3,}", "\n", text)             # --- dividers
    text = re.sub(r"\s{3,}", "  ", text)            # collapse whitespace
    text = re.sub(r"\n{4,}", "\n\n", text)          # collapse blank lines
    return text.strip()


def load_contract(path: Path) -> str:
    """Load a .txt or .pdf contract file."""
    if path.suffix == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            print(f"  ⚠ pdfplumber not installed, skipping {path.name}")
            return ""
    else:
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
    return ""


def chunk_contract(text: str, source: str, contract_type: str) -> list[dict]:
    """Split a contract into chunks with rich metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # legal-aware separators
    )
    chunks = splitter.split_text(text)
    
    result = []
    char_cursor = 0
    for i, chunk in enumerate(chunks):
        # Estimate char position (approximate — splitter doesn't give offsets)
        start = text.find(chunk[:40], char_cursor)
        if start == -1:
            start = char_cursor
        end = start + len(chunk)
        char_cursor = max(char_cursor, start + 1)

        result.append({
            "chunk_id": f"{source}__chunk_{i:04d}",
            "source": source,
            "contract_type": contract_type,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "char_start": start,
            "char_end": end,
            "text": chunk,
            "text_length": len(chunk),
        })
    return result


def ingest_all() -> int:
    raw_files = list(DATA_RAW_DIR.glob("*.txt")) + list(DATA_RAW_DIR.glob("*.pdf"))
    
    if not raw_files:
        print(f"⚠ No contracts found in {DATA_RAW_DIR}")
        print("  Run: python scripts/download_contracts.py first")
        return 0

    print(f"\n{'='*60}")
    print(f"Ingesting {len(raw_files)} contracts...")
    print(f"Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")
    print(f"{'='*60}")

    all_chunks = []
    stats = {"total_contracts": 0, "total_chunks": 0, "by_type": {}}

    for path in sorted(raw_files):
        raw_text = load_contract(path)
        if not raw_text or len(raw_text) < 100:
            print(f"  ⚠ Skipping (too short): {path.name}")
            continue

        text = clean_text(raw_text)
        contract_type = infer_contract_type(text, path.stem)
        source = path.stem
        chunks = chunk_contract(text, source, contract_type)

        all_chunks.extend(chunks)
        stats["total_contracts"] += 1
        stats["total_chunks"] += len(chunks)
        stats["by_type"][contract_type] = stats["by_type"].get(contract_type, 0) + 1

        print(f"  ✓ {path.name:50s} → {len(chunks):3d} chunks  [{contract_type}]")

    # Save as JSONL (one JSON object per line — easy streaming load)
    out_path = DATA_PROC_DIR / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    # Save summary stats
    stats_path = DATA_PROC_DIR / "ingest_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\n{'='*60}")
    print(f"✅ Ingestion complete!")
    print(f"   Contracts processed : {stats['total_contracts']}")
    print(f"   Total chunks        : {stats['total_chunks']}")
    print(f"   By type             : {stats['by_type']}")
    print(f"   Output              : {out_path}")
    print(f"{'='*60}")
    print("\nNext step: python pipeline/embed.py")

    return stats["total_chunks"]


if __name__ == "__main__":
    ingest_all()