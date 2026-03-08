"""
scripts/download_contracts.py
─────────────────────────────
Downloads sample contracts from two sources:
  1. SEC EDGAR full-text search (free, no login) — real filed contracts
  2. CUAD dataset (Hugging Face) — 500+ annotated contracts

Run: python scripts/download_contracts.py [--source edgar|cuad|both] [--limit N]

For Day 1 we recommend: python scripts/download_contracts.py --source cuad --limit 15
CUAD is faster to get started; EDGAR gives more authentic recent contracts.
"""

import argparse
import json
import os
import sys
import time
import requests
from pathlib import Path

# Add parent dir so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW_DIR

# ── CUAD contracts (via Hugging Face datasets API) ─────────────────────────────
# CUAD = Contract Understanding Atticus Dataset
# Paper: https://arxiv.org/abs/2103.06268
# License: CC BY 4.0
CUAD_BASE = "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/data"

# Subset of CUAD contracts by type — these are the exhibit filenames
CUAD_SAMPLE_CONTRACTS = {
    "nda": [
        "ATCESSOFTWAREINC_10-K_20050405_EX-10.9_1105538_EX-10.9.txt",
        "APOLLOINVESTMENTCORP_8-K_20050506_EX-99.1_1316005_EX-99.txt",
    ],
    "lease": [
        "ADVANCEDPHOTONIXTECHNOLOGY_10-K_20050415_EX-10.5_1086975_EX-10.5.txt",
    ],
    "employment": [
        "ACNB_10-K_20060330_EX-10.12_1361988_EX-10.12.txt",
    ],
}

# ── Hardcoded fallback: sample contract texts for offline/demo use ─────────────
SAMPLE_CONTRACTS = {
    "sample_nda.txt": """NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("Agreement") is entered into as of January 1, 2024, 
between Acme Corporation, a Delaware corporation ("Disclosing Party"), and Beta LLC, 
a California limited liability company ("Receiving Party").

1. CONFIDENTIAL INFORMATION
"Confidential Information" means any non-public information disclosed by the Disclosing Party 
to the Receiving Party, either directly or indirectly, in writing, orally or by inspection 
of tangible objects, that is designated as confidential or that reasonably should be understood 
to be confidential given the nature of the information and circumstances of disclosure.

2. OBLIGATIONS OF RECEIVING PARTY
The Receiving Party agrees to: (a) hold the Confidential Information in strict confidence; 
(b) not to disclose Confidential Information to any third parties without prior written consent; 
(c) use the Confidential Information solely for the purpose of evaluating a potential business 
relationship between the parties (the "Purpose"); and (d) protect the Confidential Information 
using at least the same degree of care it uses to protect its own confidential information, 
but in no event less than reasonable care.

3. TERM
This Agreement shall remain in effect for a period of two (2) years from the date first written above, 
unless earlier terminated by either party upon thirty (30) days prior written notice.

4. RETURN OF INFORMATION
Upon the Disclosing Party's request, the Receiving Party shall promptly return or destroy all 
Confidential Information and any copies thereof.

5. REMEDIES
The Receiving Party acknowledges that any breach of this Agreement may cause irreparable harm 
to the Disclosing Party for which monetary damages would be inadequate, and therefore the 
Disclosing Party shall be entitled to seek equitable relief, including injunction and specific 
performance, in addition to all other remedies available at law or equity.

6. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of the State of 
Delaware, without regard to its conflict of law provisions.

7. ENTIRE AGREEMENT
This Agreement constitutes the entire agreement between the parties with respect to the subject 
matter hereof and supersedes all prior or contemporaneous understandings regarding such subject matter.
""",

    "sample_employment_agreement.txt": """EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is made and entered into as of March 15, 2024 
("Effective Date"), by and between TechCorp Inc., a Nevada corporation ("Company"), 
and Jane Smith ("Employee").

1. POSITION AND DUTIES
Company hereby employs Employee as Chief Technology Officer. Employee shall report directly 
to the Chief Executive Officer and shall perform such duties as are customarily associated 
with such position, as well as such additional duties as the Company may from time to time assign.

2. TERM
The initial term of this Agreement shall be for a period of three (3) years commencing on 
the Effective Date, unless earlier terminated pursuant to Section 6 hereof. The Agreement 
shall automatically renew for successive one-year terms unless either party provides ninety 
(90) days' written notice of non-renewal.

3. COMPENSATION
(a) Base Salary. The Company shall pay Employee a base salary of $250,000 per year, 
payable in accordance with the Company's normal payroll practices.
(b) Annual Bonus. Employee shall be eligible to receive an annual performance bonus 
of up to 30% of base salary, based on achievement of performance metrics established 
by the Board of Directors.
(c) Equity. Employee shall receive a grant of 100,000 stock options, vesting over 
four (4) years with a one-year cliff, pursuant to the Company's equity incentive plan.

4. BENEFITS
Employee shall be entitled to participate in all employee benefit plans and programs 
that the Company makes available to its senior executives, including health insurance, 
dental, vision, and 401(k) with 4% company match.

5. CONFIDENTIALITY AND INTELLECTUAL PROPERTY
Employee agrees to execute the Company's standard Confidential Information and Invention 
Assignment Agreement concurrently with this Agreement.

6. TERMINATION
(a) Termination for Cause. The Company may terminate Employee's employment for Cause 
immediately upon written notice. "Cause" means: (i) willful misconduct; (ii) breach of 
fiduciary duty; (iii) conviction of a felony; or (iv) material breach of this Agreement.
(b) Termination Without Cause. The Company may terminate Employee's employment without 
Cause upon sixty (60) days' written notice, in which case Employee shall receive severance 
equal to six (6) months of base salary.
(c) Resignation. Employee may resign upon thirty (30) days' written notice.

7. NON-COMPETE
During the term and for a period of twelve (12) months following termination, Employee 
shall not directly or indirectly engage in any business that competes with the Company 
within the United States.

8. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Nevada.
""",

    "sample_lease_agreement.txt": """COMMERCIAL LEASE AGREEMENT

This Commercial Lease Agreement ("Lease") is entered into as of February 1, 2024, 
by and between Landlord Properties LLC ("Landlord") and StartupCo Inc. ("Tenant").

1. PREMISES
Landlord hereby leases to Tenant the premises located at 123 Business Park Drive, 
Suite 400, Austin, Texas 78701, consisting of approximately 5,000 square feet 
of office space ("Premises").

2. TERM
The initial lease term shall commence on February 1, 2024 ("Commencement Date") and 
expire on January 31, 2027 ("Expiration Date"), unless sooner terminated pursuant 
to the provisions hereof. Tenant shall have two (2) options to renew for additional 
three-year terms, provided Tenant is not in default and gives written notice at 
least six (6) months prior to expiration.

3. RENT
(a) Base Rent. Tenant shall pay monthly base rent of $15,000, due on the first 
day of each calendar month. Rent shall increase by 3% annually on each anniversary 
of the Commencement Date.
(b) Late Charge. Any rent payment received more than five (5) days after its due 
date shall incur a late charge equal to 5% of the overdue amount.
(c) Security Deposit. Tenant has deposited $30,000 as a security deposit, which 
shall be held by Landlord without interest.

4. USE
Tenant shall use the Premises solely for general office purposes and shall not use 
the Premises for any unlawful purpose or in violation of any applicable laws.

5. MAINTENANCE AND REPAIRS
(a) Landlord's Obligations. Landlord shall maintain the structural elements, roof, 
HVAC systems, and common areas in good condition and repair.
(b) Tenant's Obligations. Tenant shall maintain the interior of the Premises in 
good condition, and shall promptly repair any damage caused by Tenant's use.

6. ASSIGNMENT AND SUBLETTING
Tenant shall not assign this Lease or sublet the Premises without Landlord's prior 
written consent, which shall not be unreasonably withheld, conditioned, or delayed.

7. DEFAULT
(a) Tenant Default. The following constitute events of default: (i) failure to pay 
rent within five (5) days of written notice; (ii) failure to cure any non-monetary 
default within thirty (30) days of written notice; (iii) abandonment of the Premises.
(b) Landlord's Remedies. Upon Tenant's default, Landlord may terminate this Lease, 
re-enter the Premises, and pursue all remedies available at law or in equity.

8. INDEMNIFICATION
Tenant shall indemnify, defend, and hold harmless Landlord from any claims arising 
out of Tenant's use of the Premises, except to the extent caused by Landlord's 
gross negligence or willful misconduct.

9. GOVERNING LAW
This Lease shall be governed by the laws of the State of Texas.
"""
}


def download_cuad_samples(limit: int = 15) -> list[Path]:
    """
    Download contracts from the CUAD dataset via Hugging Face.
    Falls back to bundled sample contracts if network unavailable.
    """
    print(f"\n{'='*60}")
    print("Downloading CUAD contracts...")
    print(f"{'='*60}")

    saved = []

    # Try Hugging Face first
    hf_url = "https://huggingface.co/datasets/theatticusproject/cuad-qa/resolve/main/CUAD_v1.json"
    try:
        print("Fetching CUAD contract list from Hugging Face...")
        resp = requests.get(hf_url, timeout=30)
        if resp.status_code == 200:
            cuad_data = resp.json()
            contracts_seen = set()
            count = 0
            for item in cuad_data.get("data", []):
                title = item.get("title", "unknown")
                # Each paragraph has context (contract text)
                context = item.get("paragraphs", [{}])[0].get("context", "")
                if title not in contracts_seen and context and count < limit:
                    contracts_seen.add(title)
                    safe_name = "".join(c if c.isalnum() else "_" for c in title[:60])
                    out_path = DATA_RAW_DIR / f"cuad_{safe_name}.txt"
                    out_path.write_text(context, encoding="utf-8")
                    saved.append(out_path)
                    count += 1
                    print(f"  ✓ Saved: {out_path.name}")
            if saved:
                return saved
    except Exception as e:
        print(f"  ⚠ Hugging Face fetch failed: {e}")

    # Fallback: use bundled sample contracts
    print("\nUsing bundled sample contracts (offline mode)...")
    for filename, content in SAMPLE_CONTRACTS.items():
        out_path = DATA_RAW_DIR / filename
        out_path.write_text(content, encoding="utf-8")
        saved.append(out_path)
        print(f"  ✓ Saved: {out_path.name}")

    return saved


def download_edgar_contracts(limit: int = 10) -> list[Path]:
    """
    Download real contracts from SEC EDGAR full-text search.
    Searches for exhibit filings of type EX-10 (material contracts).
    """
    print(f"\n{'='*60}")
    print("Downloading SEC EDGAR contracts...")
    print(f"{'='*60}")

    saved = []
    headers = {"User-Agent": "MIS285N-Project student@utexas.edu"}

    # EDGAR EFTS (full-text search) API
    search_url = "https://efts.sec.gov/LATEST/search-index?q=%22non-disclosure+agreement%22&dateRange=custom&startdt=2022-01-01&enddt=2024-01-01&forms=EX-10.1,EX-10.2,EX-10.9"

    try:
        print("Querying EDGAR EFTS for NDA exhibits...")
        resp = requests.get(search_url, headers=headers, timeout=20)
        if resp.status_code != 200:
            print(f"  ⚠ EDGAR returned {resp.status_code}")
            return saved

        hits = resp.json().get("hits", {}).get("hits", [])[:limit]
        for hit in hits:
            src = hit.get("_source", {})
            file_date = src.get("file_date", "unknown")
            entity = src.get("entity_name", "entity")
            txt_url = f"https://www.sec.gov/Archives/edgar/{src.get('file_path', '')}"

            try:
                doc = requests.get(txt_url, headers=headers, timeout=15)
                if doc.status_code == 200:
                    safe = "".join(c if c.isalnum() else "_" for c in entity[:40])
                    fname = DATA_RAW_DIR / f"edgar_{safe}_{file_date}.txt"
                    fname.write_text(doc.text[:50000], encoding="utf-8")  # cap at 50K chars
                    saved.append(fname)
                    print(f"  ✓ {fname.name}")
                    time.sleep(0.5)  # EDGAR rate limit courtesy
            except Exception as e:
                print(f"  ⚠ Skipped {entity}: {e}")

    except Exception as e:
        print(f"  ⚠ EDGAR search failed: {e}")

    return saved


def summarize(saved: list[Path]):
    print(f"\n{'='*60}")
    print(f"✅ Download complete: {len(saved)} contracts saved to {DATA_RAW_DIR}")
    print(f"{'='*60}")
    total_chars = sum(p.stat().st_size for p in saved)
    print(f"Total size: {total_chars / 1024:.1f} KB")
    for p in saved:
        kb = p.stat().st_size / 1024
        print(f"  {p.name:60s} {kb:6.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download legal contracts")
    parser.add_argument("--source", choices=["edgar", "cuad", "both"], default="cuad")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    all_saved = []
    if args.source in ("cuad", "both"):
        all_saved += download_cuad_samples(args.limit)
    if args.source in ("edgar", "both"):
        all_saved += download_edgar_contracts(args.limit)

    summarize(all_saved)
    print("\nNext step: python pipeline/ingest.py")