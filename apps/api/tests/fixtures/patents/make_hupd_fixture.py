"""Regenerate hupd_2017_sample.tar.gz: a 3-record HUPD-shaped year tarball (synthetic text, real field layout).

Layout as in the real HUPD 2017.tar.gz: a "2017" directory member, then one "2017/<application_number>.json"
per application with the HUPD keys (long text fields shortened). Record 3 has a short abstract and must be
skipped by the reader. Run from anywhere: python make_hupd_fixture.py
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "hupd_2017_sample.tar.gz"

COMMON = {
    "decision": "PENDING", "patent_number": "None", "patent_issue_date": "", "abandon_date": "",
    "uspc_class": "705", "uspc_subclass": "003000", "examiner_id": "71267.0", "examiner_name_last": "EXAMPLE",
    "examiner_name_first": "ALEX", "examiner_name_middle": "", "inventor_list": [],
    "claims": "1. A method comprising a step.", "background": "Background.", "summary": "Summary.",
    "full_description": "Description.",
}

RECORDS = [
    {
        "application_number": "15900001", "publication_number": "US20180100001A1-20180412",
        "title": "SYSTEM FOR DISTRIBUTING QUANTUM KEYS BETWEEN GRID NODES",
        "date_produced": "20180328", "date_published": "20180412", "filing_date": "20171005",
        "main_cpc_label": "H04L90852", "cpc_labels": ["H04L90852", "H04L6306"],
        "main_ipcr_label": "H04L908", "ipcr_labels": ["H04L908", "H04L2906", "h04l908 "],
        "abstract": "  A key management module distributes quantum keys between smart grid nodes over\n"
                    "authenticated classical channels and rotates them on detected interference.  ",
    },
    {
        "application_number": "15900002", "publication_number": "US20180100002A1-20180419",
        "title": "",
        "date_produced": "20180404", "date_published": "", "filing_date": "20170303",
        "main_cpc_label": "F16H2522", "cpc_labels": ["F16H2522"],
        "main_ipcr_label": "", "ipcr_labels": ["F16H2522", "B62D504"],
        "abstract": "A ball screw drive converts the rotation of a motor into linear motion of a steering rack "
                    "with a recirculating ball nut.",
    },
    {
        "application_number": "15900003", "publication_number": "US20180100003A1-20180426",
        "title": "SHORT ONE", "date_produced": "20180411", "date_published": "20180426",
        "filing_date": "20170707", "main_cpc_label": "A01B100", "cpc_labels": ["A01B100"],
        "main_ipcr_label": "A01B100", "ipcr_labels": ["A01B100"],
        "abstract": "Too short to vote.",
    },
]


def main() -> None:
    with tarfile.open(OUT, mode="w:gz") as tf:
        d = tarfile.TarInfo("2017")
        d.type = tarfile.DIRTYPE
        d.mode = 0o755
        tf.addfile(d)
        for rec in RECORDS:
            data = json.dumps({**rec, **COMMON}).encode("utf-8")
            info = tarfile.TarInfo(f"2017/{rec['application_number']}.json")
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 1507161600  # fixed, so the archive is reproducible apart from gzip's header
            tf.addfile(info, io.BytesIO(data))
    print(OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
