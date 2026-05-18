"""Download and parse external skill taxonomy data.

Owner: ztt

Sources
-------
ESCO v1.2 (European Skills, Competences, Qualifications and Occupations)
    Website : https://esco.ec.europa.eu/en/use-esco/download
    Format  : ZIP archive containing CSV files
    License : CC BY 4.0 — European Commission
    Notes   : The download URL changes with each release.  If the default
              URL in config.py fails, visit the website above to get the
              current link and update ``cfg.external_align.esco_skills_url``.

O*NET Technology Skills
    Website : https://www.onetcenter.org/database.html
    Format  : Tab-delimited text file (Technology Skills.txt)
    License : O*NET® is a trademark of the U.S. Dept. of Labor, ETA.
    Notes   : Direct download is usually accessible without authentication.
              The URL is versioned (e.g. db_29_0).  Update the constant
              ``_ONET_TECH_URL`` below if the version changes.

Output
------
data/raw/external/esco/skills_index.json   — ESCO skill lookup index
data/raw/external/onet/tech_skills_index.json — O*NET tool lookup index

Usage
-----
    python -m optimize.data_collection.fetch_external_standards
    python -m optimize.data_collection.fetch_external_standards --skip-esco
    python -m optimize.data_collection.fetch_external_standards --esco-zip /path/to/local.zip
    python -m optimize.data_collection.fetch_external_standards --esco-dir optimize/esco
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from datetime import date
from pathlib import Path
from shutil import copyfile

import requests

from optimize.config import cfg
from optimize.data_collection.catalog import CatalogEntry, catalog
from optimize.utils.file_utils import ensure_dir, write_json
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("data_collection.fetch_external_standards")

_SNAPSHOT_DATE = date.today().isoformat()

# O*NET Technology Skills — update version number if the file moves
_ONET_TECH_URL = (
    "https://www.onetcenter.org/dl_files/database/"
    "db_29_0_text/Technology%20Skills.txt"
)


def fetch_esco(
    output_dir: Path | None = None,
    local_zip: Path | None = None,
    local_dir: Path | None = None,
) -> int:
    """Download (or read locally) the ESCO skills CSV and build a lookup index.

    Args:
        output_dir: Where to write the index (default: data/raw/external/esco).
        local_zip: Path to a pre-downloaded ESCO ZIP file (skips the download).
        local_dir: Path to an extracted ESCO CSV directory containing skills_en.csv.

    Returns:
        Number of skill entries in the index.
    """
    out_dir = output_dir or cfg.paths.raw_esco
    ensure_dir(out_dir)
    csv_path = out_dir / "skills.csv"

    if local_dir is not None:
        src_csv = local_dir / "skills_en.csv"
        if not src_csv.exists():
            raise FileNotFoundError(f"ESCO directory missing skills_en.csv: {src_csv}")
        logger.info("Using local ESCO directory: %s", local_dir)
        copyfile(src_csv, csv_path)
        zip_bytes = None
    elif local_zip is not None:
        zip_bytes = local_zip.read_bytes()
        logger.info("Using local ESCO ZIP: %s", local_zip)
    elif csv_path.exists():
        logger.info("ESCO CSV already exists, skipping download: %s", csv_path)
        zip_bytes = None
    else:
        logger.info("Downloading ESCO skills from %s …", cfg.external_align.esco_skills_url)
        try:
            resp = requests.get(cfg.external_align.esco_skills_url, timeout=120, stream=True)
            resp.raise_for_status()
            zip_bytes = resp.content
        except requests.RequestException as exc:
            logger.error(
                "ESCO download failed: %s\n"
                "Visit https://esco.ec.europa.eu/en/use-esco/download to get the\n"
                "current download link, then re-run with --esco-zip /path/to/file.zip",
                exc,
            )
            raise

    if zip_bytes is not None:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            skill_files = [n for n in zf.namelist() if "skill" in n.lower() and n.endswith(".csv")]
            if not skill_files:
                raise RuntimeError(f"No skills CSV found in ESCO ZIP. Contents: {zf.namelist()}")
            logger.info("Extracting %s …", skill_files[0])
            csv_path.write_bytes(zf.read(skill_files[0]))

    skills = []
    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uri  = row.get("conceptUri", "").strip()
            pref = row.get("preferredLabel", "").strip()
            if not uri or not pref:
                continue
            alts = [a.strip() for a in row.get("altLabels", "").split("\n") if a.strip()]
            skills.append({
                "uri":             uri,
                "preferred_label": pref,
                "alt_labels":      alts,
                "skill_type":      row.get("skillType", "").strip(),
                "reuse_level":     row.get("reuseLevel", "").strip(),
                "description":     row.get("description", "").strip()[:300],
            })

    index_path = out_dir / "skills_index.json"
    write_json(index_path, {"total": len(skills), "source": "esco_v1.2",
                             "date": _SNAPSHOT_DATE, "skills": skills})
    logger.info("ESCO index written: %s (%d entries)", index_path, len(skills))

    catalog.upsert(CatalogEntry(
        source_id     = "esco_skills_v1",
        source_name   = "ESCO Skills v1.2",
        source_type   = "esco",
        source_url    = cfg.external_align.esco_skills_url,
        license_note  = "CC BY 4.0 — European Commission",
        snapshot_date = _SNAPSHOT_DATE,
        record_count  = len(skills),
        local_path    = str(out_dir.relative_to(cfg.paths.raw_root.parent)),
        description   = "European Skills/Competences/Qualifications/Occupations taxonomy",
        tags          = ["esco", "skills", "external_standard"],
    ))
    return len(skills)


def fetch_onet_tech_skills(output_dir: Path | None = None) -> int:
    """Download the O*NET Technology Skills file and build a lookup index.

    Args:
        output_dir: Where to write the index (default: data/raw/external/onet).

    Returns:
        Number of distinct tools in the index.
    """
    out_dir = output_dir or cfg.paths.raw_onet
    ensure_dir(out_dir)
    txt_path = out_dir / "technology_skills.txt"

    if not txt_path.exists():
        logger.info("Downloading O*NET Technology Skills …")
        try:
            resp = requests.get(_ONET_TECH_URL, timeout=60)
            resp.raise_for_status()
            txt_path.write_bytes(resp.content)
        except requests.RequestException as exc:
            logger.error(
                "O*NET download failed: %s\n"
                "Visit https://www.onetcenter.org/database.html and download\n"
                "'Technology Skills' manually, then place it at: %s",
                exc, txt_path,
            )
            raise
    else:
        logger.info("O*NET file already exists, skipping download: %s", txt_path)

    tools: dict[str, dict] = {}
    with txt_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row.get("Example", "").strip()
            code = row.get("O*NET-SOC Code", "").strip()
            if not name:
                continue
            if name not in tools:
                tools[name] = {
                    "tool_name":            name,
                    "onet_commodity_title": row.get("Commodity Title", "").strip(),
                    "onet_commodity_code":  row.get("Commodity Code", "").strip(),
                    "is_hot_technology":    row.get("Hot Technology", "").strip() == "Y",
                    "linked_occ_codes":     [],
                }
            tools[name]["linked_occ_codes"].append(code)

    tool_list = list(tools.values())
    index_path = out_dir / "tech_skills_index.json"
    write_json(index_path, {"total": len(tool_list), "source": "onet_v29",
                             "date": _SNAPSHOT_DATE, "tools": tool_list})
    logger.info("O*NET tool index written: %s (%d entries)", index_path, len(tool_list))

    catalog.upsert(CatalogEntry(
        source_id     = "onet_tech_skills_v1",
        source_name   = "O*NET Technology Skills v29",
        source_type   = "onet",
        source_url    = _ONET_TECH_URL,
        license_note  = "O*NET® — U.S. Dept. of Labor, ETA",
        snapshot_date = _SNAPSHOT_DATE,
        record_count  = len(tool_list),
        local_path    = str(out_dir.relative_to(cfg.paths.raw_root.parent)),
        description   = "Technology tools and software linked to O*NET occupations",
        tags          = ["onet", "tools", "external_standard"],
    ))
    return len(tool_list)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-esco", action="store_true", help="Skip ESCO download")
    p.add_argument("--skip-onet", action="store_true", help="Skip O*NET download")
    p.add_argument("--esco-zip", type=Path, default=None,
                   help="Use a locally downloaded ESCO ZIP instead of fetching from URL")
    p.add_argument("--esco-dir", type=Path, default=None,
                   help="Use an extracted ESCO CSV directory containing skills_en.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.skip_esco:
        if args.esco_zip and args.esco_dir:
            raise SystemExit("--esco-zip and --esco-dir cannot be used together")
        n = fetch_esco(local_zip=args.esco_zip, local_dir=args.esco_dir)
        print(f"ESCO: {n} skills.")
    if not args.skip_onet:
        n = fetch_onet_tech_skills()
        print(f"O*NET: {n} technology tools.")
