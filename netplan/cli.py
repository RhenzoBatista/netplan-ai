"""Command line entry points.

python -m netplan.cli init-db
python -m netplan.cli ingest
python -m netplan.cli rf 3500 2.5
python -m netplan.cli plan "Rede 5G privada em 3500 MHz, raio de 2,5 km, mineradora em Itabira"
"""

from __future__ import annotations

import argparse
import asyncio
import json

from netplan.config import build_embeddings, get_settings
from netplan.rag import HybridRetriever
from netplan.rf import analyze_link
from netplan.telecom_db import build_db


def main() -> None:
    ap = argparse.ArgumentParser(prog="netplan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    sub.add_parser("ingest")
    rf = sub.add_parser("rf")
    rf.add_argument("freq_mhz", type=float)
    rf.add_argument("radius_km", type=float)
    plan = sub.add_parser("plan")
    plan.add_argument("text")
    args = ap.parse_args()
    s = get_settings()

    if args.cmd == "init-db":
        print(f"DB created at {build_db(s.db_path)}")
    elif args.cmd == "ingest":
        r = HybridRetriever.from_folder(s.docs_dir, build_embeddings(s), s.index_dir)
        print(f"Indexed {len(r.chunks)} chunks from {s.docs_dir}")
    elif args.cmd == "rf":
        print(analyze_link(args.freq_mhz, args.radius_km).model_dump_json(indent=2))
    elif args.cmd == "plan":
        from netplan.agents import build_planner

        async def run():
            p = build_planner(s)
            return await p.plan(await p.parse_request(args.text))

        print(json.dumps(asyncio.run(run()).model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
