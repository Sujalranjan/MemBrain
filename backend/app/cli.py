"""Admin CLI for API keys.

    python -m app.cli create-key --workspace acme --name "ci pipeline"
    python -m app.cli list-keys [--workspace acme]
    python -m app.cli revoke-key <prefix>
    python -m app.cli seed-demo --workspace demo     # load the sample enterprise notes (idempotent)

In Docker: `docker compose exec api python -m app.cli create-key --workspace acme --name laptop`.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from app.config import BACKEND_DIR, Settings, get_settings
from app.db import make_engine, make_sessionmaker
from app.models import Memory
from app.services import auth
from app.services.memories import store_memory

DEMO_DATA = BACKEND_DIR / "eval" / "data" / "membrain.json"
DEMO_TYPES = [
    (("SOP", "Runbook", "Error runbook", "Support macro", "Onboarding"), "sop"),
    (("Decision", "Policy", "Legal"), "decision"),
    (("Customer", "Partnership"), "client"),
    (("Meeting", "Sprint", "Retro", "Postmortem", "Incident", "Architecture", "Security review"), "meeting"),
]


def demo_type(title: str) -> str:
    return next((t for prefixes, t in DEMO_TYPES if title.startswith(prefixes)), "general")


async def seed_demo(session, settings: Settings, workspace: str) -> int:
    """Store the sample notes with staggered ages, so decay and the graph look like a lived-in workspace."""
    if await session.scalar(select(func.count()).select_from(Memory).where(Memory.workspace == workspace)):
        return 0
    from app.services.embeddings import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(settings.embedding_model)
    docs = json.loads(DEMO_DATA.read_text(encoding="utf-8"))["docs"]
    now = datetime.now(timezone.utc)
    for i, doc in enumerate(docs):
        memory = await store_memory(
            session,
            embedder,
            workspace,
            title=doc["title"],
            body=doc["body"],
            type=demo_type(doc["title"]),
            tags=["demo"],
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            commit=False,
        )
        age = timedelta(days=(i * 5) % 15, hours=i)  # deterministic spread: mostly fresh, some fading
        memory.created_at = memory.last_accessed_at = now - age
    await session.commit()
    return len(docs)


async def run(args: argparse.Namespace) -> int:
    engine = make_engine(get_settings().database_url)
    try:
        async with make_sessionmaker(engine)() as session:
            if args.command == "create-key":
                try:
                    api_key, key = await auth.create_key(session, args.workspace, args.name)
                except ValueError as e:
                    print(f"error: {e}", file=sys.stderr)
                    return 2
                print(f"Created key {api_key.prefix!r} ({api_key.name}) for workspace {api_key.workspace!r}.")
                print("Store it now; it cannot be shown again:\n")
                print(f"  {key}\n")
            elif args.command == "list-keys":
                keys = await auth.list_keys(session, args.workspace)
                print(f"{'PREFIX':<14}{'WORKSPACE':<20}{'NAME':<24}{'LAST USED':<22}STATUS")
                for k in keys:
                    used = k.last_used_at.strftime("%Y-%m-%d %H:%M") if k.last_used_at else "never"
                    status = "revoked" if k.revoked_at else "active"
                    print(f"{k.prefix:<14}{k.workspace:<20}{k.name[:22]:<24}{used:<22}{status}")
            elif args.command == "seed-demo":
                settings = get_settings()
                await auth.ensure_workspace(session, args.workspace)
                count = await seed_demo(session, settings, args.workspace)
                print(f"Seeded {count} memories into {args.workspace!r}." if count else "Workspace already has data.")
            elif args.command == "revoke-key":
                if not await auth.revoke_key(session, args.prefix):
                    print(f"error: no active key with prefix {args.prefix!r}", file=sys.stderr)
                    return 1
                print(f"Revoked key {args.prefix!r}.")
    finally:
        await engine.dispose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Manage MemBrain API keys.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-key", help="create an API key (and its workspace, if new)")
    create.add_argument("--workspace", required=True)
    create.add_argument("--name", required=True, help="what the key is for, e.g. 'laptop' or 'ci'")
    listing = sub.add_parser("list-keys", help="list API keys")
    listing.add_argument("--workspace")
    revoke = sub.add_parser("revoke-key", help="revoke an API key by its prefix")
    revoke.add_argument("prefix")
    seed = sub.add_parser("seed-demo", help="load sample enterprise notes into an empty workspace")
    seed.add_argument("--workspace", default=get_settings().workspace)
    sys.exit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
