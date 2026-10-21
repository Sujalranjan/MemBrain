import asyncio

import asyncpg

from tests.conftest import _raw_dsn


def add(client, title, body):
    r = client.post("/api/memories", json={"title": title, "body": body})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def keyword(client, query):
    r = client.post("/api/search", json={"query": query, "mode": "keyword", "k": 10})
    assert r.status_code == 200, r.text
    return [h["id"] for h in r.json()["results"]]


async def _fetch(url, query, *args):
    conn = await asyncpg.connect(_raw_dsn(url))
    try:
        return await conn.fetch(query, *args)
    finally:
        await conn.close()


def test_any_term_matches_natural_language_questions(client):
    target = add(client, "Warehouse migration", "Priya owns the Snowflake cutover at the end of Q2.")
    # Most of these words appear in no memory; requiring all of them would return nothing.
    assert keyword(client, "who is responsible for the snowflake cutover timeline") == [target]


def test_rare_terms_outweigh_common_ones(client):
    for i in range(6):
        add(client, f"Weekly sync {i}", "The team discussed the release and the deployment schedule.")
    rare = add(client, "Kafka review", "Kafka was rejected for the event bus.")
    # "release" appears in six memories, "kafka" in one: IDF should put the Kafka note first.
    assert keyword(client, "release kafka")[0] == rare


def test_title_matches_count_double(client):
    body_only = add(client, "Notes", "We talked about pricing for a while.")
    in_title = add(client, "Pricing", "We talked about it for a while.")
    assert keyword(client, "pricing")[:2] == [in_title, body_only]


def test_web_search_syntax_is_strict(client):
    both = add(client, "Failover", "Route 53 health checks switch DNS to eu-west-1.")
    dns_only = add(client, "DNS", "We moved DNS records to a new registrar.")
    assert set(keyword(client, "dns health")) == {both, dns_only}
    assert keyword(client, "dns -health") == [dns_only]
    assert keyword(client, '"health checks"') == [both]


def test_stopword_only_query_returns_nothing(client):
    add(client, "Something", "Content about deployments.")
    assert keyword(client, "the and of") == []


def test_postings_follow_memory_lifecycle(client, database_url):
    mid = add(client, "Rotation", "Rotate the Vault keys every 90 days. Rotate them again after incidents.")
    rows = asyncio.run(_fetch(database_url, "SELECT lexeme, tf FROM memory_terms WHERE memory_id = $1::uuid", mid))
    tf = {r["lexeme"]: r["tf"] for r in rows}
    assert tf["rotat"] == 2 + 2  # body occurrences + the title occurrence counted twice
    [m] = asyncio.run(_fetch(database_url, "SELECT term_count FROM memories WHERE id = $1::uuid", mid))
    assert m["term_count"] == sum(tf.values())

    client.delete(f"/api/memories/{mid}")
    assert asyncio.run(_fetch(database_url, "SELECT 1 FROM memory_terms WHERE memory_id = $1::uuid", mid)) == []
