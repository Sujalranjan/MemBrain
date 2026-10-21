def add(client, title, body, **kw):
    r = client.post("/api/memories", json={"title": title, "body": body, **kw})
    assert r.status_code == 201, r.text
    return r.json()


def search(client, query, **kw):
    r = client.post("/api/search", json={"query": query, **kw})
    assert r.status_code == 200, r.text
    return r.json()["results"]


def test_health_reports_database(client):
    assert client.get("/health").json() == {"status": "ok", "database": "ok"}


def test_create_list_delete_roundtrip(client):
    m = add(client, "Sprint review", "We agreed to ship the billing revamp in Q3.", type="meeting", tags=["billing"])
    assert m["current_strength"] == 1.0 and m["tags"] == ["billing"]

    assert [x["id"] for x in client.get("/api/memories").json()] == [m["id"]]

    assert client.delete(f"/api/memories/{m['id']}").status_code == 204
    assert client.get("/api/memories").json() == []
    assert client.delete(f"/api/memories/{m['id']}").status_code == 404


def test_validation_rejects_empty_body(client):
    assert client.post("/api/memories", json={"title": "x", "body": ""}).status_code == 422


def test_long_memory_is_chunked_and_searchable_by_late_content(client):
    body = "\n\n".join("Routine filler paragraph about nothing in particular. " * 5 for _ in range(10))
    body += "\n\nThe kubernetes cluster autoscaler threshold was raised to eighty percent."
    m = add(client, "Infra notes", body)
    top = search(client, "kubernetes autoscaler threshold")[0]
    assert top["id"] == m["id"]
    assert "autoscaler" in top["snippet"]


def test_hybrid_search_ranks_relevant_memory_first(client):
    target = add(client, "Database decision", "We chose PostgreSQL with pgvector for semantic search.", type="decision")
    add(client, "Lunch plans", "Team lunch is on Friday at the Italian place.")
    add(client, "Hiring", "Two backend engineers start next month.")

    top = search(client, "which database for semantic search")[0]
    assert top["id"] == target["id"]
    assert set(top["matched_by"]) == {"vector", "keyword"}
    assert 0 < top["score"] <= 1


def test_search_modes(client):
    add(client, "Onboarding SOP", "New hires get laptop access through the IT portal.", type="sop")
    kw = search(client, "laptop portal", mode="keyword")
    assert kw and kw[0]["matched_by"] == ["keyword"] and kw[0]["vector_similarity"] is None
    vec = search(client, "laptop portal", mode="vector")
    assert vec and vec[0]["matched_by"] == ["vector"] and vec[0]["keyword_rank"] is None


def test_workspaces_are_isolated(make_client):
    eng, other = make_client("engineering"), make_client("sales")
    m = add(eng, "Secret", "Quarterly revenue numbers.")
    assert search(other, "revenue") == []
    assert other.get("/api/memories").json() == []
    assert other.delete(f"/api/memories/{m['id']}").status_code == 404


def test_reinforce(client):
    m = add(client, "Fact", "The office wifi password rotates monthly.")
    r = client.post(f"/api/memories/{m['id']}/reinforce")
    assert r.status_code == 200 and r.json()["strength"] == 1.0
    assert client.post("/api/memories/00000000-0000-0000-0000-000000000000/reinforce").status_code == 404
