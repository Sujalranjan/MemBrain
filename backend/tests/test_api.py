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


def ask(client, question):
    r = client.post("/api/ask", json={"question": question})
    assert r.status_code == 200, r.text
    return r.json()


def test_ask_returns_cited_answer_mapped_to_memories(client, answerer):
    m = add(client, "Launch decision", "The mobile launch moves to January. Payments is understaffed.")
    add(client, "Lunch", "Pizza on Friday.")

    data = ask(client, "when is the mobile launch")
    assert data["answer"] == "The mobile launch moves to January.[1]"
    assert data["grounded"] and data["model"] == "fake-model"
    assert data["sources"][0]["id"] == m["id"]
    assert data["citations"] == [
        {
            "source": 1,
            "memory_id": m["id"],
            "title": "Launch decision",
            "cited_text": "The mobile launch moves to January",
        }
    ]
    question, sources = answerer.calls[0]
    assert question == "when is the mobile launch" and sources[0].title == "Launch decision"


def test_ask_without_memories_skips_the_model(client, answerer):
    data = ask(client, "anything")
    assert data["answer"] and not data["grounded"] and data["sources"] == []
    assert answerer.calls == []


def test_ask_falls_back_to_sources_when_answers_unavailable(client, answerer):
    answerer.fail = True
    add(client, "Launch decision", "The mobile launch moves to January.")
    data = ask(client, "mobile launch")
    assert data["answer"] is None and data["notice"]
    assert data["sources"][0]["title"] == "Launch decision"


def test_blank_fields_are_rejected_and_tags_cleaned(client):
    assert client.post("/api/memories", json={"title": "   ", "body": "x"}).status_code == 422
    assert client.post("/api/memories", json={"title": "x", "body": " \n "}).status_code == 422
    m = add(client, "  Padded  ", "body", tags=[" a ", "a", "", "b"])
    assert m["title"] == "Padded" and m["tags"] == ["a", "b"]


def test_graph_links_related_memories(client):
    k8s = add(client, "Kubernetes upgrade", "Cluster upgrade to kubernetes 1.31 finished.", tags=["infra"])
    k8s2 = add(client, "Kubernetes autoscaler", "Kubernetes cluster autoscaler tuned after upgrade.", tags=["infra"])
    lunch = add(client, "Lunch", "Pizza on Friday at noon.")

    data = client.get("/api/graph").json()
    assert {n["id"] for n in data["nodes"]} == {k8s["id"], k8s2["id"], lunch["id"]}
    pairs = [{e["source"], e["target"]} for e in data["edges"]]
    assert {k8s["id"], k8s2["id"]} in pairs
    assert all(lunch["id"] not in p for p in pairs)  # nothing similar enough to link to
    edge = next(e for e in data["edges"] if {e["source"], e["target"]} == {k8s["id"], k8s2["id"]})
    assert edge["shared_tags"] == ["infra"] and 0.5 <= edge["similarity"] <= 1


def test_graph_parameters_and_deletion(client):
    a = add(client, "Alpha notes", "alpha beta gamma")
    b = add(client, "Alpha summary", "alpha beta delta")
    assert client.get("/api/graph", params={"min_similarity": 0.99}).json()["edges"] == []
    assert len(client.get("/api/graph").json()["edges"]) == 1
    client.delete(f"/api/memories/{b['id']}")
    data = client.get("/api/graph").json()
    assert [n["id"] for n in data["nodes"]] == [a["id"]] and data["edges"] == []
