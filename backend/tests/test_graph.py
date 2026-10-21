import uuid

from app.services.graph import select_edges

A, B, C, D = (uuid.uuid4() for _ in range(4))


def key(x, y):
    return (x, y) if str(x) < str(y) else (y, x)


def test_keeps_best_neighbour_and_near_ties_only():
    pairs = [(A, B, 0.62), (A, C, 0.615), (A, D, 0.58)]
    assert set(select_edges(pairs, margin=0.01, min_similarity=0.5)) == {key(A, B), key(A, C)}


def test_floor_drops_weak_best_matches():
    assert select_edges([(A, B, 0.45)], margin=0.01, min_similarity=0.5) == {}


def test_directed_picks_merge_into_one_undirected_edge():
    edges = select_edges([(A, B, 0.6), (B, A, 0.6), (B, C, 0.55)], margin=0.01, min_similarity=0.5)
    assert edges == {key(A, B): 0.6}


def test_every_source_contributes_its_best_neighbour():
    pairs = [(A, B, 0.9), (C, A, 0.55), (C, B, 0.52)]
    assert set(select_edges(pairs, margin=0.01, min_similarity=0.5)) == {key(A, B), key(A, C)}
