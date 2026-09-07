from evaluation.synthesis_scoring import (
    citation_keep_rate,
    exclusive_labels,
    pairwise_precision_recall,
    predicted_labels,
    singleton_rate,
)


def test_pairwise_and_keep_rate():
    gold = {
        "papers": [
            {"arxiv_id": "a", "exclusive_cluster": "x"},
            {"arxiv_id": "b", "exclusive_cluster": "x"},
            {"arxiv_id": "c", "exclusive_cluster": "y"},
        ]
    }
    labels = exclusive_labels(gold)
    pred = predicted_labels(
        [
            {"cluster_id": "1", "paper_keys": ["av1", "bv1"], "size": 2},
            {"cluster_id": "2", "paper_keys": ["cv1"], "size": 1},
        ]
    )
    assert pred == {"a": "1", "b": "1", "c": "2"}
    scores = pairwise_precision_recall(pred, labels)
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert singleton_rate([{"size": 1}, {"size": 2}], 3) == round(1 / 3, 3)
    keep = citation_keep_rate(
        [{"claims": [{"validation_status": "structurally_validated"}, {"validation_status": "rejected"}]}],
        [],
    )
    assert keep == {"kept": 1, "total": 2, "rate": 0.5}
    empty = citation_keep_rate([], [])
    assert empty["rate"] is None
