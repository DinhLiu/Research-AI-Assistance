from evaluation.extraction_scoring import evidence_precision, precision_recall


def test_precision_recall_and_support_labels():
    stats = precision_recall(["EL2N", "GraNd", "made-up"], ["EL2N", "GraNd", "data diet"])
    assert stats["precision"] == round(2 / 3, 3)
    assert stats["recall"] == round(2 / 3, 3)
    labels = evidence_precision(["supported", "unsupported", "supported"])
    assert labels["supported"] == round(2 / 3, 3)
    assert labels["n"] == 3
