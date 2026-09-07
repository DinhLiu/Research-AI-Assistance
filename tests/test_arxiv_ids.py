from research_assistant.arxiv_ids import format_version, normalize_arxiv_id, split_arxiv_id


def test_split_keeps_version():
    core, version = split_arxiv_id("https://arxiv.org/abs/2205.09329v2")
    assert core == "2205.09329"
    assert version == "v2"


def test_normalize_strips_version():
    assert normalize_arxiv_id("2205.09329v1") == "2205.09329"
    assert normalize_arxiv_id("hep-th/9901001v3") == "hep-th/9901001"


def test_format_version():
    assert format_version(None) == "v1"
    assert format_version("2") == "v2"
    assert format_version("v3") == "v3"
