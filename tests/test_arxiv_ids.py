from research_assistant.arxiv_ids import format_version, normalize_arxiv_id, paper_key, split_arxiv_id, version_number


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


def test_paper_key_and_version_number():
    assert paper_key("2205.09329", "v2") == "2205.09329v2"
    assert paper_key("2205.09329v3") == "2205.09329v3"
    assert version_number("v10") == 10
    assert version_number(None) == 1
