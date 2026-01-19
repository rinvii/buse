import pytest
import buse.main as main


class MockBrowserTab:
    def __init__(self, target_id):
        self.target_id = target_id


def test_parse_image_quality(monkeypatch):
    assert main._parse_image_quality(50) == 50
    assert main._parse_image_quality(None) is None

    monkeypatch.setenv("BUSE_IMAGE_QUALITY", "80")
    assert main._parse_image_quality(None) == 80

    monkeypatch.setenv("BUSE_IMAGE_QUALITY", "invalid")
    with pytest.raises(main.typer.BadParameter):
        main._parse_image_quality(None)

    monkeypatch.setenv("BUSE_IMAGE_QUALITY", "150")
    with pytest.raises(main.typer.BadParameter):
        main._parse_image_quality(None)


def test_parse_bool_env(monkeypatch):
    monkeypatch.setenv("TEST_BOOL", "1")
    assert main._parse_bool_env("TEST_BOOL") is True
    monkeypatch.setenv("TEST_BOOL", "true")
    assert main._parse_bool_env("TEST_BOOL") is True
    monkeypatch.setenv("TEST_BOOL", "false")
    assert main._parse_bool_env("TEST_BOOL") is False
    monkeypatch.setenv("TEST_BOOL", "")
    assert main._parse_bool_env("TEST_BOOL") is None


def test_is_reserved_key_sequence():
    assert main._is_reserved_key_sequence("Control+C") is True
    assert main._is_reserved_key_sequence("Enter") is True
    assert main._is_reserved_key_sequence("Tab") is True
    assert main._is_reserved_key_sequence("a") is False
    assert main._is_reserved_key_sequence("hello") is False
    assert main._is_reserved_key_sequence(None) is False


def test_get_selector_cache_ttl_seconds(monkeypatch):
    monkeypatch.delenv("BUSE_SELECTOR_CACHE_TTL", raising=False)
    assert main._get_selector_cache_ttl_seconds() == 0.0

    monkeypatch.setenv("BUSE_SELECTOR_CACHE_TTL", "1.5")
    assert main._get_selector_cache_ttl_seconds() == 1.5

    monkeypatch.setenv("BUSE_SELECTOR_CACHE_TTL", "invalid")
    assert main._get_selector_cache_ttl_seconds() == 0.0


def test_tab_present():
    tabs = [MockBrowserTab(target_id="tab1-abcd")]
    assert main._tab_present(tabs, "abcd") is True
    assert main._tab_present(tabs, "wxyz") is False


def test_normalize_tab_id():
    tabs = [
        MockBrowserTab(target_id="long-id-1234"),
        MockBrowserTab(target_id="other-5678"),
    ]
    id_val, matched = main._normalize_tab_id("1234", tabs)
    assert id_val == "1234"
    assert matched is False

    id_val, matched = main._normalize_tab_id("long", tabs)
    assert id_val == "1234"
    assert matched is True


def test_coerce_index_error():
    msg = "Element index 5 not available in browser state"
    assert main._coerce_index_error(msg) == msg
    assert main._coerce_index_error("Some other error") is None


def test_augment_error():
    assert main._augment_error("other", {}, "Error") == "Error"

    err = main._augment_error("click", {}, "Error")
    assert "Provide an index or use --id/--class" in err

    err = main._augment_error("navigate", {"url": "example.com"}, "site unavailable")
    assert "Include a scheme" in err
