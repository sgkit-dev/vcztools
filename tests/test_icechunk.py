import pytest

from vcztools.icechunk import make_icechunk_storage

icechunk = pytest.importorskip("icechunk")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        pytest.param(
            "az://myaccount/my-container/some/prefix/store.vcz",
            {
                "account": "myaccount",
                "container": "my-container",
                "prefix": "some/prefix/store.vcz",
                "from_env": True,
            },
            id="az",
        ),
        pytest.param(
            "azure://myaccount/my-container/some/prefix/store.vcz",
            {
                "account": "myaccount",
                "container": "my-container",
                "prefix": "some/prefix/store.vcz",
                "from_env": True,
            },
            id="azure",
        ),
        pytest.param(
            "abfs://my-container@myaccount.dfs.core.windows.net/some/prefix/store.vcz",
            {
                "account": "myaccount",
                "container": "my-container",
                "prefix": "some/prefix/store.vcz",
                "from_env": True,
            },
            id="abfs",
        ),
        pytest.param(
            "abfss://my-container@myaccount.dfs.core.windows.net/some/prefix/store.vcz",
            {
                "account": "myaccount",
                "container": "my-container",
                "prefix": "some/prefix/store.vcz",
                "from_env": True,
            },
            id="abfss",
        ),
        pytest.param(
            "https://myaccount.blob.core.windows.net/my-container/some/prefix/store.vcz",
            {
                "account": "myaccount",
                "container": "my-container",
                "prefix": "some/prefix/store.vcz",
                "from_env": True,
            },
            id="https-blob",
        ),
        pytest.param(
            "https://myaccount.dfs.core.windows.net/my-container/some/prefix/store.vcz",
            {
                "account": "myaccount",
                "container": "my-container",
                "prefix": "some/prefix/store.vcz",
                "from_env": True,
            },
            id="https-dfs",
        ),
    ],
)
def test_make_icechunk_storage_azure_urls(monkeypatch, url, expected):
    captured = {}

    def fake_azure_storage(**kwargs):
        captured.update(kwargs)
        return "azure-store"

    monkeypatch.setattr(icechunk, "azure_storage", fake_azure_storage)

    assert make_icechunk_storage(url) == "azure-store"
    assert captured == expected
