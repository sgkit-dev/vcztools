import shutil
import socket
import subprocess
import time
import uuid

import icechunk
import numpy as np
import pytest
import zarr
from icechunk import Repository

from vcztools.utils import make_icechunk_storage

from .utils import copy_store

AZURITE_ACCOUNT = "devstoreaccount1"
AZURITE_KEY = (
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr"
    "/KBHBeksoGMGw=="
)


def _find_free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(host, port, process, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Azurite exited early with code {process.returncode}")
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for Azurite on {host}:{port}")


def _assert_roots_equal(root1, root2):
    assert sorted(root1.keys()) == sorted(root2.keys())
    for key in root1.keys():
        np.testing.assert_array_equal(root1[key][:], root2[key][:], err_msg=key)


@pytest.fixture
def azurite_env(tmp_path_factory, monkeypatch):
    if shutil.which("azurite") is None:
        pytest.skip("azurite is required for Azure integration tests")
    blob = pytest.importorskip("azure.storage.blob")

    blob_port = _find_free_port()
    queue_port = _find_free_port()
    table_port = _find_free_port()
    workspace = tmp_path_factory.mktemp("azurite")

    process = subprocess.Popen(
        [
            "azurite",
            "--location",
            str(workspace),
            "--blobHost",
            "127.0.0.1",
            "--blobPort",
            str(blob_port),
            "--queueHost",
            "127.0.0.1",
            "--queuePort",
            str(queue_port),
            "--tableHost",
            "127.0.0.1",
            "--tablePort",
            str(table_port),
            "--silent",
            "--disableTelemetry",
            "--skipApiVersionCheck",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_port("127.0.0.1", blob_port, process)

        endpoint = f"http://127.0.0.1:{blob_port}/{AZURITE_ACCOUNT}"
        connection_string = (
            "DefaultEndpointsProtocol=http;"
            f"AccountName={AZURITE_ACCOUNT};"
            f"AccountKey={AZURITE_KEY};"
            f"BlobEndpoint={endpoint};"
        )
        container = f"vczstore{uuid.uuid4().hex[:16]}"
        client = blob.BlobServiceClient.from_connection_string(connection_string)
        client.create_container(container)

        monkeypatch.setenv("AZURE_STORAGE_ACCESS_KEY", AZURITE_KEY)
        monkeypatch.setenv("AZURE_ENDPOINT", endpoint)
        monkeypatch.setenv("AZURE_ALLOW_HTTP", "true")

        yield {
            "account": AZURITE_ACCOUNT,
            "container": container,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def test_azure_icechunk(fx_sample_vcz3, azurite_env):
    azure_url = f"az://{azurite_env['account']}/{azurite_env['container']}/store.vcz"
    icechunk_storage = make_icechunk_storage(azure_url)
    repo = Repository.create(icechunk_storage)

    with repo.transaction("main", message="create") as dest:
        copy_store(fx_sample_vcz3.group, dest)

    repo = Repository.open(icechunk_storage)
    session = repo.readonly_session("main")
    actual_root = zarr.open_group(store=session.store, mode="r")
    expected_root = fx_sample_vcz3.group

    assert actual_root["sample_id"][:].tolist() == ["NA00001", "NA00002", "NA00003"]
    _assert_roots_equal(expected_root, actual_root)


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
