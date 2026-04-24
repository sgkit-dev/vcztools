import shutil
import socket
import subprocess
import time
import uuid

import numpy as np
import pytest
import zarr

from vcztools.icechunk import make_icechunk_storage

from .utils import copy_store_to_icechunk, to_vcz_icechunk

icechunk = pytest.importorskip("icechunk")
from icechunk import Repository  # noqa E402

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


def test_copy_then_append_azure_icechunk(tmp_path, fx_sample_vcz3, azurite_env):
    vcz = to_vcz_icechunk(fx_sample_vcz3.directory_path, tmp_path)

    azure_url = f"az://{azurite_env['account']}/{azurite_env['container']}/store.vcz"

    copy_store_to_icechunk(vcz, azure_url)

    repo = Repository.open(make_icechunk_storage(azure_url))
    session = repo.readonly_session("main")
    actual_root = zarr.open_group(store=session.store, mode="r")
    expected_root = zarr.open_group(store=vcz, mode="r")

    assert actual_root["sample_id"][:].tolist() == ["NA00001", "NA00002", "NA00003"]
    _assert_roots_equal(expected_root, actual_root)
