import vcztools


class TestPublicAPI:
    """Pin the public API surface of the top-level ``vcztools`` package.

    Adding a symbol is a deliberate API change — extend ``EXPECTED`` and
    note it in the changelog. Removing a symbol is a breaking change.
    """

    EXPECTED = [
        "BedEncoder",
        "BgenEncoder",
        "ViewBgenOptions",
        "ViewPlinkOptions",
        "__version__",
        "open_zarr",
    ]

    def test_all_matches_expected(self):
        assert sorted(vcztools.__all__) == sorted(self.EXPECTED)

    def test_every_exported_name_is_accessible(self):
        for name in vcztools.__all__:
            assert hasattr(vcztools, name), name
