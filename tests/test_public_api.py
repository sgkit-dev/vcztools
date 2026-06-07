import vcztools


class TestPublicAPI:
    """Pin the public API surface of the top-level ``vcztools`` package.

    Adding a symbol is a deliberate API change — extend ``EXPECTED`` and
    note it in the changelog. Removing a symbol is a breaking change.
    """

    EXPECTED = [
        "BcftoolsFilter",
        "BedEncoder",
        "BgenEncoder",
        "FieldInfo",
        "FormatEncoder",
        "GroupedCommand",
        "VariantFilter",
        "VczReader",
        "ViewBgenOptions",
        "ViewPlinkOptions",
        "__version__",
        "is_fill",
        "is_missing",
        "open_zarr",
        "trim_fill",
        "write_bgen",
        "write_bgi",
        "write_bim",
        "write_fam",
        "write_plink",
        "write_sample",
        "write_vcf",
    ]

    def test_all_matches_expected(self):
        assert sorted(vcztools.__all__) == sorted(self.EXPECTED)

    def test_every_exported_name_is_accessible(self):
        for name in vcztools.__all__:
            assert hasattr(vcztools, name), name
