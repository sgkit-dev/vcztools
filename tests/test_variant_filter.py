import numpy as np

from vcztools import variant_filter


class TestAndFilter:
    """Direct unit tests for :class:`vcztools.variant_filter.AndFilter`."""

    @staticmethod
    def _variant_filter(mask, fields=("variant_position",)):
        class _F:
            scope = "variant"
            referenced_fields = frozenset(fields)

            def evaluate(self, chunk_data):
                return np.asarray(mask, dtype=bool)

        return _F()

    @staticmethod
    def _sample_filter(mask, fields=("call_DP",)):
        class _F:
            scope = "sample"
            referenced_fields = frozenset(fields)

            def evaluate(self, chunk_data):
                return np.asarray(mask, dtype=bool)

        return _F()

    def test_variant_and_variant(self):
        a = self._variant_filter([True, True, False])
        b = self._variant_filter([True, False, False], fields=("variant_id",))
        combined = variant_filter.AndFilter([a, b])
        assert combined.scope == "variant"
        assert combined.referenced_fields == frozenset(
            {"variant_position", "variant_id"}
        )
        np.testing.assert_array_equal(combined.evaluate({}), [True, False, False])

    def test_variant_and_sample_broadcasts(self):
        # Variant mask (1-D) broadcasts along axis 1 against a 2-D
        # sample mask. Variant 0: variant True & sample [T, F] → [T, F].
        # Variant 1: variant False zeros the row regardless of samples.
        var = self._variant_filter([True, False])
        samp = self._sample_filter([[True, False], [True, True]])
        combined = variant_filter.AndFilter([var, samp])
        assert combined.scope == "sample"
        np.testing.assert_array_equal(
            combined.evaluate({}), [[True, False], [False, False]]
        )

    def test_sample_and_variant_broadcasts(self):
        # Same logic, opposite operand order.
        samp = self._sample_filter([[True, False], [True, True]])
        var = self._variant_filter([True, False])
        combined = variant_filter.AndFilter([samp, var])
        assert combined.scope == "sample"
        np.testing.assert_array_equal(
            combined.evaluate({}), [[True, False], [False, False]]
        )

    def test_sample_and_sample_elementwise(self):
        a = self._sample_filter([[True, True], [False, True]])
        b = self._sample_filter([[True, False], [True, True]])
        combined = variant_filter.AndFilter([a, b])
        assert combined.scope == "sample"
        np.testing.assert_array_equal(
            combined.evaluate({}), [[True, False], [False, True]]
        )


class TestCompose:
    """Direct unit tests for :func:`vcztools.variant_filter.compose`."""

    @staticmethod
    def _filter():
        class _F:
            scope = "variant"
            referenced_fields = frozenset()

            def evaluate(self, chunk_data):
                return np.array([True])

        return _F()

    def test_compose_with_none_returns_new(self):
        f = self._filter()
        assert variant_filter.compose(None, f) is f

    def test_compose_with_existing_returns_andfilter(self):
        a = self._filter()
        b = self._filter()
        composed = variant_filter.compose(a, b)
        assert isinstance(composed, variant_filter.AndFilter)
