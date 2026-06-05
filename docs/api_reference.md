(sec-python-api)=
# API reference

The complete public Python API. See {ref}`sec-reading-vcz` for a worked,
example-driven introduction to opening datasets and iterating variants, and
{ref}`sec-format-conversion` for the PLINK and BGEN writers.

```{eval-rst}

.. autofunction:: vcztools.open_zarr

.. autoclass:: vcztools.VczReader
   :members: sample_ids, raw_sample_ids, non_null_sample_indices,
             contig_ids, num_variants, num_samples,
             field_names, virtual_field_names, set_samples,
             set_sample_indexes, set_variants,
             set_variant_filter, materialise_variant_filter,
             get_field_info, variant_chunks, variants
   :member-order: bysource

.. autoclass:: vcztools.FieldInfo
   :members:

.. autofunction:: vcztools.is_missing
.. autofunction:: vcztools.is_fill
.. autofunction:: vcztools.trim_fill

.. autoclass:: vcztools.VariantFilter
   :members:
   :undoc-members:

.. autoclass:: vcztools.BcftoolsFilter

.. autofunction:: vcztools.write_plink
.. autofunction:: vcztools.write_bgen

.. autoclass:: vcztools.FormatEncoder
   :members:

.. autoclass:: vcztools.BedEncoder
   :members:
   :show-inheritance:

.. autoclass:: vcztools.BgenEncoder
   :members:
   :show-inheritance:

.. autofunction:: vcztools.write_bim
.. autofunction:: vcztools.write_fam
.. autofunction:: vcztools.write_bgi
.. autofunction:: vcztools.write_sample

```
