# import pytest
# import pandas as pd

# from xeofs.preprocessing.multi_index_converter import (
#     DataListMultiIndexConverter,
# )
# from xeofs.utils.data_types import DataArray
# from ..utilities import assert_expected_dims, data_is_dask, data_has_multiindex

# # =============================================================================
# # GENERALLY VALID TEST CASES
# # =============================================================================
# N_ARRAYS = [1, 2]
# N_SAMPLE_DIMS = [1, 2]
# N_FEATURE_DIMS = [1, 2]
# INDEX_POLICY = ["index"]
# NAN_POLICY = ["no_nan"]
# DASK_POLICY = ["no_dask", "dask"]
# SEED = [0]

# VALID_TEST_DATA = [
#     (na, ns, nf, index, nan, dask)
#     for na in N_ARRAYS
#     for ns in N_SAMPLE_DIMS
#     for nf in N_FEATURE_DIMS
#     for index in INDEX_POLICY
#     for nan in NAN_POLICY
#     for dask in DASK_POLICY
# ]


# # TESTS
# # =============================================================================
# @pytest.mark.parametrize(
#     "synthetic_datalist",
#     VALID_TEST_DATA,
#     indirect=["synthetic_datalist"],
# )
# def test_transform(synthetic_datalist):
#     converter = DataListMultiIndexConverter()
#     converter.fit(synthetic_datalist)
#     transformed_data = converter.transform(synthetic_datalist)

#     is_dask_before = data_is_dask(synthetic_datalist)
#     is_dask_after = data_is_dask(transformed_data)

#     # Transforming does not affect dimensions
#     assert_expected_dims(transformed_data, synthetic_datalist, policy="all")

#     # Transforming doesn't change the dask-ness of the data
#     assert is_dask_before == is_dask_after

#     # Transforming removes MultiIndex
#     assert data_has_multiindex(transformed_data) is False

#     # Result is robust to calling the method multiple times
#     transformed_data = converter.transform(synthetic_datalist)
#     assert data_has_multiindex(transformed_data) is False

#     # Transforming data twice won't change the data
#     transformed_data2 = converter.transform(transformed_data)
#     assert data_has_multiindex(transformed_data2) is False
#     assert all(
#         trans.identical(data)
#         for trans, data in zip(transformed_data, transformed_data2)
#     )


# @pytest.mark.parametrize(
#     "synthetic_datalist",
#     VALID_TEST_DATA,
#     indirect=["synthetic_datalist"],
# )
# def test_inverse_transform(synthetic_datalist):
#     converter = DataListMultiIndexConverter()
#     converter.fit(synthetic_datalist)
#     transformed_data = converter.transform(synthetic_datalist)
#     inverse_transformed_data = converter.inverse_transform_data(transformed_data)

#     is_dask_before = data_is_dask(synthetic_datalist)
#     is_dask_after = data_is_dask(transformed_data)

#     # Transforming doesn't change the dask-ness of the data
#     assert is_dask_before == is_dask_after

#     has_multiindex_before = data_has_multiindex(synthetic_datalist)
#     has_multiindex_after = data_has_multiindex(inverse_transformed_data)

#     assert all(
#         trans.identical(data)
#         for trans, data in zip(inverse_transformed_data, synthetic_datalist)
#     )
#     assert has_multiindex_before == has_multiindex_after
