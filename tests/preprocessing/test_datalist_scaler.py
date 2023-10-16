# import pytest
# import xarray as xr
# import numpy as np

# from xeofs.preprocessing.scaler import DataListScaler
# from xeofs.utils.data_types import DimsList


# @pytest.mark.parametrize(
#     "with_std, with_coslat",
#     [
#         (True, True),
#         (True, False),
#         (False, True),
#         (False, False),
#     ],
# )
# def test_fit_params(with_std, with_coslat, mock_data_array_list):
#     listscalers = DataListScaler(with_std=with_std, with_coslat=with_coslat)
#     data = mock_data_array_list.copy()
#     sample_dims = ["time"]
#     feature_dims: DimsList = [["lat", "lon"]] * 3
#     size_lats_list = [da.lat.size for da in data]
#     weights = [
#         xr.DataArray(np.random.rand(size), dims=["lat"]) for size in size_lats_list
#     ]
#     listscalers.fit(mock_data_array_list, sample_dims, feature_dims, weights)

#     for s in listscalers.scalers:
#         assert hasattr(s, "mean_"), "Scaler has no mean attribute."
#         if with_std:
#             assert hasattr(s, "std_"), "Scaler has no std attribute."
#         if with_coslat:
#             assert hasattr(
#                 s, "coslat_weights_"
#             ), "Scaler has no coslat_weights attribute."

#         assert s.mean_ is not None, "Scaler mean is None."
#         if with_std:
#             assert s.std_ is not None, "Scaler std is None."
#         if with_coslat:
#             assert s.coslat_weights_ is not None, "Scaler coslat_weights is None."


# @pytest.mark.parametrize(
#     "with_std, with_coslat, with_weights",
#     [
#         (True, True, True),
#         (True, False, True),
#         (False, True, True),
#         (False, False, True),
#         (True, True, False),
#         (True, False, False),
#         (False, True, False),
#         (False, False, False),
#     ],
# )
# def test_transform_params(with_std, with_coslat, with_weights, mock_data_array_list):
#     listscalers = DataListScaler(with_std=with_std, with_coslat=with_coslat)
#     data = mock_data_array_list.copy()
#     sample_dims = ["time"]
#     feature_dims: DimsList = [("lat", "lon")] * 3
#     size_lats_list = [da.lat.size for da in data]
#     if with_weights:
#         weights = [
#             xr.DataArray(np.random.rand(size), dims=["lat"]) for size in size_lats_list
#         ]
#     else:
#         weights = None
#     listscalers.fit(
#         mock_data_array_list,
#         sample_dims,
#         feature_dims,
#         weights,
#     )

#     transformed = listscalers.transform(mock_data_array_list)
#     transformed2 = listscalers.fit_transform(
#         mock_data_array_list, sample_dims, feature_dims, weights
#     )

#     for t, t2, s, ref in zip(transformed, transformed2, listscalers.scalers, data):
#         assert t is not None, "Transformed data is None."

#         t_mean = t.mean(sample_dims, skipna=False)
#         assert np.allclose(t_mean, 0), "Mean of the transformed data is not zero."

#         if with_std:
#             t_std = t.std(sample_dims, skipna=False)
#             if with_coslat or with_weights:
#                 assert (
#                     t_std <= 1
#                 ).all(), "Standard deviation of the transformed data is larger one."
#             else:
#                 assert np.allclose(
#                     t_std, 1
#                 ), "Standard deviation of the transformed data is not one."

#         if with_coslat:
#             assert s.coslat_weights_ is not None, "Scaler coslat_weights is None."
#             assert not np.array_equal(
#                 t, mock_data_array_list
#             ), "Data has not been transformed."

#         xr.testing.assert_allclose(t, t2)


# @pytest.mark.parametrize(
#     "with_std, with_coslat",
#     [
#         (True, True),
#         (True, False),
#         (False, True),
#         (False, False),
#     ],
# )
# def test_inverse_transform_params(with_std, with_coslat, mock_data_array_list):
#     listscalers = DataListScaler(
#         with_std=with_std,
#         with_coslat=with_coslat,
#     )
#     data = mock_data_array_list.copy()
#     sample_dims = ["time"]
#     feature_dims: DimsList = [["lat", "lon"]] * 3
#     size_lats_list = [da.lat.size for da in data]
#     weights = [
#         xr.DataArray(np.random.rand(size), dims=["lat"]) for size in size_lats_list
#     ]
#     listscalers.fit(mock_data_array_list, sample_dims, feature_dims, weights)
#     transformed = listscalers.transform(mock_data_array_list)
#     inverted = listscalers.inverse_transform_data(transformed)

#     # check that inverse transform is the same as the original data
#     for inv, ref in zip(inverted, mock_data_array_list):
#         xr.testing.assert_allclose(inv, ref)


# @pytest.mark.parametrize(
#     "dim_sample, dim_feature",
#     [
#         (("time",), ("lat", "lon")),
#         (("time",), ("lon", "lat")),
#         (("lat", "lon"), ("time",)),
#         (("lon", "lat"), ("time",)),
#     ],
# )
# def test_fit_dims(dim_sample, dim_feature, mock_data_array_list):
#     listscalers = DataListScaler(with_std=True)
#     data = mock_data_array_list.copy()
#     dim_feature = [dim_feature] * 3

#     for s in listscalers.scalers:
#         assert hasattr(s, "mean"), "Scaler has no mean attribute."
#         assert s.mean is not None, "Scaler mean is None."
#         assert hasattr(s, "std"), "Scaler has no std attribute."
#         assert s.std is not None, "Scaler std is None."
#         # check that all dimensions are present except the sample dimensions
#         assert set(s.mean.dims) == set(mock_data_array_list.dims) - set(
#             dim_sample
#         ), "Mean has wrong dimensions."
#         assert set(s.std.dims) == set(mock_data_array_list.dims) - set(
#             dim_sample
#         ), "Standard deviation has wrong dimensions."


# @pytest.mark.parametrize(
#     "dim_sample, dim_feature",
#     [
#         (("time",), ("lat", "lon")),
#         (("time",), ("lon", "lat")),
#         (("lat", "lon"), ("time",)),
#         (("lon", "lat"), ("time",)),
#     ],
# )
# def test_fit_transform_dims(dim_sample, dim_feature, mock_data_array_list):
#     listscalers = DataListScaler(with_std=True)
#     data = mock_data_array_list.copy()
#     dim_feature = [dim_feature] * 3
#     transformed = listscalers.fit_transform(
#         mock_data_array_list, dim_sample, dim_feature
#     )

#     for trns, ref in zip(transformed, mock_data_array_list):
#         # check that all dimensions are present
#         assert set(trns.dims) == set(ref.dims), "Transformed data has wrong dimensions."
#         # check that the coordinates are the same
#         for dim in ref.dims:
#             xr.testing.assert_allclose(trns[dim], ref[dim])


# # Test input types
# @pytest.mark.parametrize(
#     "dim_sample, dim_feature",
#     [
#         (("time",), ("lat", "lon")),
#         (("time",), ("lon", "lat")),
#         (("lat", "lon"), ("time",)),
#         (("lon", "lat"), ("time",)),
#     ],
# )
# def test_fit_input_type(
#     dim_sample, dim_feature, mock_data_array, mock_dataset, mock_data_array_list
# ):
#     s = DataListScaler()
#     dim_feature = [dim_feature] * 3
#     with pytest.raises(TypeError):
#         s.fit(mock_dataset, dim_sample, dim_feature)
#     with pytest.raises(TypeError):
#         s.fit(mock_data_array, dim_sample, dim_feature)

#     s.fit(mock_data_array_list, dim_sample, dim_feature)
#     with pytest.raises(TypeError):
#         s.transform(mock_dataset)
#     with pytest.raises(TypeError):
#         s.transform(mock_data_array)
