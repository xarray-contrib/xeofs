import numpy as np

from xeofs.models._base_scaler import _BaseScaler
from ..utils.constants import VALID_LATITUDE_NAMES



class Scaler(_BaseScaler):
    
    def fit(self, da):
        dim_sample = self._params['dims']['sample']
        dim_feature = self._params['dims']['feature']
        if self._params['with_mean']:
            self.mean = da.mean(dim_sample)
        if self._params['with_std']:
            self.std = da.std(dim_sample)
        if self._params['with_coslat']:
            self.coslat_weights = self._compute_coslat_weights(da, dim_feature)

    def transform(self, da):
        if self._params['with_copy']:
            da = da.copy(deep=True)
        if self._params['with_mean']:
            da -= self.mean
        if self._params['with_std']:
            da /= self.std
        if self._params['with_coslat']:
            da *= self.coslat_weights
        if self._params['with_weights']:
            da *= self.weights
        return da
    
    def inverse_transform(self, da):
        if self._params['with_copy']:
            da = da.copy(deep=True)
        if self._params['with_weights']:
            da /= self.weights
        if self._params['with_coslat']:
            da /= self.coslat_weights
        if self._params['with_std']:
            da *= self.std
        if self._params['with_mean']:
            da += self.mean
        return da

    def _compute_coslat_weights(self, da, dim):
        # Find latitude coordinate
        is_lat_coord = np.isin(dim, VALID_LATITUDE_NAMES)

        # Select latitude coordinate and compute coslat weights
        lat_coord = np.array(dim)[is_lat_coord]
        
        if len(lat_coord) > 1:
            raise ValueError(f'{lat_coord} are ambiguous latitude coordinates. Only ONE of the following is allowed for computing coslat weights: {VALID_LATITUDE_NAMES}')

        if len(lat_coord) == 1:
            weights = np.sqrt(np.cos(np.deg2rad(da.coords[lat_coord[0]]))).clip(0, 1)
            # Features that cannot be associated to a latitude receive a weight of 1
            weights = weights.where(weights.notnull(), 1)
        else:
            raise ValueError('No latitude coordinate was found to compute coslat weights. Must be one of the following: {:}'.format(VALID_LATITUDE_NAMES))
        weights.name = 'coslat_weights'
        return weights


class ListScaler(_BaseScaler):
    def __init__(self, sample_dims, feature_dims, with_copy=True, with_mean=True, with_std=True, with_coslat=False, weights=None):
        super().__init__(sample_dims, feature_dims, with_copy=with_copy, with_mean=with_mean, with_std=with_std, with_coslat=with_coslat, weights=weights)
        self.scalers = None

    def fit(self, da_list):
        self.scalers = []
        for da in da_list:
            scaler = Scaler(sample_dims=self._params['dims']['sample'], feature_dims=self._params['dims']['feature'], with_copy=self._params['with_copy'], with_mean=self._params['with_mean'], with_std=self._params['with_std'], with_coslat=self._params['with_coslat'], weights=self._params['with_weights'])
            scaler.fit(da)
            self.scalers.append(scaler)

    def transform(self, da_list):
        if self._params['with_copy']:
            da_list = [da.copy(deep=True) for da in da_list]
        for scaler, da in zip(self.scalers, da_list):
            scaler.transform(da)
        return da_list
    
    def inverse_transform(self, da_list):
        if self._params['with_copy']:
            da_list = [da.copy(deep=True) for da in da_list]
        for scaler, da in zip(self.scalers, da_list):
            scaler.inverse_transform(da)
        return da_list