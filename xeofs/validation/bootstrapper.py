from abc import ABC, abstractmethod

import numpy as np
import xarray as xr
from tqdm import trange

from ..models import EOF, ComplexEOF
from ..utils.statistics import pearson_correlation


class _BaseBootstrapper(ABC):
    '''Bootstrap a model to obtain significant modes and confidence intervals.

    '''

    def __init__(self, n_bootstraps=20):
        self._params = {
            'n_bootstraps': n_bootstraps,
        }
    


    @abstractmethod
    def bootstrap(self, model):
        '''Bootstrap a given model.'''

        # VARIABLES ARE DEFINED IN THE SUBCLASS ACCORDING TO THE MODEL
        raise NotImplementedError
    
    def explained_variance(self):
        '''Get bootstrapped explained variance.'''

        return self._explained_variance

    def components(self):
        '''Get bootstrapped components.'''

        return self.model.stacker.inverse_transform_components(self._components)
    
    def scores(self):
        '''Get bootstrapped scores.'''

        return self.model.stacker.inverse_transform_scores(self._scores)




class EOFBootstrapper(_BaseBootstrapper):
    '''Bootstrap a model to obtain significant modes and confidence intervals.

    '''

    def __init__(self, n_bootstraps=20):
        super().__init__(n_bootstraps)
    
    def bootstrap(self, model : EOF):
        '''Bootstrap a given model.'''
        
        self.model = model

        # NOTE: not sure if we actually need a copy of the data because we reassign the dimensions
        data = self.model.data
        n_samples = data.shape[0]
        n_features = data.shape[1]

        # Replace sample and feature dimensions with indices to avoid conflicts with model implementation
        data = data.drop(('sample', 'feature'))
        data = data.assign_coords(sample=range(n_samples), feature=range(n_features))
        data = data.rename({'sample': 'sample_bst', 'feature': 'feature_bst'})
                                     
        model_params = model.get_params()
        n_modes = model_params.get('n_modes')
        n_bootstraps = self._params['n_bootstraps']

        bst_expvar = []
        bst_components = []
        bst_scores = []

        for i in trange(n_bootstraps):
            idx_rnd = np.random.choice(n_samples, n_samples, replace=True)
            bst_data = data.isel(sample_bst=idx_rnd)
            # No scaling because we used the already scaled data from the model
            bst_model = EOF(n_modes=n_modes, standardize=False, use_coslat=False, use_weights=False)
            bst_model.fit(bst_data, dim='sample_bst')
            expvar = bst_model.explained_variance()
            components = bst_model.components()
            scores = bst_model.transform(data)
            bst_expvar.append(expvar)
            bst_components.append(components)
            bst_scores.append(scores)
        
        bst_expvar = xr.concat(bst_expvar, dim='n').assign_coords(n=np.arange(1, n_bootstraps+1))
        bst_components = xr.concat(bst_components, dim='n').assign_coords(n=np.arange(1, n_bootstraps+1))
        bst_scores = xr.concat(bst_scores, dim='n').assign_coords(n=np.arange(1, n_bootstraps+1))

        bst_components = bst_components.rename({'feature_bst': 'feature'})
        bst_scores = bst_scores.rename({'sample_bst': 'sample'})


        # Re-assign original coordinates
        bst_components = bst_components.assign_coords(feature=model.stacker.coords_out_['feature'])
        bst_scores = bst_scores.assign_coords(sample=model.stacker.coords_out_['sample'])

        # NOTE: this is a bit of an ugly workaround to set the index of the DataArray. Will have to dig more 
        # into this to find a better solution
        try:
            indexes = [k for k in model.stacker.coords_out_['feature'].coords.keys() if k != 'feature']
            bst_components = bst_components.set_index(feature=indexes)
        # DataArrayListStacker does not have dims but then we don't need to set the index
        except ValueError:
            pass
        try:
            indexes = [k for k in model.stacker.coords_out_['sample'].coords.keys() if k != 'sample']
            bst_scores = bst_scores.set_index(sample=indexes)
        # DataArrayListStacker does not have dims but then we don't need to set the index
        except ValueError:
            pass

        # Fix sign of individual components determined by correlation coefficients
        # for a given mode with all the individual bootstrap members
        # NOTE: we use scores as they have typically a lower dimensionality than components
        model_scores = model._scores
        corr = (bst_scores * model_scores).mean('sample') / bst_scores.std('sample') / model_scores.std('sample')
        signs = np.sign(corr)
        bst_components = bst_components * signs
        bst_scores = bst_scores * signs
        
        self._explained_variance = bst_expvar
        self._components = bst_components
        self._scores = bst_scores


