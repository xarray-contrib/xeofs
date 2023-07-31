from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

import numpy as np
import xarray as xr
from tqdm import trange

from ..models import EOF
from ..data_container.eof_bootstrapper_data_container import EOFBootstrapperDataContainer
from .._version import __version__


class _BaseBootstrapper(ABC):
    '''Bootstrap a model to obtain significant modes and confidence intervals.

    '''

    def __init__(self, n_bootstraps=20, seed=None):
        self._params = {
            'n_bootstraps': n_bootstraps,
            'seed': seed,
        }

        # Define analysis-relevant meta data
        self.attrs: Dict[str, Any] = {'model': 'BaseBootstrapper'}
        self.attrs.update(self._params)
        self.attrs.update({
            'software': 'xeofs',
            'version': __version__,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })


    @abstractmethod
    def fit(self, model):
        '''Bootstrap a given model.'''

        # VARIABLES ARE DEFINED IN THE SUBCLASS ACCORDING TO THE MODEL
        raise NotImplementedError


class EOFBootstrapper(_BaseBootstrapper, EOF):
    '''Bootstrap a model to obtain significant modes and confidence intervals.

    '''

    def __init__(self, n_bootstraps=20, seed=None):
        # Call the constructor of _BaseBootstrapper
        super().__init__(
            n_bootstraps=n_bootstraps,
            seed=seed
        )
        self.attrs.update({'model': 'Bootstrapped EOF analysis'})

        # Initialize the DataContainer to store the results
        self.data: EOFBootstrapperDataContainer = EOFBootstrapperDataContainer()
    
    def fit(self, model : EOF):
        '''Bootstrap a given model.'''
        
        self.model = model
        self.preprocessor = model.preprocessor

        # NOTE: not sure if we actually need a copy of the data because we reassign the dimensions
        input_data = model.data.input_data
        n_samples = input_data.sample.size
        n_features = input_data.feature.size

        # Replace sample and feature dimensions with indices to avoid conflicts with model implementation
        input_data = input_data.drop_vars(['sample', 'feature'])
        # use assign_coords instead of update to create a copy of the data, so that
        # we don't modify the original data
        input_data = input_data.assign_coords(sample=range(n_samples), feature=range(n_features))
        input_data = input_data.rename({'sample': 'sample_bst', 'feature': 'feature_bst'})
                                     
        model_params = model.get_params()
        n_modes = model_params.get('n_modes')
        n_bootstraps = self._params['n_bootstraps']

        # Set seed for reproducibility
        rng = np.random.default_rng(self._params['seed'])

        # Bootstrap the model
        bst_expvar = []
        bst_total_variance = []
        bst_components = []
        bst_scores = []
        bst_idx_modes_sorted = []
        for i in trange(n_bootstraps):
            # Sample with replacement
            idx_rnd = rng.choice(n_samples, n_samples, replace=True)
            bst_data = input_data.isel(sample_bst=idx_rnd)
            # Perform EOF analysis with the subsampled data
            # No scaling because we use the pre-scaled data from the model
            bst_model = EOF(n_modes=n_modes, standardize=False, use_coslat=False, use_weights=False)
            bst_model.fit(bst_data, dim='sample_bst')
            # Save results
            expvar = bst_model.data.explained_variance
            totvar = bst_model.data.total_variance
            idx_modes_sorted = bst_model.data.idx_modes_sorted
            components = bst_model.data.components
            scores = bst_model.transform(input_data)
            bst_expvar.append(expvar)
            bst_total_variance.append(totvar)
            bst_idx_modes_sorted.append(idx_modes_sorted)
            bst_components.append(components)
            bst_scores.append(scores)
        
        bst_expvar = xr.concat(bst_expvar, dim='n').assign_coords(n=np.arange(1, n_bootstraps+1))
        bst_total_variance = xr.concat(bst_total_variance, dim='n').assign_coords(n=np.arange(1, n_bootstraps+1))
        bst_idx_modes_sorted = xr.concat(bst_idx_modes_sorted, dim='n').assign_coords(n=np.arange(1, n_bootstraps+1))
        bst_components = xr.concat(bst_components, dim='n').assign_coords(n=np.arange(1, n_bootstraps+1))
        bst_scores = xr.concat(bst_scores, dim='n').assign_coords(n=np.arange(1, n_bootstraps+1))

        # Rename dimensions only for scores, because `transform` returned the "unstacked" version with "sample_bst" dimension
        # bst_components = bst_components.rename({'feature_bst': 'feature'})
        bst_scores = bst_scores.rename({'sample_bst': 'sample'})


        # Re-assign original coordinates
        bst_components = bst_components.assign_coords(feature=self.preprocessor.stacker.coords_out_['feature'])
        bst_scores = bst_scores.assign_coords(sample=self.preprocessor.stacker.coords_out_['sample'])

        # NOTE: this is a bit of an ugly workaround to set the index of the DataArray. Will have to dig more 
        # into this to find a better solution
        try:
            indexes = [k for k in self.preprocessor.stacker.coords_out_['feature'].coords.keys() if k != 'feature']
            bst_components = bst_components.set_index(feature=indexes)
        # ListDataArrayStacker does not have dims but then we don't need to set the index
        except ValueError:
            pass
        try:
            indexes = [k for k in self.preprocessor.stacker.coords_out_['sample'].coords.keys() if k != 'sample']
            bst_scores = bst_scores.set_index(sample=indexes)
        # ListDataArrayStacker does not have dims but then we don't need to set the index
        except ValueError:
            pass

        # Fix sign of individual components determined by correlation coefficients
        # for a given mode with all the individual bootstrap members
        # NOTE: we use scores as they have typically a lower dimensionality than components
        model_scores = model.data.scores
        corr = (bst_scores * model_scores).mean('sample') / bst_scores.std('sample') / model_scores.std('sample')
        signs = np.sign(corr)
        bst_components = bst_components * signs
        bst_scores = bst_scores * signs
        

        self.data.set_data(
            input_data=self.model.data.input_data,
            components=bst_components,
            scores=bst_scores,
            explained_variance=bst_expvar,
            total_variance=bst_total_variance,
            idx_modes_sorted=bst_idx_modes_sorted,
        )
        # Assign the same attributes as the original model
        self.data.set_attrs(self.attrs)



