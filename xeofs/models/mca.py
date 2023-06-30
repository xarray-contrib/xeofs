import numpy as np
import xarray as xr

from ._base_cross_model import _BaseCrossModel
from .decomposer import CrossDecomposer
from ..utils.data_types import XarrayData, DataArrayList
from ..utils.statistics import pearson_correlation


class MCA(_BaseCrossModel):

    def fit(self, data1: XarrayData | DataArrayList, data2: XarrayData | DataArrayList, dims, weights1=None, weights2=None):
        '''
        Fit the model.

        Parameters:
        -------------
        data1: xr.DataArray or list of xarray.DataArray
            Left input data.
        data2: xr.DataArray or list of xarray.DataArray
            Right input data.
        dims: tuple
            Tuple specifying the sample dimensions. The remaining dimensions 
            will be treated as feature dimensions.
        weights1: xr.DataArray or xr.Dataset or None, default=None
            If specified, the left input data will be weighted by this array.
        weights2: xr.DataArray or xr.Dataset or None, default=None
            If specified, the right input data will be weighted by this array.

        '''
        self._preprocessing(data1, data2, dims, weights1, weights2)

        decomposer = CrossDecomposer(n_modes=self._params['n_modes'])
        decomposer.fit(self.data1, self.data2)

        # Note:
        # - explained variance is given by the singular values of the SVD;
        # - We use the term singular_values_pca as used in the context of PCA:
        # Considering data X1 = X2, MCA is the same as PCA. In this case,
        # singular_values_pca is equivalent to the singular values obtained
        # when performing PCA of X1 or X2.
        self._singular_values = decomposer.singular_values_
        self._explained_variance = decomposer.singular_values_  # actually covariance
        self._squared_total_variance = decomposer.squared_total_variance_
        self._squared_covariance_fraction = self._explained_variance**2 / self._squared_total_variance
        self._singular_values_pca = np.sqrt(self._singular_values * (self.data1.shape[0] - 1))
        self._singular_vectors1 = decomposer.singular_vectors1_
        self._singular_vectors2 = decomposer.singular_vectors2_
        self._norm1 = np.sqrt(self._singular_values)
        self._norm2 = np.sqrt(self._singular_values)

        self._explained_variance.name = 'explained_variance'
        self._squared_total_variance.name = 'squared_total_variance'
        self._squared_covariance_fraction.name = 'squared_covariance_fraction'
        self._norm1.name = 'left_norm'
        self._norm2.name = 'right_norm'

        # Project the data onto the singular vectors
        sqrt_expvar = np.sqrt(self._explained_variance)
        self._scores1 = xr.dot(self.data1, self._singular_vectors1) / sqrt_expvar
        self._scores2 = xr.dot(self.data2, self._singular_vectors2) / sqrt_expvar

    def transform(self, **kwargs):
        '''Project new unseen data onto the singular vectors.

        Parameters:
        -------------
        data1: xr.DataArray or list of xarray.DataArray
            Left input data. Must be provided if `data2` is not provided.
        data2: xr.DataArray or list of xarray.DataArray
            Right input data. Must be provided if `data1` is not provided.

        Returns:
        ----------
        scores1: DataArray | Dataset | List[DataArray]
            Left scores.
        scores2: DataArray | Dataset | List[DataArray]
            Right scores.

        '''
        results = []
        if 'data1' in kwargs:
            data1 = kwargs['data1']
            data1 = self.scaler1.transform(data1)
            data1 = self.stacker1.transform(data1) # type: ignore
            scores1 = xr.dot(data1, self._singular_vectors1) / self._norm1
            scores1 = self.stacker1.inverse_transform_scores(scores1)
            results.append(scores1)

        if 'data2' in kwargs:
            data2 = kwargs['data2']
            data2 = self.scaler2.transform(data2)
            data2 = self.stacker2.transform(data2) # type: ignore
            scores2 = xr.dot(data2, self._singular_vectors2) / self._norm2
            scores2 = self.stacker2.inverse_transform_scores(scores2)
            results.append(scores2)

        return results

    def inverse_transform(self, mode):
        '''Reconstruct the original data from transformed data.

        Parameters:
        -------------
        mode: scalars, slices or array of tick labels.
            The mode(s) used to reconstruct the data. If a scalar is given,
            the data will be reconstructed using the given mode. If a slice
            is given, the data will be reconstructed using the modes in the
            given slice. If a array is given, the data will be reconstructed
            using the modes in the given array.

        Returns:
        ----------
        Xrec1: DataArray | Dataset | List[DataArray]
            Reconstructed data of left field.
        Xrec2: DataArray | Dataset | List[DataArray]
            Reconstructed data of right field.

        '''
        svecs1 = self._singular_vectors1.sel(mode=mode)  # type: ignore
        svecs2 = self._singular_vectors2.sel(mode=mode)  # type: ignore

        scores1 = self._scores1.sel(mode=mode)  # type: ignore
        scores2 = self._scores2.sel(mode=mode)  # type: ignore

        Xrec1 = svecs1.dot(scores1.conj().T)
        Xrec2 = svecs2.dot(scores2.conj().T)

        Xrec1 = self.stacker1.inverse_transform_data(Xrec1)
        Xrec2 = self.stacker2.inverse_transform_data(Xrec2)

        Xrec1 = self.scaler1.inverse_transform(Xrec1)  # type: ignore
        Xrec2 = self.scaler2.inverse_transform(Xrec2)  # type: ignore

        return Xrec1, Xrec2

    def singular_values(self):
        '''Return the singular values of the model.

        Returns:
        ----------
        singular_values: DataArray
            Singular values of the fitted model.

        '''
        return self._singular_values
    
    def explained_variance(self):
        '''Return explained variance.'''
        return self._explained_variance
    
    def squared_covariance_fraction(self):
        '''Return the squared covariance fraction.
        
        The squared covariance fraction (SCF) is the ratio of the squared covariance
        to the total squared variance and is a measure of importance of the
        mode. It is the analogue of the explained variance ratio in PCA.

        The SCF for mode `i` is given by:

        .. math::
            SCF_i = \\frac{\\sigma_i^2}{\\sum_{i=1}^{n} \\sigma_i^2}

        where :math:`\\sigma_i` is the singular value of mode `i`.

        '''
        return self._squared_covariance_fraction
    
    def components(self):
        '''Return the singular vectors of the left and right field.
        
        Returns:
        ----------
        components1: DataArray | Dataset | List[DataArray]
            Left components of the fitted model.
        components2: DataArray | Dataset | List[DataArray]
            Right components of the fitted model.

        '''
        svecs1 = self.stacker1.inverse_transform_components(self._singular_vectors1)
        svecs2 = self.stacker2.inverse_transform_components(self._singular_vectors2)
        return svecs1, svecs2
    
    def scores(self):
        '''Return the scores of the left and right field.

        The scores in MCA are the projection of the data matrix onto the
        singular vectors of the cross-covariance matrix.
        
        Returns:
        ----------
        scores1: DataArray | Dataset | List[DataArray]
            Left scores.
        scores2: DataArray | Dataset | List[DataArray]
            Right scores.

        '''
        scores1 = self.stacker1.inverse_transform_scores(self._scores1)
        scores2 = self.stacker2.inverse_transform_scores(self._scores2)
        return scores1, scores2

    def homogeneous_patterns(self, correction=None, alpha=0.05):
        '''Return the homogeneous patterns of the left and right field.

        The homogeneous patterns are the correlation coefficients between the 
        input data and the scores.

        More precisely, the homogeneous patterns `r_{hom}` are defined as

        .. math::
          r_{hom, x} = \\corr \\left(X, A_x \\right)
        .. math::
          r_{hom, y} = \\corr \\left(Y, A_y \\right)

        where :math:`X` and :math:`Y` are the input data, :math:`A_x` and :math:`A_y`
        are the scores of the left and right field, respectively.

        Parameters:
        -------------
        correction: str, default=None
            Method to apply a multiple testing correction. If None, no correction
            is applied.  Available methods are:
            - bonferroni : one-step correction
            - sidak : one-step correction
            - holm-sidak : step down method using Sidak adjustments
            - holm : step-down method using Bonferroni adjustments
            - simes-hochberg : step-up method (independent)
            - hommel : closed method based on Simes tests (non-negative)
            - fdr_bh : Benjamini/Hochberg (non-negative) (default)
            - fdr_by : Benjamini/Yekutieli (negative)
            - fdr_tsbh : two stage fdr correction (non-negative)
            - fdr_tsbky : two stage fdr correction (non-negative)
        alpha: float, default=0.05
            The desired family-wise error rate. Not used if `correction` is None.

        Returns:
        ----------
        patterns1: DataArray | Dataset | List[DataArray]
            Left homogenous patterns.
        patterns2: DataArray | Dataset | List[DataArray]
            Right homogenous patterns.
        pvals1: DataArray | Dataset | List[DataArray]
            Left p-values.
        pvals2: DataArray | Dataset | List[DataArray]
            Right p-values.

        '''
        patterns1, pvals1 = pearson_correlation(self.data1, self._scores1, correction=correction, alpha=alpha)
        patterns2, pvals2 = pearson_correlation(self.data2, self._scores2, correction=correction, alpha=alpha)

        patterns1 = self.stacker1.inverse_transform_data(self._singular_vectors1)
        patterns2 = self.stacker2.inverse_transform_data(self._singular_vectors2)

        return patterns1, patterns2, pvals1, pvals2

    def heterogeneous_patterns(self, correction=None, alpha=0.05):
        '''Return the heterogeneous patterns of the left and right field.
        
        The heterogeneous patterns are the correlation coefficients between the
        input data and the scores of the other field.
        
        More precisely, the heterogeneous patterns `r_{het}` are defined as
        
        .. math::
          r_{het, x} = \\corr \\left(X, A_y \\right)
        .. math::
          r_{het, y} = \\corr \\left(Y, A_x \\right)
        
        where :math:`X` and :math:`Y` are the input data, :math:`A_x` and :math:`A_y`
        are the scores of the left and right field, respectively.

        Parameters:
        -------------
        correction: str, default=None
            Method to apply a multiple testing correction. If None, no correction
            is applied.  Available methods are: 
            - bonferroni : one-step correction
            - sidak : one-step correction
            - holm-sidak : step down method using Sidak adjustments
            - holm : step-down method using Bonferroni adjustments
            - simes-hochberg : step-up method (independent)
            - hommel : closed method based on Simes tests (non-negative)
            - fdr_bh : Benjamini/Hochberg (non-negative) (default)
            - fdr_by : Benjamini/Yekutieli (negative)
            - fdr_tsbh : two stage fdr correction (non-negative)
            - fdr_tsbky : two stage fdr correction (non-negative)
        alpha: float, default=0.05
            The desired family-wise error rate. Not used if `correction` is None.

        '''
        patterns1, pvals1 = pearson_correlation(self.data1, self._scores2, correction=correction, alpha=alpha)
        patterns2, pvals2 = pearson_correlation(self.data2, self._scores1, correction=correction, alpha=alpha)

        patterns1 = self.stacker1.inverse_transform_data(self._singular_vectors2)
        patterns2 = self.stacker2.inverse_transform_data(self._singular_vectors1)

        return patterns1, patterns2, pvals1, pvals2

    def compute(self):
        '''Computing the model will compute and load all DaskArrays.'''

        self._singular_values = self._singular_values.compute()
        self._explained_variance = self._explained_variance.compute()
        self._squared_total_variance = self._squared_total_variance.compute()
        self._squared_covariance_fraction = self._squared_covariance_fraction.compute()
        self._singular_vectors1 = self._singular_vectors1.compute()
        self._singular_vectors2 = self._singular_vectors2.compute()
        self._norm1 = self._norm1.compute()
        self._norm2 = self._norm2.compute()
        self._scores1 = self._scores1.compute()
        self._scores2 = self._scores2.compute()