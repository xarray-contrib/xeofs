from typing import Tuple

import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from ._base_cross_model import _BaseCrossModel
from .decomposer import CrossDecomposer
from ..utils.data_types import AnyDataObject, DataArray
from ..data_container.mca_data_container import MCADataContainer, ComplexMCADataContainer
from ..utils.statistics import pearson_correlation
from ..utils.xarray_utils import hilbert_transform


class MCA(_BaseCrossModel):
    '''Maximum Covariance Analyis (MCA).

    MCA is a statistical method that finds patterns of maximum covariance between two datasets.
    
    Parameters
    ----------
    n_modes: int, default=10
        Number of modes to calculate.
    standardize: bool, default=False
        Whether to standardize the input data.
    use_coslat: bool, default=False
        Whether to use cosine of latitude for scaling.
    use_weights: bool, default=False
        Whether to use additional weights.

    Notes
    -----
    MCA is similar to Principal Component Analysis (PCA) and Canonical Correlation Analysis (CCA), 
    but while PCA finds modes of maximum variance and CCA finds modes of maximum correlation, 
    MCA finds modes of maximum covariance. See [1]_ [2]_ for more details.

    References
    ----------
    .. [1] Bretherton, C., Smith, C., Wallace, J., 1992. An intercomparison of methods for finding coupled patterns in climate data. Journal of climate 5, 541–560.
    .. [2] Cherry, S., 1996. Singular value decomposition analysis and canonical correlation analysis. Journal of Climate 9, 2003–2009.

    Examples
    --------
    >>> model = MCA(n_modes=5, standardize=True)
    >>> model.fit(data1, data2) 
    
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attrs.update({'model': 'MCA'})

        # Initialize the DataContainer to store the results
        self.data: MCADataContainer = MCADataContainer()

    def fit(self, data1: AnyDataObject, data2: AnyDataObject, dim, weights1=None, weights2=None):
        data1_processed: DataArray = self.preprocessor1.fit_transform(data1, dim, weights1)
        data2_processed: DataArray = self.preprocessor2.fit_transform(data2, dim, weights2)

        decomposer = CrossDecomposer(n_modes=self._params['n_modes'])
        decomposer.fit(data1_processed, data2_processed)

        # Note:
        # - explained variance is given by the singular values of the SVD;
        # - We use the term singular_values_pca as used in the context of PCA:
        # Considering data X1 = X2, MCA is the same as PCA. In this case,
        # singular_values_pca is equivalent to the singular values obtained
        # when performing PCA of X1 or X2.
        singular_values = decomposer.singular_values_
        singular_vectors1 = decomposer.singular_vectors1_
        singular_vectors2 = decomposer.singular_vectors2_

        squared_covariance = singular_values**2
        total_squared_covariance = decomposer.total_squared_covariance_

        norm1 = np.sqrt(singular_values)
        norm2 = np.sqrt(singular_values)

        # Index of the sorted squared covariance
        idx_sorted_modes = squared_covariance.compute().argsort()[::-1]
        idx_sorted_modes.coords.update(squared_covariance.coords)

        # Project the data onto the singular vectors
        scores1 = xr.dot(data1_processed, singular_vectors1, dims='feature') / norm1
        scores2 = xr.dot(data2_processed, singular_vectors2, dims='feature') / norm2

        self.data.set_data(
            input_data1=data1_processed,
            input_data2=data2_processed,
            components1=singular_vectors1,
            components2=singular_vectors2,
            scores1=scores1,
            scores2=scores2,
            squared_covariance=squared_covariance,
            total_squared_covariance=total_squared_covariance,
            idx_modes_sorted=idx_sorted_modes,
            norm1=norm1,
            norm2=norm2,
        )
        # Assign analysis-relevant meta data
        self.data.set_attrs(self.attrs)

    def transform(self, **kwargs):
        '''Project new unseen data onto the singular vectors.

        Parameters
        ----------
        data1: xr.DataArray or list of xarray.DataArray
            Left input data. Must be provided if `data2` is not provided.
        data2: xr.DataArray or list of xarray.DataArray
            Right input data. Must be provided if `data1` is not provided.

        Returns
        -------
        scores1: DataArray | Dataset | List[DataArray]
            Left scores.
        scores2: DataArray | Dataset | List[DataArray]
            Right scores.

        '''
        results = []
        if 'data1' in kwargs.keys():
            # Preprocess input data
            data1 = kwargs['data1']
            data1 = self.preprocessor1.transform(data1)
            # Project data onto singular vectors
            comps1 = self.data.components1
            norm1 = self.data.norm1
            scores1 = xr.dot(data1, comps1) / norm1
            # Inverse transform scores
            scores1 = self.preprocessor1.inverse_transform_scores(scores1)
            results.append(scores1)

        if 'data2' in kwargs.keys():
            # Preprocess input data
            data2 = kwargs['data2']
            data2 = self.preprocessor2.transform(data2)
            # Project data onto singular vectors
            comps2 = self.data.components2
            norm2 = self.data.norm2
            scores2 = xr.dot(data2, comps2) / norm2
            # Inverse transform scores
            scores2 = self.preprocessor2.inverse_transform_scores(scores2)
            results.append(scores2)

        return results

    def inverse_transform(self, mode):
        '''Reconstruct the original data from transformed data.

        Parameters
        ----------
        mode: scalars, slices or array of tick labels.
            The mode(s) used to reconstruct the data. If a scalar is given,
            the data will be reconstructed using the given mode. If a slice
            is given, the data will be reconstructed using the modes in the
            given slice. If a array is given, the data will be reconstructed
            using the modes in the given array.

        Returns
        -------
        Xrec1: DataArray | Dataset | List[DataArray]
            Reconstructed data of left field.
        Xrec2: DataArray | Dataset | List[DataArray]
            Reconstructed data of right field.

        '''
        # Singular vectors
        comps1 = self.data.components1.sel(mode=mode)
        comps2 = self.data.components2.sel(mode=mode)

        # Scores = projections
        scores1 = self.data.scores1.sel(mode=mode)
        scores2 = self.data.scores2.sel(mode=mode)

        # Norms
        norm1 = self.data.norm1.sel(mode=mode)
        norm2 = self.data.norm2.sel(mode=mode)

        # Reconstruct the data
        data1 = xr.dot(scores1, comps1.conj() * norm1, dims='mode')
        data2 = xr.dot(scores2, comps2.conj() * norm2, dims='mode')

        # Enforce real output
        data1 = data1.real
        data2 = data2.real
        
        # Unstack and rescale the data
        data1 = self.preprocessor1.inverse_transform_data(data1)
        data2 = self.preprocessor2.inverse_transform_data(data2)

        return data1, data2

    def squared_covariance(self):
        '''Get the squared covariance.

        The squared covariance corresponds to the explained variance in PCA and is given by the 
        squared singular values of the covariance matrix.
            
        '''
        return self.data.squared_covariance
    
    def squared_covariance_fraction(self):
        '''Calculate the squared covariance fraction (SCF).

        The SCF is a measure of the proportion of the total squared covariance that is explained by each mode `i`. It is computed 
        as follows:

        .. math::
        SCF_i = \\frac{\\sigma_i^2}{\\sum_{i=1}^{m} \\sigma_i^2}

        where `m` is the total number of modes and :math:`\\sigma_i` is the `ith` singular value of the covariance matrix.

        '''
        return self.data.squared_covariance_fraction
    
    def singular_values(self):
        '''Get the singular values of the cross-covariance matrix.

        '''
        return self.data.singular_values

    def covariance_fraction(self):
        '''Get the covariance fraction (CF).

        Cheng and Dunkerton (1995) define the CF as follows:

        .. math::
        CF_i = \\frac{\\sigma_i}{\\sum_{i=1}^{m} \\sigma_i}

        where `m` is the total number of modes and :math:`\\sigma_i` is the `ith` singular value of the covariance matrix.

        In this implementation the sum of singular values is estimated from the first `n` modes, therefore one should aim to
        retain as many modes as possible to get a good estimate of the covariance fraction.

        Note
        ----
        It is important to differentiate the CF from the squared covariance fraction (SCF). While the SCF is an invariant quantity in MCA, the CF is not.
        Therefore, the SCF is used to assess the relative importance of each mode. Cheng and Dunkerton (1995) [1]_ introduced the CF in the context of
        Varimax-rotated MCA to compare the relative importance of each mode before and after rotation. In the special case of both data fields in MCA being identical,
        the CF is equivalent to the explained variance ratio in EOF analysis.

        References
        ----------
        .. [1] Cheng, X., Dunkerton, T.J., 1995. Orthogonal Rotation of Spatial Patterns Derived from Singular Value Decomposition Analysis. J. Climate 8, 2631–2643. https://doi.org/10.1175/1520-0442(1995)008<2631:OROSPD>2.0.CO;2


        '''
        # Check how sensitive the CF is to the number of modes
        svals = self.data.singular_values
        cf = svals[0] / svals.cumsum()
        change_per_mode = cf.shift({'mode': 1}) - cf
        change_in_cf_in_last_mode = change_per_mode.isel(mode=-1)
        if change_in_cf_in_last_mode > 0.001:
            print(f'Warning: CF is sensitive to the number of modes retained. Please increase `n_modes` for a better estimate.')
        return self.data.covariance_fraction

    def components(self):
        '''Return the singular vectors of the left and right field.
        
        Returns
        -------
        components1: DataArray | Dataset | List[DataArray]
            Left components of the fitted model.
        components2: DataArray | Dataset | List[DataArray]
            Right components of the fitted model.

        '''
        return super().components()
    
    def scores(self):
        '''Return the scores of the left and right field.

        The scores in MCA are the projection of the left and right field onto the
        left and right singular vector of the cross-covariance matrix.
        
        Returns
        -------
        scores1: DataArray
            Left scores.
        scores2: DataArray
            Right scores.

        '''
        return super().scores()

    def homogeneous_patterns(self, correction=None, alpha=0.05):
        '''Return the homogeneous patterns of the left and right field.

        The homogeneous patterns are the correlation coefficients between the 
        input data and the scores.

        More precisely, the homogeneous patterns `r_{hom}` are defined as

        .. math::
          r_{hom, x} = corr \\left(X, A_x \\right)
        .. math::
          r_{hom, y} = corr \\left(Y, A_y \\right)

        where :math:`X` and :math:`Y` are the input data, :math:`A_x` and :math:`A_y`
        are the scores of the left and right field, respectively.

        Parameters
        ----------
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

        Returns
        -------
        patterns1: DataArray | Dataset | List[DataArray]
            Left homogenous patterns.
        patterns2: DataArray | Dataset | List[DataArray]
            Right homogenous patterns.
        pvals1: DataArray | Dataset | List[DataArray]
            Left p-values.
        pvals2: DataArray | Dataset | List[DataArray]
            Right p-values.

        '''
        input_data1 = self.data.input_data1
        input_data2 = self.data.input_data2

        scores1 = self.data.scores1
        scores2 = self.data.scores2

        hom_pat1, pvals1 = pearson_correlation(input_data1, scores1, correction=correction, alpha=alpha)
        hom_pat2, pvals2 = pearson_correlation(input_data2, scores2, correction=correction, alpha=alpha)

        hom_pat1 = self.preprocessor1.inverse_transform_components(hom_pat1)
        hom_pat2 = self.preprocessor2.inverse_transform_components(hom_pat2)

        pvals1 = self.preprocessor1.inverse_transform_components(pvals1)
        pvals2 = self.preprocessor2.inverse_transform_components(pvals2)

        hom_pat1.name = 'left_homogeneous_patterns'
        hom_pat2.name = 'right_homogeneous_patterns'

        pvals1.name = 'pvalues_of_left_homogeneous_patterns'
        pvals2.name = 'pvalues_of_right_homogeneous_patterns'

        return (hom_pat1, hom_pat2), (pvals1, pvals2)

    def heterogeneous_patterns(self, correction=None, alpha=0.05):
        '''Return the heterogeneous patterns of the left and right field.
        
        The heterogeneous patterns are the correlation coefficients between the
        input data and the scores of the other field.
        
        More precisely, the heterogeneous patterns `r_{het}` are defined as
        
        .. math::
          r_{het, x} = corr \\left(X, A_y \\right)
        .. math::
          r_{het, y} = corr \\left(Y, A_x \\right)
        
        where :math:`X` and :math:`Y` are the input data, :math:`A_x` and :math:`A_y`
        are the scores of the left and right field, respectively.

        Parameters
        ----------
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
        input_data1 = self.data.input_data1
        input_data2 = self.data.input_data2

        scores1 = self.data.scores1
        scores2 = self.data.scores2

        patterns1, pvals1 = pearson_correlation(input_data1, scores2, correction=correction, alpha=alpha)
        patterns2, pvals2 = pearson_correlation(input_data2, scores1, correction=correction, alpha=alpha)

        patterns1 = self.preprocessor1.inverse_transform_components(patterns1)
        patterns2 = self.preprocessor2.inverse_transform_components(patterns2)

        pvals1 = self.preprocessor1.inverse_transform_components(pvals1)
        pvals2 = self.preprocessor2.inverse_transform_components(pvals2)

        patterns1.name = 'left_heterogeneous_patterns'
        patterns2.name = 'right_heterogeneous_patterns'

        pvals1.name = 'pvalues_of_left_heterogeneous_patterns'
        pvals2.name = 'pvalues_of_right_heterogeneous_patterns'

        return (patterns1, patterns2), (pvals1, pvals2)



class ComplexMCA(MCA):
    '''Complex Maximum Covariance Analysis (MCA). 

    Complex MCA, also referred to as Analytical SVD (ASVD) by Shane et al. (2017)[1]_, 
    enhances traditional MCA by accommodating both amplitude and phase information. 
    It achieves this by utilizing the Hilbert transform to preprocess the data, 
    thus allowing for a more comprehensive analysis in the subsequent MCA computation.

    An optional padding with exponentially decaying values can be applied prior to
    the Hilbert transform in order to mitigate the impact of spectral leakage.

    Parameters
    ----------
    n_modes: int, default=10
        Number of modes to calculate.
    standardize: bool, default=False
        Whether to standardize the input data.
    use_coslat: bool, default=False
        Whether to use cosine of latitude for scaling.
    use_weights: bool, default=False
        Whether to use additional weights.
    padding : str, optional
        Specifies the method used for padding the data prior to applying the Hilbert
        transform. This can help to mitigate the effect of spectral leakage. 
        Currently, only 'exp' for exponential padding is supported. Default is 'exp'.
    decay_factor : float, optional
        Specifies the decay factor used in the exponential padding. This parameter
        is only used if padding='exp'. The recommended value typically ranges between 0.05 to 0.2 
        but ultimately depends on the variability in the data. 
        A smaller value (e.g. 0.05) is recommended for
        data with high variability, while a larger value (e.g. 0.2) is recommended
        for data with low variability. Default is 0.2.

    Notes
    -----
    Complex MCA extends MCA to complex-valued data that contain both magnitude and phase information. 
    The Hilbert transform is used to transform real-valued data to complex-valued data, from which both 
    amplitude and phase can be extracted.

    Similar to MCA, Complex MCA is used in climate science to identify coupled patterns of variability 
    between two different climate variables. But unlike MCA, Complex MCA can identify coupled patterns 
    that involve phase shifts.

    References
    ----------
    [1]_: Elipot, S., Frajka-Williams, E., Hughes, C.W., Olhede, S., Lankhorst, M., 2017. Observed Basin-Scale Response of the North Atlantic Meridional Overturning Circulation to Wind Stress Forcing. Journal of Climate 30, 2029–2054. https://doi.org/10.1175/JCLI-D-16-0664.1

    
    Examples
    --------
    >>> model = ComplexMCA(n_modes=5, standardize=True)
    >>> model.fit(data1, data2)

    '''
    def __init__(self, padding='exp', decay_factor=.2, **kwargs):
        super().__init__(**kwargs)
        self.attrs.update({'model': 'Complex MCA'})
        self._params.update({'padding': padding, 'decay_factor': decay_factor})

        # Initialize the DataContainer to store the results
        self.data: ComplexMCADataContainer = ComplexMCADataContainer()

    def fit(self, data1: AnyDataObject, data2: AnyDataObject, dim, weights1=None, weights2=None):
        '''Fit the model.

        Parameters
        ----------
        data1: xr.DataArray or list of xarray.DataArray
            Left input data.
        data2: xr.DataArray or list of xarray.DataArray
            Right input data.
        dim: tuple
            Tuple specifying the sample dimensions. The remaining dimensions 
            will be treated as feature dimensions.
        weights1: xr.DataArray or xr.Dataset or None, default=None
            If specified, the left input data will be weighted by this array.
        weights2: xr.DataArray or xr.Dataset or None, default=None
            If specified, the right input data will be weighted by this array.

        '''

        data1_processed: DataArray = self.preprocessor1.fit_transform(data1, dim, weights2)
        data2_processed: DataArray = self.preprocessor2.fit_transform(data2, dim, weights2)
        
        # apply hilbert transform:
        padding = self._params['padding']
        decay_factor = self._params['decay_factor']
        data1_processed = hilbert_transform(data1_processed, dim='sample', padding=padding, decay_factor=decay_factor)
        data2_processed = hilbert_transform(data2_processed, dim='sample', padding=padding, decay_factor=decay_factor)
        
        decomposer = CrossDecomposer(n_modes=self._params['n_modes'])
        decomposer.fit(data1_processed, data2_processed)

        # Note:
        # - explained variance is given by the singular values of the SVD;
        # - We use the term singular_values_pca as used in the context of PCA:
        # Considering data X1 = X2, MCA is the same as PCA. In this case,
        # singular_values_pca is equivalent to the singular values obtained
        # when performing PCA of X1 or X2.
        singular_values = decomposer.singular_values_
        singular_vectors1 = decomposer.singular_vectors1_
        singular_vectors2 = decomposer.singular_vectors2_
        
        squared_covariance = singular_values**2
        total_squared_covariance = decomposer.total_squared_covariance_
        
        norm1 = np.sqrt(singular_values)
        norm2 = np.sqrt(singular_values)

        # Index of the sorted squared covariance
        idx_sorted_modes = squared_covariance.compute().argsort()[::-1]
        idx_sorted_modes.coords.update(squared_covariance.coords)

        # Project the data onto the singular vectors
        scores1 = xr.dot(data1_processed, singular_vectors1) / norm1
        scores2 = xr.dot(data2_processed, singular_vectors2) / norm2

        self.data.set_data(
            input_data1=data1_processed,
            input_data2=data2_processed,
            components1=singular_vectors1,
            components2=singular_vectors2,
            scores1=scores1,
            scores2=scores2,
            squared_covariance=squared_covariance,
            total_squared_covariance=total_squared_covariance,
            idx_modes_sorted=idx_sorted_modes,
            norm1=norm1,
            norm2=norm2,
        )
        # Assign analysis relevant meta data
        self.data.set_attrs(self.attrs)

    def components_amplitude(self) -> Tuple[AnyDataObject, AnyDataObject]:
        '''Compute the amplitude of the components.

        The amplitude of the components are defined as

        .. math::
            A_ij = |C_ij|

        where :math:`C_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
        :math:`|\\cdot|` denotes the absolute value.

        Returns
        -------
        AnyDataObject
            Amplitude of the left components.
        AnyDataObject
            Amplitude of the left components.

        '''
        comps1 = self.data.components_amplitude1
        comps2 = self.data.components_amplitude2

        comps1 = self.preprocessor1.inverse_transform_components(comps1)
        comps2 = self.preprocessor2.inverse_transform_components(comps2)

        return (comps1, comps2)

    def components_phase(self) -> Tuple[AnyDataObject, AnyDataObject]:
        '''Compute the phase of the components.

        The phase of the components are defined as

        .. math::
            \\phi_{ij} = \\arg(C_{ij})

        where :math:`C_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
        :math:`\\arg(\\cdot)` denotes the argument of a complex number.

        Returns
        -------
        AnyDataObject
            Phase of the left components.
        AnyDataObject
            Phase of the right components.

        '''
        comps1 = self.data.components_phase1
        comps2 = self.data.components_phase2

        comps1 = self.preprocessor1.inverse_transform_components(comps1)
        comps2 = self.preprocessor2.inverse_transform_components(comps2)

        return (comps1, comps2)
    
    def scores_amplitude(self) -> Tuple[DataArray, DataArray]:
        '''Compute the amplitude of the scores.

        The amplitude of the scores are defined as

        .. math::
            A_ij = |S_ij|

        where :math:`S_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
        :math:`|\\cdot|` denotes the absolute value.

        Returns
        -------
        DataArray
            Amplitude of the left scores.
        DataArray
            Amplitude of the right scores.

        '''
        scores1 = self.data.scores_amplitude1
        scores2 = self.data.scores_amplitude2

        scores1 = self.preprocessor1.inverse_transform_scores(scores1)
        scores2 = self.preprocessor2.inverse_transform_scores(scores2)
        return (scores1, scores2)
    
    def scores_phase(self) -> Tuple[DataArray, DataArray]:
        '''Compute the phase of the scores.

        The phase of the scores are defined as
        
        .. math::
            \\phi_{ij} = \\arg(S_{ij})

        where :math:`S_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
        :math:`\\arg(\\cdot)` denotes the argument of a complex number.

        Returns
        -------
        DataArray
            Phase of the left scores.
        DataArray
            Phase of the right scores.

        '''
        scores1 = self.data.scores_phase1
        scores2 = self.data.scores_phase2

        scores1 = self.preprocessor1.inverse_transform_scores(scores1)
        scores2 = self.preprocessor2.inverse_transform_scores(scores2)

        return (scores1, scores2)

    
    def transform(self, data1: AnyDataObject, data2: AnyDataObject):
        raise NotImplementedError('Complex MCA does not support transform method.')

    def homogeneous_patterns(self, correction=None, alpha=0.05):
        raise NotImplementedError('Complex MCA does not support homogeneous_patterns method.')
    
    def heterogeneous_patterns(self, correction=None, alpha=0.05):
        raise NotImplementedError('Complex MCA does not support heterogeneous_patterns method.')

