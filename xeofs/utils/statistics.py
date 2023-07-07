from typing import Tuple

import xarray as xr
import scipy as sc
from statsmodels.stats.multitest import multipletests as statsmodels_multipletests

from .constants import MULTIPLE_TESTS

def pearson_correlation(data1, data2, correction=None, alpha=0.05):
    """Compute Pearson correlation between two xarray objects.
     
    Additionally, compute two-sided p-values and adjust them for multiple testing.
     
    Parameters
    ----------
    data1 : xr.DataArray
         First data array.
    data2 : xr.DataArray
        Second data array.
    correction : Optional, str
        Method used for testing and adjustment of pvalues. Can be
            either the full name or initial letters.  Available methods are:
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
    alpha : float
        The desired family-wise error rate.
        
    Returns
    -------
    corr : xr.DataArray
        Pearson correlation between the original data and the scores.
    pvalues : xr.DataArray
        Adjusted p-values for the Pearson correlation.

    """
    n_samples = data1.shape[0]

    # Compute Pearson correlation coefficients
    corr = (data1 * data2).mean('sample') / data1.std('sample') / data2.std('sample')
    
    # Compute two-sided p-values
    pvals = _compute_pvalues(corr, n_samples, dims=['feature'])
    
    if correction is not None:
        # Adjust p-values for multiple testing
        rejected, pvals = _multipletests(pvals, dim='feature', alpha=alpha, method=correction)
        return corr, pvals

    else:
        return corr, pvals


def _compute_pvalues(pearsonr, n_samples: int, dims) -> xr.DataArray:
        # Compute two-sided p-values
        # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html#r8c6348c62346-1
        a = n_samples / 2 - 1
        dist = sc.stats.beta(a, a, loc=-1, scale=2)  # type: ignore
        pvals = 2 * xr.apply_ufunc(
            dist.cdf,
            -abs(pearsonr),
            input_core_dims=[dims],
            output_core_dims=[dims],
            dask='allowed',
            vectorize=False
        )
        return pvals

def _multipletests(p, dim, alpha=0.05, method=None, **multipletests_kwargs):
    """Apply statsmodels.stats.multitest.multipletests for 2-dimensional
    xr.objects.

    Parameters
    ----------
    p: xr.object
        uncorrected p-values.
    alpha: (optional float)
        FWER, family-wise error rate. Defaults to 0.05.
    method: str
        Method used for testing and adjustment of pvalues. Can be
            either the full name or initial letters.  Available methods are:
            - bonferroni : one-step correction
            - sidak : one-step correction
            - holm-sidak : step down method using Sidak adjustments
            - holm : step-down method using Bonferroni adjustments
            - simes-hochberg : step-up method (independent)
            - hommel : closed method based on Simes tests (non-negative)
            - fdr_bh : Benjamini/Hochberg (non-negative)
            - fdr_by : Benjamini/Yekutieli (negative)
            - fdr_tsbh : two stage fdr correction (non-negative)
            - fdr_tsbky : two stage fdr correction (non-negative)
    **multipletests_kwargs (optional dict): is_sorted, returnsorted
        see statsmodels.stats.multitest.multitest

    Returns:
        reject (xr.object): true for hypothesis that can be rejected for given
            alpha
        pvals_corrected (xr.object): p-values corrected for multiple tests

    """
    if method is None:
        raise ValueError(
            f"Please indicate a method using the 'method=...' keyword. "
            f"Select from {MULTIPLE_TESTS}"
        )
    elif method not in MULTIPLE_TESTS:
        raise ValueError(
            f"Your method '{method}' is not in the accepted methods: {MULTIPLE_TESTS}"
        )

    rej, pvals_corr, *_ = xr.apply_ufunc(
        statsmodels_multipletests,
        p,
        input_core_dims=[[dim]],
        output_core_dims=[[dim], [dim], [], []],
        dask='allowed',
        vectorize=True,
    )
    return rej, pvals_corr

