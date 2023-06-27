from typing import Tuple

import xarray as xr
from statsmodels.stats.multitest import multipletests as statsmodels_multipletests
from ..models import EOF
from ..models.rotator import EOFRotator

from .constants import MULTIPLE_TESTS

def compute_components_as_correlation(model: EOF | EOFRotator, alpha=0.05, method='fdr_bh') -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute the components as Pearson correlation between the original data and the
    scores.

    Parameters
    ----------
    model : EOF | Rotator
        A EOF model solution.
    alpha : float
        The desired family-wise error rate.
    method : str
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

    Returns
    -------
    corr : xr.DataArray
        Pearson correlation between the original data and the scores.
    pvalues : xr.DataArray
        Adjusted p-values for the Pearson correlation.
    rejected : xr.DataArray
        Boolean array of the same shape as ``pvalues``, where ``True`` indicates
        a rejected hypothesis.

    """
    if isinstance(model, EOFRotator):
        scores = model._model._scores
        data = model._model.data
    else:
        scores = model._scores
        data = model.data
    
    # Compute Pearson correlation coefficients
    corr = (data * scores).mean('sample') / data.std('sample') / scores.std('sample')

    # Compute two-sided p-values
    pvals = _compute_pvalues(corr, data.shape[0], dims=['mode', 'feature'])

    # Adjust p-values for multiple testing
    rejected, pvals_corr = _multipletests(pvals, dim='feature', alpha=alpha, method=method)

    return corr, pvals_corr, rejected


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

    Example:
        >>> from esmtools.testing import multipletests
        >>> reject, xpvals_corrected = multipletests(p, method='fdr_bh')

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

