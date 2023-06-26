import xarray as xr
from statsmodels.stats.multitest import multipletests as statsmodels_multipletests

from .constants import MULTIPLE_TESTS


def multipletests(p, dim, alpha=0.05, method=None, **multipletests_kwargs):
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

