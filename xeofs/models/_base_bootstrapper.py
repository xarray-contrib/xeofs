from typing import Optional

import numpy as np
import warnings
from tqdm import trange

from ._base_eof import _BaseEOF


class _BaseBootstrapper:

    def __init__(self, n_boot: int, alpha : float = 0.05, test_type : Optional[str] = 'one-sided'):
        self._params = {
            'n_boot': n_boot,
            'alpha': alpha,
            'test_type': test_type,
        }
        if test_type == 'one-sided':
            pvalue = alpha
        elif test_type == 'two-sided':
            pvalue = alpha / 2
        else:
            raise ValueError('{:} is not a valid test_type'.format(test_type))

        n_boot_min = round(1. / pvalue)
        if n_boot_min > n_boot:
            msg = (
                'To reach a significance level of {:} '
                'use at least {:} bootstraps'
            )
            msg = msg.format(alpha, n_boot_min)
            warnings.warn(msg)
        self._params['quantiles'] = [0. + pvalue, 1. - pvalue]

    def bootstrap(self, model):
        self._model = model
        n_boot = self._params['n_boot']
        n_samples = model.n_samples
        n_modes = model.n_modes
        shape_eofs = model._eofs.shape
        shape_pcs = model._pcs.shape

        self._explained_variance = np.zeros((n_boot, n_modes)) * np.nan
        self._eofs = np.zeros((n_boot,) + shape_eofs) * np.nan
        self._pcs = np.zeros((n_boot,) + shape_pcs) * np.nan

        for i in trange(n_boot, desc='Bootstrap'):
            idx_rnd = np.random.choice(n_samples, n_samples, replace=True)
            X_rnd = self._model.X[idx_rnd]
            bs_pca = _BaseEOF(X_rnd, n_modes=n_modes)
            bs_pca.solve()
            self._explained_variance[i] = bs_pca.explained_variance()
            self._eofs[i] = bs_pca.eofs()
            # Project original data onto bootstrap EOFs to obtain bootstrap PCs
            self._pcs[i] = bs_pca.project_onto_eofs(self._model.X)

        # return None
        # Fix sign of individual EOFs determined by correlation coefficients
        # for a given mode with all the individual bootstrap members
        for mode in range(n_modes):
            corr_mat = np.corrcoef(self._model._eofs[:, mode], self._eofs[..., mode])
            corr_coefs = corr_mat[0][1:]
            signs = np.sign(corr_coefs)
            self._eofs[:, :, mode] = self._eofs[:, :, mode] * signs[..., None]
            self._pcs[:, :, mode] = self._pcs[:, :, mode] * signs[..., None]

        # return None
        # Extract quantiles of all quantities
        # (eigenvalues/exp. var., eigenvectors/EOFs, projections/PCs)
        self._explained_variance = np.quantile(
            self._explained_variance, self._params['quantiles'], axis=0
        )
        self._eofs = np.quantile(self._eofs, self._params['quantiles'], axis=0)
        self._pcs = np.quantile(self._pcs, self._params['quantiles'], axis=0)

        # Determine which modes are significant
        # If the lower quantile of explained variance for a given mode
        # is lower than the upper quantile of a subsquent mode, the first mode
        # is considered not significant.
        quantile_lower, quantile_upper = self._explained_variance
        is_not_sig = (quantile_lower - np.roll(quantile_upper, -1)) < 0
        self._is_significant_mode = is_not_sig.cumsum()[:-1] == 0

        # Determine which elements of the EOFs/eigenvectors are significant
        # If for a given element of the eigenvectors the quantiles cross zero,
        # the given element can not be discerned from zero ==> insignificant
        # We can easily check this by testing whether the sign of both
        # quantiles is the same (i.e. significant) or not (i.e. not significant)
        q_low, q_up = self._eofs
        self._is_significant_eof_element = np.sign(q_low) == np.sign(q_up)

        # Determine which elements of the PCs are significant
        q_low, q_up = self._pcs
        self._is_significant_pc_element = np.sign(q_low) == np.sign(q_up)

    def get_params(self):
        return self._params

    def n_significant_modes(self):
        return self._is_significant_mode.sum()

    def explained_variance(self):
        return self._explained_variance, self._is_significant_mode

    def eofs(self):
        return self._eofs, self._is_significant_eof_element

    def pcs(self):
        return self._pcs, self._is_significant_pc_element
