import numpy as np
import pytest

from xeofs.models import (
    EOF,
    MCA,
    EOFRotator,
    HilbertEOF,
    HilbertEOFRotator,
    HilbertMCA,
    HilbertMCARotator,
    MCARotator,
)


# Orthogonality
# =============================================================================
# EOF
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_eof_components(dim, use_coslat, mock_data_array):
    """Components are orthogonal"""
    model = EOF(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(mock_data_array, dim=dim)
    V = model.data["components"].values
    assert np.allclose(
        V.T @ V, np.eye(V.shape[1]), atol=1e-5
    ), "Components are not orthogonal"


@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_eof_scores(dim, use_coslat, mock_data_array):
    """Scores are orthogonal"""
    model = EOF(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(mock_data_array, dim=dim)
    U = model.data["scores"].values / model.data["norms"].values
    assert np.allclose(
        U.T @ U, np.eye(U.shape[1]), atol=1e-5
    ), "Scores are not orthogonal"


# Hilbert EOF
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_ceof_components(dim, use_coslat, mock_data_array):
    """Components are unitary"""
    model = HilbertEOF(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(mock_data_array, dim=dim)
    V = model.data["components"].values
    assert np.allclose(
        V.conj().T @ V, np.eye(V.shape[1]), atol=1e-5
    ), "Components are not unitary"


@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_ceof_scores(dim, use_coslat, mock_data_array):
    """Scores are unitary"""
    model = HilbertEOF(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(mock_data_array, dim=dim)
    U = model.data["scores"].values / model.data["norms"].values
    assert np.allclose(
        U.conj().T @ U, np.eye(U.shape[1]), atol=1e-5
    ), "Scores are not unitary"


# Rotated EOF
@pytest.mark.parametrize(
    "dim, use_coslat, power",
    [
        (("time",), True, 1),
        (("lat", "lon"), False, 1),
        (("lon", "lat"), False, 1),
        (("time",), True, 2),
        (("lat", "lon"), False, 2),
        (("lon", "lat"), False, 2),
    ],
)
def test_reof_components(dim, use_coslat, power, mock_data_array):
    """Components are NOT orthogonal"""
    model = EOF(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(mock_data_array, dim=dim)
    rot = EOFRotator(n_modes=5, power=power)
    rot.fit(model)
    V = rot.data["components"].values
    K = V.conj().T @ V
    assert np.allclose(
        np.diag(K), np.ones(V.shape[1]), atol=1e-5
    ), "Components are not normalized"
    # Assert that off-diagonals are not zero
    assert not np.allclose(K, np.eye(K.shape[0])), "Rotated components are orthogonal"


@pytest.mark.parametrize(
    "dim, use_coslat, power",
    [
        (("time",), True, 1),
        (("lat", "lon"), False, 1),
        (("lon", "lat"), False, 1),
        (("time",), True, 2),
        (("lat", "lon"), False, 2),
        (("lon", "lat"), False, 2),
    ],
)
def test_reof_scores(dim, use_coslat, power, mock_data_array):
    """Components are orthogonal only for Varimax rotation"""
    model = EOF(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(mock_data_array, dim=dim)
    rot = EOFRotator(n_modes=5, power=power)
    rot.fit(model)
    U = rot.data["scores"].values / rot.data["norms"].values
    K = U.conj().T @ U
    if power == 1:
        # Varimax rotation does guarantee orthogonality
        assert np.allclose(
            K, np.eye(K.shape[1]), atol=1e-5
        ), "Components are not orthogonal"
    else:
        assert not np.allclose(K, np.eye(K.shape[1])), "Components are orthogonal"


# Hilbert rotated EOF
@pytest.mark.parametrize(
    "dim, use_coslat, power",
    [
        (("time",), True, 1),
        (("lat", "lon"), False, 1),
        (("lon", "lat"), False, 1),
        (("time",), True, 2),
        (("lat", "lon"), False, 2),
        (("lon", "lat"), False, 2),
    ],
)
def test_creof_components(dim, use_coslat, power, mock_data_array):
    """Components are NOT unitary"""
    model = HilbertEOF(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(mock_data_array, dim=dim)
    rot = HilbertEOFRotator(n_modes=5, power=power)
    rot.fit(model)
    V = rot.data["components"].values
    K = V.conj().T @ V
    assert np.allclose(
        np.diag(K), np.ones(V.shape[1]), atol=1e-5
    ), "Components are not normalized"
    # Assert that off-diagonals are not zero
    assert not np.allclose(K, np.eye(K.shape[0])), "Rotated components are unitary"


@pytest.mark.parametrize(
    "dim, use_coslat, power",
    [
        (("time",), True, 1),
        (("lat", "lon"), False, 1),
        (("lon", "lat"), False, 1),
        (("time",), True, 2),
        (("lat", "lon"), False, 2),
        (("lon", "lat"), False, 2),
    ],
)
def test_creof_scores(dim, use_coslat, power, mock_data_array):
    """Components are unitary only for Varimax rotation"""
    model = HilbertEOF(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(mock_data_array, dim=dim)
    rot = HilbertEOFRotator(n_modes=5, power=power)
    rot.fit(model)
    U = rot.data["scores"].values / rot.data["norms"].values
    K = U.conj().T @ U
    if power == 1:
        # Varimax rotation does guarantee unitarity
        assert np.allclose(
            K, np.eye(K.shape[1]), atol=1e-5
        ), "Components are not unitary"
    else:
        assert not np.allclose(K, np.eye(K.shape[1])), "Components are unitary"


# MCA
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_mca_components(dim, use_coslat, mock_data_array):
    """Components are orthogonal"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = MCA(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    V1 = model.data["components1"].values
    V2 = model.data["components2"].values
    K1 = V1.T @ V1
    K2 = V2.T @ V2
    assert np.allclose(
        K1, np.eye(K1.shape[0]), rtol=1e-8
    ), "Left components are not orthogonal"
    assert np.allclose(
        K2, np.eye(K2.shape[0]), rtol=1e-8
    ), "Right components are not orthogonal"


@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_mca_scores(dim, use_coslat, mock_data_array):
    """Scores are orthogonal"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = MCA(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    U1 = model.data["scores1"].values
    U2 = model.data["scores2"].values
    K = U1.T @ U2
    target = np.eye(K.shape[0]) * (model.data["input_data1"].sample.size - 1)
    assert np.allclose(K, target, atol=1e-5), "Scores are not orthogonal"


# Hilbert MCA
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_cmca_components(dim, use_coslat, mock_data_array):
    """Components are unitary"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = HilbertMCA(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    V1 = model.data["components1"].values
    V2 = model.data["components2"].values
    K1 = V1.conj().T @ V1
    K2 = V2.conj().T @ V2
    assert np.allclose(
        K1, np.eye(K1.shape[0]), atol=1e-5
    ), "Left components are not unitary"
    assert np.allclose(
        K2, np.eye(K2.shape[0]), atol=1e-5
    ), "Right components are not unitary"


@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_cmca_scores(dim, use_coslat, mock_data_array):
    """Scores are unitary"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = HilbertMCA(n_modes=10, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    U1 = model.data["scores1"].values
    U2 = model.data["scores2"].values
    K = U1.conj().T @ U2
    target = np.eye(K.shape[0]) * (model.data["input_data1"].sample.size - 1)
    assert np.allclose(K, target, atol=1e-5), "Scores are not unitary"


# Rotated MCA
@pytest.mark.parametrize(
    "dim, use_coslat, power, squared_loadings",
    [
        (("time",), True, 1, False),
        (("lat", "lon"), False, 1, False),
        (("lon", "lat"), False, 1, False),
        (("time",), True, 2, False),
        (("lat", "lon"), False, 2, False),
        (("lon", "lat"), False, 2, False),
        (("time",), True, 1, True),
        (("lat", "lon"), False, 1, True),
        (("lon", "lat"), False, 1, True),
        (("time",), True, 2, True),
        (("lat", "lon"), False, 2, True),
        (("lon", "lat"), False, 2, True),
    ],
)
def test_rmca_components(dim, use_coslat, power, squared_loadings, mock_data_array):
    """Components are NOT orthogonal"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = MCA(n_modes=19, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    rot = MCARotator(n_modes=5, power=power, squared_loadings=squared_loadings)
    rot.fit(model)
    V1 = rot.data["components1"].values
    V2 = rot.data["components2"].values
    K1 = V1.conj().T @ V1
    K2 = V2.conj().T @ V2
    assert np.allclose(
        np.diag(K1), np.ones(K1.shape[0]), rtol=1e-5
    ), "Components are not normalized"
    assert np.allclose(
        np.diag(K2), np.ones(K2.shape[0]), rtol=1e-5
    ), "Components are not normalized"
    # Assert that off-diagonals are not zero
    assert not np.allclose(K1, np.eye(K1.shape[0])), "Rotated components are orthogonal"
    assert not np.allclose(K2, np.eye(K2.shape[0])), "Rotated components are orthogonal"


@pytest.mark.parametrize(
    "dim, use_coslat, power, squared_loadings",
    [
        (("time",), True, 1, False),
        (("lat", "lon"), False, 1, False),
        (("lon", "lat"), False, 1, False),
        (("time",), True, 2, False),
        (("lat", "lon"), False, 2, False),
        (("lon", "lat"), False, 2, False),
        (("time",), True, 1, True),
        (("lat", "lon"), False, 1, True),
        (("lon", "lat"), False, 1, True),
        (("time",), True, 2, True),
        (("lat", "lon"), False, 2, True),
        (("lon", "lat"), False, 2, True),
    ],
)
def test_rmca_scores(dim, use_coslat, power, squared_loadings, mock_data_array):
    """Components are orthogonal only for Varimax rotation"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = MCA(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    rot = MCARotator(n_modes=5, power=power, squared_loadings=squared_loadings)
    rot.fit(model)
    U1 = rot.data["scores1"].values
    U2 = rot.data["scores2"].values
    K = U1.conj().T @ U2
    target = np.eye(K.shape[0]) * (model.data["input_data1"].sample.size - 1)
    if power == 1:
        # Varimax rotation does guarantee orthogonality
        assert np.allclose(K, target, atol=1e-5), "Components are not orthogonal"
    else:
        assert not np.allclose(K, target), "Components are orthogonal"


# Hilbert Rotated MCA
@pytest.mark.parametrize(
    "dim, use_coslat, power, squared_loadings",
    [
        (("time",), True, 1, False),
        (("lat", "lon"), False, 1, False),
        (("lon", "lat"), False, 1, False),
        (("time",), True, 2, False),
        (("lat", "lon"), False, 2, False),
        (("lon", "lat"), False, 2, False),
        (("time",), True, 1, True),
        (("lat", "lon"), False, 1, True),
        (("lon", "lat"), False, 1, True),
        (("time",), True, 2, True),
        (("lat", "lon"), False, 2, True),
        (("lon", "lat"), False, 2, True),
    ],
)
def test_crmca_components(dim, use_coslat, power, squared_loadings, mock_data_array):
    """Components are NOT orthogonal"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = HilbertMCA(n_modes=19, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    rot = HilbertMCARotator(n_modes=5, power=power, squared_loadings=squared_loadings)
    rot.fit(model)
    V1 = rot.data["components1"].values
    V2 = rot.data["components2"].values
    K1 = V1.conj().T @ V1
    K2 = V2.conj().T @ V2
    assert np.allclose(
        np.diag(K1), np.ones(K1.shape[0]), rtol=1e-5
    ), "Components are not normalized"
    assert np.allclose(
        np.diag(K2), np.ones(K2.shape[0]), rtol=1e-5
    ), "Components are not normalized"
    # Assert that off-diagonals are not zero
    assert not np.allclose(K1, np.eye(K1.shape[0])), "Rotated components are orthogonal"
    assert not np.allclose(K2, np.eye(K2.shape[0])), "Rotated components are orthogonal"


@pytest.mark.parametrize(
    "dim, use_coslat, power, squared_loadings",
    [
        (("time",), True, 1, False),
        (("lat", "lon"), False, 1, False),
        (("lon", "lat"), False, 1, False),
        (("time",), True, 2, False),
        (("lat", "lon"), False, 2, False),
        (("lon", "lat"), False, 2, False),
        (("time",), True, 1, True),
        (("lat", "lon"), False, 1, True),
        (("lon", "lat"), False, 1, True),
        (("time",), True, 2, True),
        (("lat", "lon"), False, 2, True),
        (("lon", "lat"), False, 2, True),
    ],
)
def test_crmca_scores(dim, use_coslat, power, squared_loadings, mock_data_array):
    """Components are orthogonal only for Varimax rotation"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = HilbertMCA(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    rot = HilbertMCARotator(n_modes=5, power=power, squared_loadings=squared_loadings)
    rot.fit(model)
    U1 = rot.data["scores1"].values
    U2 = rot.data["scores2"].values
    K = U1.conj().T @ U2
    target = np.eye(K.shape[0]) * (model.data["input_data1"].sample.size - 1)
    if power == 1:
        # Varimax rotation does guarantee orthogonality
        assert np.allclose(K, target, atol=1e-5), "Components are not orthogonal"
    else:
        assert not np.allclose(K, target), "Components are orthogonal"


# Transform
# =============================================================================
# EOF
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
@pytest.mark.parametrize("normalized", [True, False])
def test_eof_transform(dim, use_coslat, mock_data_array, normalized):
    """Transforming the original data results in the model scores"""
    model = EOF(
        n_modes=5,
        standardize=True,
        use_coslat=use_coslat,
        random_state=5,
    )
    model.fit(mock_data_array, dim=dim)
    scores = model.scores(normalized=normalized)
    pseudo_scores = model.transform(mock_data_array, normalized=normalized)
    assert np.allclose(
        scores, pseudo_scores, atol=1e-4
    ), "Transformed data does not match the scores"


# Hilbert EOF
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
@pytest.mark.parametrize("normalized", [True, False])
def test_ceof_transform(dim, use_coslat, mock_data_array, normalized):
    """Not implemented yet"""
    model = HilbertEOF(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(mock_data_array, dim=dim)
    model.scores(normalized=normalized)
    with pytest.raises(NotImplementedError):
        model.transform(mock_data_array, normalized=normalized)


# Rotated EOF
@pytest.mark.parametrize(
    "dim, use_coslat, power",
    [
        (("time",), True, 1),
        (("lat", "lon"), False, 1),
        (("lon", "lat"), False, 1),
        (("time",), True, 2),
        (("lat", "lon"), False, 2),
        (("lon", "lat"), False, 2),
    ],
)
@pytest.mark.parametrize("normalized", [True, False])
def test_reof_transform(dim, use_coslat, power, mock_data_array, normalized):
    """Transforming the original data results in the model scores"""
    model = EOF(n_modes=5, standardize=True, use_coslat=use_coslat, random_state=5)
    model.fit(mock_data_array, dim=dim)
    rot = EOFRotator(n_modes=5, power=power)
    rot.fit(model)
    scores = rot.scores(normalized=normalized)
    pseudo_scores = rot.transform(mock_data_array, normalized=normalized)
    np.testing.assert_allclose(
        scores,
        pseudo_scores,
        rtol=5e-3,
        err_msg="Transformed data does not match the scores",
    )


# Hilbert Rotated EOF
@pytest.mark.parametrize(
    "dim, use_coslat, power",
    [
        (("time",), True, 1),
        (("lat", "lon"), False, 1),
        (("lon", "lat"), False, 1),
        (("time",), True, 2),
        (("lat", "lon"), False, 2),
        (("lon", "lat"), False, 2),
    ],
)
@pytest.mark.parametrize("normalized", [True, False])
def test_creof_transform(dim, use_coslat, power, mock_data_array, normalized):
    """not implemented yet"""
    model = HilbertEOF(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(mock_data_array, dim=dim)
    rot = HilbertEOFRotator(n_modes=5, power=power)
    rot.fit(model)
    rot.scores(normalized=normalized)
    with pytest.raises(NotImplementedError):
        rot.transform(mock_data_array, normalized=normalized)


# MCA
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_mca_transform(dim, use_coslat, mock_data_array):
    """Transforming the original data results in the model scores"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = MCA(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    scores1, scores2 = model.scores()
    pseudo_scores1, pseudo_scores2 = model.transform(data1=data1, data2=data2)
    assert np.allclose(
        scores1, pseudo_scores1, atol=1e-4
    ), "Transformed data does not match the scores"
    assert np.allclose(
        scores2, pseudo_scores2, atol=1e-4
    ), "Transformed data does not match the scores"


# Hilbert MCA
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_cmca_transform(dim, use_coslat, mock_data_array):
    """Transforming the original data results in the model scores"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = HilbertMCA(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    scores1, scores2 = model.scores()
    with pytest.raises(NotImplementedError):
        pseudo_scores1, pseudo_scores2 = model.transform(data1=data1, data2=data2)


# Rotated MCA
@pytest.mark.parametrize(
    "dim, use_coslat, power, squared_loadings",
    [
        (("time",), True, 1, False),
        (("lat", "lon"), False, 1, False),
        (("lon", "lat"), False, 1, False),
        (("time",), True, 2, False),
        (("lat", "lon"), False, 2, False),
        (("lon", "lat"), False, 2, False),
        (("time",), True, 1, True),
        (("lat", "lon"), False, 1, True),
        (("lon", "lat"), False, 1, True),
        (("time",), True, 2, True),
        (("lat", "lon"), False, 2, True),
        (("lon", "lat"), False, 2, True),
    ],
)
def test_rmca_transform(dim, use_coslat, power, squared_loadings, mock_data_array):
    """Transforming the original data results in the model scores"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = MCA(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    rot = MCARotator(n_modes=5, power=power, squared_loadings=squared_loadings)
    rot.fit(model)
    scores1, scores2 = rot.scores()
    pseudo_scores1, pseudo_scores2 = rot.transform(data1=data1, data2=data2)
    assert np.allclose(
        scores1, pseudo_scores1, atol=1e-5
    ), "Transformed data does not match the scores"
    assert np.allclose(
        scores2, pseudo_scores2, atol=1e-5
    ), "Transformed data does not match the scores"


# Hilbert Rotated MCA
@pytest.mark.parametrize(
    "dim, use_coslat, power, squared_loadings",
    [
        (("time",), True, 1, False),
        (("lat", "lon"), False, 1, False),
        (("lon", "lat"), False, 1, False),
        (("time",), True, 2, False),
        (("lat", "lon"), False, 2, False),
        (("lon", "lat"), False, 2, False),
        (("time",), True, 1, True),
        (("lat", "lon"), False, 1, True),
        (("lon", "lat"), False, 1, True),
        (("time",), True, 2, True),
        (("lat", "lon"), False, 2, True),
        (("lon", "lat"), False, 2, True),
    ],
)
def test_crmca_transform(dim, use_coslat, power, squared_loadings, mock_data_array):
    """Transforming the original data results in the model scores"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = HilbertMCA(n_modes=5, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    rot = HilbertMCARotator(n_modes=5, power=power, squared_loadings=squared_loadings)
    rot.fit(model)
    scores1, scores2 = rot.scores()
    with pytest.raises(NotImplementedError):
        pseudo_scores1, pseudo_scores2 = rot.transform(data1=data1, data2=data2)


# Reconstruct
# =============================================================================
def r2_score(x, y, dim=None):
    """Compute the R2 score between two DataArrays

    Parameters
    ----------
    x : xr.DataArray
        Reference data
    y : xr.DataArray
        Testing data to be compared with the reference data
    dim : str or sequence of str, optional
        Dimension(s) over which to compute the R2 score (the default is None, which
        means that the R2 score is computed over all dimensions)

    Returns
    -------
    r2_score : xr.DataArray
       R2 score between x and y

    """
    ssres = ((x - y) ** 2).sum(dim)
    sstot = ((x - x.mean(dim)) ** 2).sum(dim)
    return 1 - (ssres / sstot)


# EOF
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
@pytest.mark.parametrize("normalized", [True, False])
def test_eof_inverse_transform(dim, use_coslat, mock_data_array, normalized):
    """Inverse transform produces an approximate reconstruction of the original data"""
    data = mock_data_array
    model = EOF(n_modes=19, standardize=True, use_coslat=use_coslat)
    model.fit(data, dim=dim)
    scores = model.scores(normalized=normalized)
    data_rec = model.inverse_transform(scores, normalized=normalized)
    r2 = r2_score(data, data_rec, dim=dim)
    r2 = r2.mean()
    # Choose a threshold of 0.95; a bit arbitrary
    assert r2 > 0.95, "Inverse transform does not produce a good reconstruction"


# Hilbert EOF
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
@pytest.mark.parametrize("normalized", [True, False])
def test_ceof_inverse_transform(dim, use_coslat, mock_data_array, normalized):
    """Inverse transform produces an approximate reconstruction of the original data"""
    data = mock_data_array
    model = HilbertEOF(n_modes=19, standardize=True, use_coslat=use_coslat)
    model.fit(data, dim=dim)
    scores = model.scores(normalized=normalized)
    data_rec = model.inverse_transform(scores, normalized=normalized).real
    r2 = r2_score(data, data_rec, dim=dim)
    r2 = r2.mean()
    # Choose a threshold of 0.95; a bit arbitrary
    assert r2 > 0.95, "Inverse transform does not produce a good reconstruction"


# Rotated EOF
@pytest.mark.parametrize(
    "dim, use_coslat, power",
    [
        (("time",), True, 1),
        (("lat", "lon"), False, 1),
        (("lon", "lat"), False, 1),
        (("time",), True, 2),
        (("lat", "lon"), False, 2),
        (("lon", "lat"), False, 2),
    ],
)
@pytest.mark.parametrize("normalized", [True, False])
def test_reof_inverse_transform(dim, use_coslat, power, mock_data_array, normalized):
    """Inverse transform produces an approximate reconstruction of the original data"""
    data = mock_data_array
    model = EOF(n_modes=19, standardize=True, use_coslat=use_coslat)
    model.fit(data, dim=dim)
    rot = EOFRotator(n_modes=19, power=power)
    rot.fit(model)
    scores = rot.scores(normalized=normalized)
    data_rec = rot.inverse_transform(scores, normalized=normalized).real
    r2 = r2_score(data, data_rec, dim=dim)
    r2 = r2.mean()
    # Choose a threshold of 0.95; a bit arbitrary
    assert (
        r2 > 0.95
    ), f"Inverse transform does not produce a good reconstruction (R2={r2.values:.2f})"


# Hilbert Rotated EOF
@pytest.mark.parametrize(
    "dim, use_coslat, power",
    [
        (("time",), True, 1),
        (("lat", "lon"), False, 1),
        (("lon", "lat"), False, 1),
        (("time",), True, 2),
        (("lat", "lon"), False, 2),
        (("lon", "lat"), False, 2),
    ],
)
@pytest.mark.parametrize("normalized", [True, False])
def test_creof_inverse_transform(dim, use_coslat, power, mock_data_array, normalized):
    """Inverse transform produces an approximate reconstruction of the original data"""
    data = mock_data_array
    model = HilbertEOF(n_modes=19, standardize=True, use_coslat=use_coslat)
    model.fit(data, dim=dim)
    rot = HilbertEOFRotator(n_modes=19, power=power)
    rot.fit(model)
    scores = rot.scores(normalized=normalized)
    data_rec = rot.inverse_transform(scores, normalized=normalized).real
    r2 = r2_score(data, data_rec, dim=dim)
    r2 = r2.mean()
    # Choose a threshold of 0.95; a bit arbitrary
    assert (
        r2 > 0.95
    ), f"Inverse transform does not produce a good reconstruction (R2={r2.values:.2f})"


# MCA
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_mca_inverse_transform(dim, use_coslat, mock_data_array):
    """Inverse transform produces an approximate reconstruction of the original data"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = MCA(n_modes=19, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    scores1 = model.data["scores1"]
    scores2 = model.data["scores2"]
    data1_rec, data2_rec = model.inverse_transform(scores1, scores2)
    r2_1 = r2_score(data1, data1_rec, dim=dim)
    r2_2 = r2_score(data2, data2_rec, dim=dim)
    r2_1 = r2_1.mean()
    r2_2 = r2_2.mean()
    # Choose a threshold of 0.95; a bit arbitrary
    assert (
        r2_1 > 0.95
    ), f"Inverse transform does not produce a good reconstruction of left field (R2={r2_1.values:.2f})"
    assert (
        r2_2 > 0.95
    ), f"Inverse transform does not produce a good reconstruction of right field (R2={r2_2.values:.2f})"


# Hilbert MCA
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_cmca_inverse_transform(dim, use_coslat, mock_data_array):
    """Inverse transform produces an approximate reconstruction of the original data"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = HilbertMCA(n_modes=19, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    scores1 = model.data["scores1"]
    scores2 = model.data["scores2"]
    data1_rec, data2_rec = model.inverse_transform(scores1, scores2)
    r2_1 = r2_score(data1, data1_rec, dim=dim)
    r2_2 = r2_score(data2, data2_rec, dim=dim)
    r2_1 = r2_1.mean()
    r2_2 = r2_2.mean()
    # Choose a threshold of 0.95; a bit arbitrary
    assert (
        r2_1 > 0.95
    ), f"Inverse transform does not produce a good reconstruction of left field (R2={r2_1.values:.2f})"
    assert (
        r2_2 > 0.95
    ), f"Inverse transform does not produce a good reconstruction of right field (R2={r2_2.values:.2f})"


# Rotated MCA
@pytest.mark.parametrize(
    "dim, use_coslat, power, squared_loadings",
    [
        (("time",), True, 1, False),
        (("lat", "lon"), False, 1, False),
        (("lon", "lat"), False, 1, False),
        (("time",), True, 2, False),
        (("lat", "lon"), False, 2, False),
        (("lon", "lat"), False, 2, False),
        (("time",), True, 1, True),
        (("lat", "lon"), False, 1, True),
        (("lon", "lat"), False, 1, True),
        (("time",), True, 2, True),
        (("lat", "lon"), False, 2, True),
        (("lon", "lat"), False, 2, True),
    ],
)
def test_rmca_inverse_transform(
    dim, use_coslat, power, squared_loadings, mock_data_array
):
    """Inverse transform produces an approximate reconstruction of the original data"""
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = MCA(n_modes=10, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    rot = MCARotator(n_modes=10, power=power, squared_loadings=squared_loadings)
    rot.fit(model)
    scores1 = rot.data["scores1"]
    scores2 = rot.data["scores2"]
    data1_rec, data2_rec = rot.inverse_transform(scores1, scores2)
    r2_1 = r2_score(data1, data1_rec, dim=dim)
    r2_2 = r2_score(data2, data2_rec, dim=dim)
    r2_1 = r2_1.mean()
    r2_2 = r2_2.mean()
    # Choose a threshold of 0.90; a bit arbitrary
    assert (
        r2_1 > 0.75
    ), f"Inverse transform does not produce a good reconstruction of left field (R2={r2_1.values:.2f})"
    assert (
        r2_2 > 0.75
    ), f"Inverse transform does not produce a good reconstruction of right field (R2={r2_2.values:.2f})"


# Hilbert Rotated MCA
@pytest.mark.parametrize(
    "dim, use_coslat, power, squared_loadings",
    [
        (("time",), True, 1, False),
        (("lat", "lon"), False, 1, False),
        (("lon", "lat"), False, 1, False),
        (("time",), True, 2, False),
        (("lat", "lon"), False, 2, False),
        (("lon", "lat"), False, 2, False),
        (("time",), True, 1, True),
        (("lat", "lon"), False, 1, True),
        (("lon", "lat"), False, 1, True),
        (("time",), True, 2, True),
        (("lat", "lon"), False, 2, True),
        (("lon", "lat"), False, 2, True),
    ],
)
def test_crmca_inverse_transform(
    dim, use_coslat, power, squared_loadings, mock_data_array
):
    """Inverse transform produces an approximate reconstruction of the original data"""
    # NOTE: The lobpcg SVD solver for Hilbert matrices requires a small number of modes
    # compared to the actual data size. Since we have a small test set here we only use
    # 10 modes for the test. Therefore, the threshold for the R2 score is lower than for
    # the other tests.
    data1 = mock_data_array.copy()
    data2 = data1.copy() ** 2
    model = HilbertMCA(n_modes=10, standardize=True, use_coslat=use_coslat)
    model.fit(data1, data2, dim=dim)
    rot = HilbertMCARotator(n_modes=10, power=power, squared_loadings=squared_loadings)
    rot.fit(model)
    scores1 = rot.data["scores1"]
    scores2 = rot.data["scores2"]
    data1_rec, data2_rec = rot.inverse_transform(scores1, scores2)
    r2_1 = r2_score(data1, data1_rec, dim=dim)
    r2_2 = r2_score(data2, data2_rec, dim=dim)
    r2_1 = r2_1.mean()
    r2_2 = r2_2.mean()
    # Choose a threshold of 0.80; a bit arbitrary
    assert (
        r2_1 > 0.80
    ), f"Inverse transform does not produce a good reconstruction of left field (R2={r2_1.values:.2f})"
    assert (
        r2_2 > 0.80
    ), f"Inverse transform does not produce a good reconstruction of right field (R2={r2_2.values:.2f})"
