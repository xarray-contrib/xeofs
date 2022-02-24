import pytest

from xeofs.utils.tools import get_mode_selector


@pytest.mark.parametrize('input, output_type', [
    (5, list),
    ([5], list),
    ([1, 4, 6], list),
    (slice(5), slice),
    (slice(2, 4), slice),
    (None, slice),
])
def test_get_mode_selector(input, output_type):
    # Input is turned into list or slice
    assert isinstance(get_mode_selector(input), output_type)

@pytest.mark.parametrize('input', [
    ('4'),
    ((1, 2, 3)),
])
def test_invalid_get_mode_selector(input):
    # Input raises an error
    with pytest.raises(Exception):
        _ = get_mode_selector(input)
