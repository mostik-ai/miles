from miles.backends.megatron_utils.flash_attn_4_window import unbounded_as_none


def test_unbounded_window_sides_become_none_and_bounded_sides_stay():
    assert unbounded_as_none((-1, 0)) == (None, 0)
    assert unbounded_as_none((-1, -1)) == (None, None)
    assert unbounded_as_none((128, 0)) == (128, 0)
    assert unbounded_as_none(None) is None
