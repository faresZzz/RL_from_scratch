from tests.conftest import discretize, get_bin_counts, make_discretizer


def test_cartpole_discretizer_clips_observations_into_valid_bin_indices(small_config):
    discretizer = make_discretizer(config=small_config)
    bin_counts = get_bin_counts(small_config)

    low_state = discretize(discretizer, (-99.0, -99.0, -99.0, -99.0))
    mid_state = discretize(discretizer, (0.0, 0.0, 0.0, 0.0))
    high_state = discretize(discretizer, (99.0, 99.0, 99.0, 99.0))

    assert low_state == (0, 0, 0, 0)
    assert len(mid_state) == 4
    assert high_state == tuple(count - 1 for count in bin_counts)
    assert all(0 <= index < count for index, count in zip(mid_state, bin_counts))
