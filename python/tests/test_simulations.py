import numpy as np
import pandas as pd
import pytest
import torch
from survivalgpu.simulations import (
    ConstantCovariate,
    Covariate,
    CoxCovariate,
    TDhist,
    TimeDependentCovariate,
    WCECovariate,
    bi_linear_scenario,
    compress_dataset,
    constant_scenario,
    event_censor_generation,
    event_FUP_Ti_generation,
    exponential_scenario,
    generate_WCEmat,
    generate_Xmat,
    get_dataset,
    get_probas,
    get_scenario,
    get_WCEmat_time_event,
    hat_scenario,
    matching_algo,
    simulate_dataset,
    simulate_dataset_batch,
    simulate_for_experiment,
)

# Almost every random function in this module accepts an explicit `rng` or
# `seed`, so most of the tests below rely on that instead of fighting
# non-determinism: pure functions get exact-value assertions, seeded
# pipelines get reproducibility + golden-value assertions, and only the
# hazard-ratio-weighted matching gets a (generously-toleranced) statistical
# check.

# Test the deterministic scenario functions ===============================


def test_scenario_functions_exact_values():
    assert exponential_scenario(0) == pytest.approx(7)
    assert bi_linear_scenario(0) == pytest.approx(1)
    assert bi_linear_scenario(365) == 0
    assert constant_scenario(180) == pytest.approx(1 / 180)
    assert constant_scenario(181) == 0
    assert hat_scenario(0) == 0
    assert hat_scenario(90) == pytest.approx(0.5)
    assert hat_scenario(180) == pytest.approx(1)
    assert hat_scenario(210) == pytest.approx(0.5)
    assert hat_scenario(240) == 0


@pytest.mark.parametrize(
    "scenario_name",
    [
        "exponential_scenario",
        "bi_linear_scenario",
        "early_peak_scenario",
        "inverted_u_scenario",
        "constant_scenario",
        "hat_scenario",
    ],
)
def test_get_scenario_is_normalized(scenario_name):
    max_time = 300
    scenario = get_scenario(scenario_name, max_time)

    assert scenario.shape == (max_time,)
    assert (scenario >= 0).all()
    assert scenario.sum() == pytest.approx(1.0)


def test_get_scenario_matches_manual_normalization():
    max_time = 5
    scenario = get_scenario("constant_scenario", max_time)
    # constant_scenario(u_t) = 1/180 for u_t in [0, 4], so it is uniform
    # once normalized.
    np.testing.assert_allclose(scenario, np.full(max_time, 1 / max_time))


def test_get_scenario_invalid_name_raises():
    with pytest.raises(ValueError, match="is not defined"):
        get_scenario("not_a_real_scenario", 10)


# Test the event/censoring helpers =========================================


def test_event_FUP_Ti_generation_exact():
    # Patient 0 is censored first (event at t=5 > censor at t=2).
    # Patient 1 has the event first (event at t=1 < censor at t=9).
    eventRandom = np.array([5, 1])
    censorRandom = np.array([2, 9])

    events, FUP_Ti = event_FUP_Ti_generation(eventRandom, censorRandom)

    # Output is sorted by follow-up time.
    np.testing.assert_array_equal(FUP_Ti, [1, 2])
    np.testing.assert_array_equal(events, [1, 0])


@pytest.mark.parametrize(
    ("bad_ratio", "match"),
    [(1.5, "must be inferior to 1"), (-0.1, "must be positive")],
)
def test_event_censor_generation_invalid_ratio_raises(bad_ratio, match):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=match):
        event_censor_generation(
            max_time=10, n_patients=3, censoring_ratio=bad_ratio, rng=rng
        )


def test_event_censor_generation_shapes_and_reproducibility():
    max_time, n_patients = 50, 20

    eventRandom_1, censorRandom_1 = event_censor_generation(
        max_time,
        n_patients,
        censoring_ratio=0.5,
        rng=np.random.default_rng(42),
    )
    eventRandom_2, censorRandom_2 = event_censor_generation(
        max_time,
        n_patients,
        censoring_ratio=0.5,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(eventRandom_1, eventRandom_2)
    np.testing.assert_array_equal(censorRandom_1, censorRandom_2)

    assert eventRandom_1.shape == (n_patients,)
    assert censorRandom_1.shape == (n_patients,)
    assert (eventRandom_1 >= 1).all()
    assert (eventRandom_1 <= max_time).all()
    assert (censorRandom_1 >= 1).all()


# Test TDhist ================================================================


def test_TDhist_reproducibility_and_shape():
    max_time, doses = 60, [1, 2, 3]

    v1 = TDhist(max_time, doses, np.random.default_rng(7))
    v2 = TDhist(max_time, doses, np.random.default_rng(7))

    np.testing.assert_array_equal(v1, v2)
    assert v1.shape == (max_time,)
    assert set(np.unique(v1)).issubset({0, *doses})


# Test the Covariate classes ===============================================


def test_covariate_base_class():
    assert Covariate(name="x").name == "x"


def test_constant_covariate_generates_one_value_per_patient():
    n_patients, max_time = 5, 4
    cov = ConstantCovariate(
        name="sex", values=[0, 1], weights=[0.5, 0.5], coef=0.7
    ).initialize_experiment(
        n_patients=n_patients, max_time=max_time, rng=np.random.default_rng(0)
    )

    assert cov.Xvector.shape == (n_patients * max_time,)
    assert set(np.unique(cov.Xvector)).issubset({0, 1})
    # The value must be constant within each patient's block.
    per_patient = cov.Xvector.reshape(n_patients, max_time)
    assert (per_patient == per_patient[:, [0]]).all()


def test_time_dependent_covariate_cumulate_exposure_exact():
    cov = TimeDependentCovariate(
        name="x", values=[1], coef=0.1, cumulative=True
    )
    cov.n_patients = 1
    cov.max_time = 4
    cov.Xvector = np.array([1.0, 2.0, 3.0, 4.0])

    cov.cumulate_exposure(cutoff=2)

    np.testing.assert_allclose(cov.Xvector, [1.0, 3.0, 6.0, 9.0])


def test_time_dependent_covariate_cumulate_exposure_without_xvector_raises():
    cov = TimeDependentCovariate(name="x", values=[1], coef=0.1)
    with pytest.raises(ValueError, match="Xvector has not been generated"):
        cov.cumulate_exposure(cutoff=2)


def test_cox_covariate_wraps_precomputed_xvector():
    xvector = np.arange(6.0)
    cov = CoxCovariate(name="z", Xvector=xvector, coef=0.3)
    cov.initialize_experiment(n_patients=2, max_time=3)

    assert cov.n_patients == 2
    assert cov.max_time == 3
    np.testing.assert_array_equal(cov.Xvector, xvector)


def test_wce_covariate_generate_wcevector_exact():
    cov = WCECovariate(
        name="dose",
        values=[1],
        scenario_name="constant_scenario",
        HR_target=2.0,
    )
    cov.n_patients = 1
    cov.max_time = 3
    # constant_scenario normalizes to a uniform [1/3, 1/3, 1/3] weight.
    cov.Xvector = np.array([2.0, 3.0, 5.0])

    cov.generate_WCEvector()

    expected = np.array([2 / 3, (2 + 3) / 3, (2 + 3 + 5) / 3])
    np.testing.assert_allclose(cov.WCEvector, expected)


def test_wce_covariate_generate_wcevector_without_xvector_raises():
    cov = WCECovariate(
        name="dose",
        values=[1],
        scenario_name="constant_scenario",
        HR_target=1.0,
    )
    cov.n_patients = 1
    cov.max_time = 3
    with pytest.raises(ValueError, match="Xvector has not been generated"):
        cov.generate_WCEvector()


def test_wce_covariate_full_initialization_is_well_formed():
    n_patients, max_time = 4, 10
    cov = WCECovariate(
        name="dose",
        values=[1, 2, 3],
        scenario_name="hat_scenario",
        HR_target=1.5,
    ).initialize_experiment(
        n_patients=n_patients, max_time=max_time, rng=np.random.default_rng(1)
    )

    assert cov.Xvector.shape == (n_patients * max_time,)
    assert cov.WCEvector.shape == (n_patients * max_time,)
    assert not np.isnan(cov.WCEvector).any()


# Test matrix assembly (generate_Xmat / generate_WCEmat) ===================


def _make_manual_covariates():
    wce_cov = WCECovariate(
        name="w", values=[1], scenario_name="constant_scenario", HR_target=2.0
    )
    wce_cov.Xvector = np.array([1.0, 2.0, 3.0, 4.0])
    wce_cov.WCEvector = np.array([10.0, 20.0, 30.0, 40.0])

    cox_cov = ConstantCovariate(
        name="c", values=[0, 1], weights=[1, 1], coef=0.5
    )
    cox_cov.Xvector = np.array([5.0, 5.0, 6.0, 6.0])

    return wce_cov, cox_cov


def test_generate_Xmat_assembly():
    wce_cov, cox_cov = _make_manual_covariates()

    Xmat = generate_Xmat([wce_cov], [cox_cov], max_time=2, n_patients=2)

    assert Xmat.shape == (4, 3)
    np.testing.assert_array_equal(Xmat[:, 0], [0, 0, 1, 1])
    np.testing.assert_array_equal(Xmat[:, 1], wce_cov.Xvector)
    np.testing.assert_array_equal(Xmat[:, 2], cox_cov.Xvector)


def test_generate_WCEmat_assembly():
    wce_cov, cox_cov = _make_manual_covariates()

    WCEmat = generate_WCEmat([wce_cov], [cox_cov], max_time=2, n_patients=2)

    assert WCEmat.shape == (4, 3)
    np.testing.assert_array_equal(WCEmat[:, 0], [0, 0, 1, 1])
    np.testing.assert_array_equal(WCEmat[:, 1], wce_cov.WCEvector)
    np.testing.assert_array_equal(WCEmat[:, 2], cox_cov.Xvector)


def test_get_WCEmat_time_event_selects_matching_row_per_block():
    WCEmat = np.array(
        [
            [0, 1.0],
            [0, 2.0],
            [1, 3.0],
            [1, 4.0],
        ]
    )
    selected = get_WCEmat_time_event(WCEmat, time_event=2, max_time=2)
    np.testing.assert_array_equal(selected, [[0, 2.0], [1, 4.0]])


# Test get_probas =============================================================


def test_get_probas_matches_manual_softmax():
    WCEmat_time_event = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    HR_target_list = torch.tensor([2.0])

    probas = get_probas(WCEmat_time_event, HR_target_list)

    weights = torch.tensor([1.0, 2.0, 4.0])  # exp(log(2) * [0, 1, 2])
    expected = weights / weights.sum()
    torch.testing.assert_close(probas, expected)
    assert probas.sum().item() == pytest.approx(1.0)


# Test get_dataset ============================================================


def test_get_dataset_exact_assembly():
    # Two patients, max_time=3, one WCE-style and one Cox-style covariate.
    Xmat = np.array(
        [
            [0, 100.0, 200.0],
            [0, 101.0, 201.0],
            [0, 102.0, 202.0],
            [1, 110.0, 210.0],
            [1, 111.0, 211.0],
            [1, 112.0, 212.0],
        ]
    )

    dataset = get_dataset(
        Xmat=Xmat,
        covariate_names=["w", "c"],
        n_patients=2,
        FUP_tis=[2, 3],
        events=[1, 0],
        wce_id_indexes=[0, 1],
        max_time=3,
    )

    expected = pd.DataFrame(
        {
            "patients": [1, 1, 2, 2, 2],
            "fup": [2, 2, 3, 3, 3],
            "start": [0, 1, 0, 1, 2],
            "stop": [1, 2, 1, 2, 3],
            "events": [0, 1, 0, 0, 0],
            "w": [100.0, 101.0, 110.0, 111.0, 112.0],
            "c": [200.0, 201.0, 210.0, 211.0, 212.0],
        }
    )

    pd.testing.assert_frame_equal(dataset, expected, check_dtype=False)


# Test compress_dataset =======================================================


def test_compress_dataset_merges_consecutive_identical_rows():
    dataset = pd.DataFrame(
        {
            "patients": [1, 1, 1, 2, 2],
            "fup": [3, 3, 3, 2, 2],
            "start": [0, 1, 2, 0, 1],
            "stop": [1, 2, 3, 1, 2],
            "events": [0, 0, 1, 0, 1],
            "covA": [5, 5, 7, 9, 9],
        }
    )

    compressed = compress_dataset(dataset)

    expected = pd.DataFrame(
        {
            "patients": [1, 1, 2],
            "group_id": [1, 2, 1],
            "start": [0, 2, 0],
            "stop": [2, 3, 2],
            "events": [0, 1, 1],
            "covA": [5, 7, 9],
        }
    )

    pd.testing.assert_frame_equal(compressed, expected, check_dtype=False)


# Test matching_algo ==========================================================


@pytest.mark.parametrize("seed", range(5))
def test_matching_algo_returns_a_permutation(seed):
    rng = np.random.default_rng(seed)
    n_patients, max_time = 6, 3

    WCEmat = np.zeros((n_patients * max_time, 2))
    WCEmat[:, 0] = np.repeat(np.arange(n_patients), max_time)
    WCEmat[:, 1] = rng.normal(size=n_patients * max_time)

    events = [1, 0, 1, 0, 1, 0]
    FUP_tis = [1, 2, 3, 1, 2, 3]

    selected = matching_algo(
        WCEmat=WCEmat,
        HR_target_list=np.array([2.0]),
        max_time=max_time,
        n_patients=n_patients,
        events=events,
        FUP_tis=FUP_tis,
        torch_generator=torch.Generator().manual_seed(seed),
    )

    assert sorted(selected.tolist()) == list(range(n_patients))


def test_matching_algo_is_reproducible_with_a_seeded_generator():
    rng = np.random.default_rng(0)
    n_patients, max_time = 6, 3
    WCEmat = np.zeros((n_patients * max_time, 2))
    WCEmat[:, 0] = np.repeat(np.arange(n_patients), max_time)
    WCEmat[:, 1] = rng.normal(size=n_patients * max_time)
    events = [1, 0, 1, 0, 1, 0]
    FUP_tis = [1, 2, 3, 1, 2, 3]

    kwargs = dict(
        WCEmat=WCEmat,
        HR_target_list=np.array([2.0]),
        max_time=max_time,
        n_patients=n_patients,
        events=events,
        FUP_tis=FUP_tis,
    )
    selected_1 = matching_algo(
        torch_generator=torch.Generator().manual_seed(123), **kwargs
    )
    selected_2 = matching_algo(
        torch_generator=torch.Generator().manual_seed(123), **kwargs
    )

    np.testing.assert_array_equal(selected_1, selected_2)


def test_matching_algo_prefers_high_hazard_ratio_covariate_for_events():
    # A single time point (max_time=1), two candidate blocks: one with a
    # covariate value of 0, one with a covariate value of 5. With a strongly
    # positive HR_target, the event patient processed first should almost
    # always be matched to the high-value block.
    WCEmat = np.array([[0.0, 0.0], [1.0, 5.0]])
    HR_target_list = np.array([np.exp(2.0)])  # log(HR) = 2

    n_trials = 200
    high_value_selected_first = 0
    for seed in range(n_trials):
        selected = matching_algo(
            WCEmat=WCEmat,
            HR_target_list=HR_target_list,
            max_time=1,
            n_patients=2,
            events=[1, 1],
            FUP_tis=[1, 1],
            torch_generator=torch.Generator().manual_seed(seed),
        )
        if selected[0] == 1:
            high_value_selected_first += 1

    # Expected probability is exp(10) / (1 + exp(10)) ~ 0.99995, so this is
    # nowhere near flaky while still checking the direction of the effect.
    assert high_value_selected_first / n_trials > 0.9


# Test the end-to-end simulation pipeline ====================================


def _small_wce_covariate():
    return WCECovariate(
        name="dose",
        values=[1, 2, 3],
        scenario_name="hat_scenario",
        HR_target=1.5,
    )


def test_simulate_dataset_is_reproducible_with_a_seed():
    kwargs = dict(
        max_time=20, n_patients=8, list_covariates=[_small_wce_covariate()]
    )
    dataset_1 = simulate_dataset(seed=42, **kwargs)
    dataset_2 = simulate_dataset(seed=42, **kwargs)

    pd.testing.assert_frame_equal(dataset_1, dataset_2)


def test_simulate_dataset_output_is_well_formed():
    max_time, n_patients = 20, 8
    dataset = simulate_dataset(
        max_time=max_time,
        n_patients=n_patients,
        list_covariates=[_small_wce_covariate()],
        seed=0,
    )

    assert set(dataset["patients"]) == set(range(1, n_patients + 1))
    assert not dataset["dose"].isna().any()

    for _, group in dataset.groupby("patients"):
        assert len(group) == group["fup"].iloc[0]
        assert group["stop"].iloc[-1] == group["fup"].iloc[0]
        assert group["events"].sum() <= 1
        if group["events"].sum() == 1:
            assert group["events"].iloc[-1] == 1


def test_simulate_dataset_none_covariates_raises():
    with pytest.raises(ValueError, match="list of covariates is None"):
        simulate_dataset(max_time=10, n_patients=5, list_covariates=None)


def test_simulate_dataset_unrecognized_covariate_raises():
    class NotACovariate:
        pass

    with pytest.raises(ValueError, match="not recognized"):
        simulate_dataset(
            max_time=10, n_patients=5, list_covariates=[NotACovariate()]
        )


def test_simulate_dataset_invalid_scenario_name_propagates():
    bad_covariate = WCECovariate(
        name="dose", values=[1], scenario_name="not_a_scenario", HR_target=1.0
    )
    with pytest.raises(ValueError, match="is not defined"):
        simulate_dataset(
            max_time=10, n_patients=5, list_covariates=[bad_covariate]
        )


def test_simulate_dataset_compress_reduces_or_preserves_row_count():
    kwargs = dict(
        max_time=20,
        n_patients=8,
        list_covariates=[_small_wce_covariate()],
        seed=0,
    )
    uncompressed = simulate_dataset(compress=False, **kwargs)
    compressed = simulate_dataset(compress=True, **kwargs)

    assert len(compressed) <= len(uncompressed)
    assert set(compressed["patients"]) == set(uncompressed["patients"])


def test_simulate_dataset_batch_matches_total_patient_count():
    n_patients = 10
    dataset = simulate_dataset_batch(
        max_time=15,
        n_patients=n_patients,
        list_covariates=[_small_wce_covariate()],
        batchsize=4,
        seed=0,
    )

    assert set(dataset["patients"]) == set(range(1, n_patients + 1))


def test_simulate_for_experiment_smoke():
    dataset = simulate_for_experiment(
        n_patients=6,
        max_time=15,
        HR_target=2.0,
        scenario_name="constant_scenario",
        seed=0,
    )

    assert "dose" in dataset.columns
    assert not dataset["dose"].isna().any()
