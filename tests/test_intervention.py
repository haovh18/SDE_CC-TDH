import pandas as pd

from sde_cctdh.intervention import (
    InterventionEffects,
    validate_effect_table,
    validation_passes,
)


def test_both_operators_must_strictly_pass():
    passing = InterventionEffects(0.5, 0.6, 0.2, 0.1, 0.1, 0.2)
    tied = InterventionEffects(0.5, 0.2, 0.5, 0.1, 0.1, 0.0)
    negative = InterventionEffects(-0.1, 0.6, -0.2, 0.1, -0.3, 0.2)
    assert validation_passes(passing)
    assert not validation_passes(tied)
    assert not validation_passes(negative)


def test_effect_table_contract_emits_kappa():
    rows = pd.DataFrame([{
        "target_mask": 0.5, "target_replace": 0.6,
        "control_mask": 0.2, "control_replace": 0.1,
        "donor_mask": 0.1, "donor_replace": 0.2,
    }])
    assert bool(validate_effect_table(rows).loc[0, "kappa"])
