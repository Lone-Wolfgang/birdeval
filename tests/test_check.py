"""`birdeval check` runs end to end on the fake adapter, and one broken model
doesn't stop the others from being checked."""

from birdeval.check import format_report, run_check


def test_check_reports_each_model():
    cfg = {
        "models": {
            "fake": {"adapter": "fake", "classes": ["Turdus merula", "Erithacus rubecula"]},
            "broken": {"adapter": "fake", "classes": ["Turdus merula"], "no_such_arg": 1},
        }
    }
    good, bad = run_check(cfg)

    assert good.ok, good.error
    assert (good.n_labels, good.n_mapped) == (2, 2)
    assert (good.window_s, good.hop_s) == (3.0, 1.5)
    # 12 s clip: windows start every 1.5 s up to 9 s; 2 s clip: one clipped window
    assert good.windows == {"synth_12s": 7, "synth_2s": 1}

    assert not bad.ok and "no_such_arg" in bad.error
    report = format_report([good, bad])
    assert "OK" in report and "FAIL" in report
