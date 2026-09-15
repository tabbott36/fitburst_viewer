from frb_viewer import FRBViewer


def test_compute_dm_total_uses_fixed_baseline():
    assert FRBViewer.compute_dm_total(0.0) == 219.456
    assert FRBViewer.compute_dm_total(0.123) == 219.579
    assert FRBViewer.compute_dm_total(-1.0) == 218.456


def test_compute_dm_change_from_total():
    assert FRBViewer.compute_dm_change(219.456) == 0.0
    assert FRBViewer.compute_dm_change(219.579) == 0.123
    assert FRBViewer.compute_dm_change(218.456) == -1.0
