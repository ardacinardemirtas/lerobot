import pytest

from keyboard_pnp import detections_from_roboflow_accumulated


def _pred(label: str, x: float, y: float, confidence: float = 0.9) -> dict:
    return {
        "class": label,
        "x": x,
        "y": y,
        "width": 10.0,
        "height": 10.0,
        "confidence": confidence,
    }


def test_accumulation_merges_partial_frames_into_layout_keys():
    batches = [
        [_pred("q", 10.0, 20.0), _pred("w", 30.0, 20.0)],
        [_pred("e", 50.0, 20.0), _pred("r", 70.0, 20.0)],
    ]

    dets = detections_from_roboflow_accumulated(batches)

    assert dets == {
        "q": (10.0, 20.0),
        "w": (30.0, 20.0),
        "e": (50.0, 20.0),
        "r": (70.0, 20.0),
    }


def test_accumulation_keeps_best_duplicate_per_frame_then_weights_across_frames():
    batches = [
        [
            _pred("a", 100.0, 50.0, confidence=0.2),
            _pred("a", 10.0, 20.0, confidence=1.0),
        ],
        [_pred("a", 20.0, 40.0, confidence=3.0)],
    ]

    dets = detections_from_roboflow_accumulated(batches)

    assert dets["a"] == pytest.approx((17.5, 35.0))


def test_accumulation_filters_non_layout_labels_keyboard_box_and_min_hits():
    batches = [
        [
            _pred("keyboard", 100.0, 100.0, confidence=0.99),
            _pred("banana", 20.0, 20.0, confidence=0.99),
            _pred("capslock", 10.0, 10.0, confidence=0.9),
            _pred("space", 50.0, 60.0, confidence=0.9),
        ],
        [_pred("capslock", 12.0, 14.0, confidence=0.9)],
    ]

    dets = detections_from_roboflow_accumulated(batches, min_hits=2)

    assert set(dets) == {"caps"}
    assert dets["caps"] == pytest.approx((11.0, 12.0))
