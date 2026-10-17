import json

import matplotlib.pyplot as plt

import make_plots


def test_index_bars_and_intervals_use_supplied_aggregates(tmp_path, monkeypatch):
    names = ["Kev-0.8B", "Kev-4B", "Kev-9B v2", "FuZev v1.0", "FuZev v1.1", "Kev-27B v2", "Jev"]
    reference = {"version": "Kev 1.0", "label": "Published results", "systems": {}}
    for index, name in enumerate(names):
        if not name.startswith("FuZev"):
            reference["systems"][name] = {split: {"documented_aggregate": {"overall": {"index": index + 10.}},
                                                   "index_ci95": [index + 8., index + 12.]}
                                          for split in ("development", "test")}
    record = {"systems": {name: {"overall": {"index": index + 10.}}
                           for index, name in enumerate(names) if name.startswith("FuZev")},
              "uncertainty": {"index": {name: [index + 8., index + 12.]
                  for index, name in enumerate(names) if name.startswith("FuZev")}, "samples": 2000, "seed": 0},
              "measurement_note": "Measured release candidate", "questions": 100}
    for split in ("dev", "test"):
        (tmp_path / f"kev-{split}-v1.1-vs-v1.0.json").write_text(json.dumps(record))
    captured = []
    monkeypatch.setattr(make_plots, "save", lambda figure, *args, **kwargs: captured.append(figure))
    make_plots.index_chart(tmp_path, tmp_path, reference)
    try:
        for ax in captured[0].axes:
            assert [bar.get_width() for bar in ax.patches] == [index + 10. for index in range(len(names))]
            assert ax.get_xlim()[0] == 0
            segments = [segment for collection in ax.collections for segment in collection.get_segments()]
            assert any(list(segment[:, 0]) == [8., 12.] for segment in segments)
    finally:
        for figure in captured:
            plt.close(figure)
