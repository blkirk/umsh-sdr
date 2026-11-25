from umsh_sdr.db import DB
from umsh_sdr.emission_buckets import bw_bucket_label, buckets, capture_seconds


def test_buckets(tmp_path):
    db = DB(tmp_path / "t.sqlite")
    db.event(1000, "channel_start", {}); db.event(4000, "channel_stop", {})  # 50 min of capture
    db.event(5000, "channel_start", {}); db.event(5600, "channel_stop", {})  # +10 min
    db.event(9000, "channel_start", {})                                      # killed: no stop event
    db.insert("band_seconds", {"ts": 9099, "band": "channel"})                # last row -> ends at 9100
    def em(ts, f, bw, dur=0.02, peak=-50.0, in_ch=0):
        db.insert("emissions", dict(ts=ts, ts_end=ts + dur, f_center_hz=f, bw_hz=bw, peak_dbfs=peak, above_floor_db=12.0,
                                    cls="narrowband", in_channel=in_ch, umsh=0))
    em(1100, 916_201_000, 2e3); em(1200, 916_299_000, 20e3, peak=-40.0); em(1300, 916_301_000, 2e3)
    em(1400, 917_500_000, 50e3, dur=0.5, in_ch=1); em(10, 916_201_000, 2e3)  # last one is before t0
    assert capture_seconds(db, 0, now=1e9) == 3600 + 100
    assert capture_seconds(db, 3900, now=1e9) == 100 + 600 + 100
    r = buckets(db, 1000)
    assert r["observed_s"] == 3700 and r["total"] == 4
    b0 = r["buckets"][0]
    assert (b0["f_lo_hz"], b0["f_hi_hz"], b0["bw_label"], b0["count"]) == (916_200_000, 916_300_000, "< 25", 2)
    assert abs(b0["per_hour"] - 2 * 3600 / 3700) < 1e-9 and b0["peak_max_dbfs"] == -40.0 and abs(b0["peak_dbfs"] + 45) < 1e-9
    wide = next(b for b in r["buckets"] if b["bw_label"] == "25–200")
    assert wide["in_channel"] and abs(wide["duty_pct"] - 100 * 0.5 / 3700) < 1e-9
    assert len(r["buckets"]) == 3
    assert bw_bucket_label(3) == "≥ 600" and bw_bucket_label(2) == "200–600"


def test_clear_endpoint_empties_emissions_and_resets_capture_time(tmp_path):
    import time
    from umsh_sdr import config as C
    from umsh_sdr.db import DB
    from umsh_sdr.dashboard import app as APP
    from umsh_sdr.emission_buckets import buckets
    s = C.Settings(data_dir=str(tmp_path))
    db = DB(s.db_path)
    now = time.time()
    db.event(now - 3600, "channel_start")
    for i in range(5):
        db.insert("emissions", {"ts": now - 100 + i, "ts_end": now - 99.99 + i, "f_center_hz": 916.45e6, "bw_hz": 5e3,
                                "peak_dbfs": -40, "above_floor_db": 12, "cls": "narrowband", "in_channel": 0, "umsh": 0})
    assert buckets(db, 0)["total"] == 5
    app = APP.create_app(s, None)
    clear = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/emissions/clear")
    assert clear() == {"ok": True, "deleted": 5}
    d = buckets(db, 0)
    assert d["total"] == 0 and d["observed_s"] < 5
