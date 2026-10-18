from aq import config, ingest


def test_no_single_day_laqn_requests(monkeypatch):
    urls = []

    def fake_get(url, params=None, retries=3):
        urls.append(url)
        return {"RawAQData": {"Data": []}}

    monkeypatch.setattr(ingest, "_get", fake_get)
    monkeypatch.setattr(config, "START_DATE", "2026-08-15")
    monkeypatch.setattr(config, "END_DATE", "2026-10-01")   # ends on the 1st of a month
    try:
        ingest.fetch_no2()
    except RuntimeError:
        pass                                                # the fake returns no data: expected
    assert urls
    for url in urls:
        start = url.split("StartDate=")[1].split("/")[0]
        end = url.split("EndDate=")[1].split("/")[0]
        assert start != end