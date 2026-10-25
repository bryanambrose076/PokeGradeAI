import pytest
import requests

from pokegrade import pricing
from pokegrade.pricing import grading_roi, get_card_prices


# grading_roi: profit = PSA 10 price - raw price - grading fee for that value tier
@pytest.mark.parametrize("raw_price, psa10_price, expected", [
    (10, 100, 63),      # profitable, cheapest tier ($27 fee)
    (50, 60, -17),      # loses money: should stay negative, not clamp to 0
    (0, 500, 473),      # top edge of the $27 tier
    (0, 501, 435),      # just over 500 moves to the $66 tier
    (100, 800, 634),    # $66 tier
    (0, 3000, 2620),    # highest tier ($380 fee)
])
def test_grading_roi(raw_price, psa10_price, expected):
    assert grading_roi(raw_price, psa10_price) == pytest.approx(expected)


@pytest.mark.parametrize("raw_price, psa10_price", [
    (None, 100),
    (10, None),
    (None, None),
])
def test_grading_roi_missing_price(raw_price, psa10_price):
    assert grading_roi(raw_price, psa10_price) is None


# get_card_prices: fake the API so tests never use real requests (or the daily quota)
class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self.json_data


def fake_api(monkeypatch, response):
    monkeypatch.setattr(pricing.requests, "get", lambda *args, **kwargs: response)


def test_get_card_prices_reads_raw_psa9_psa10(monkeypatch):
    fake_api(monkeypatch, FakeResponse({"data": [{
        "prices": {"market": 2.28},
        "ebay": {"salesByGrade": {
            "psa9": {"smartMarketPrice": {"price": 15.38}},
            "psa10": {"smartMarketPrice": {"price": 82}},
        }},
    }]}))
    assert get_card_prices("Okidogi ex", "Shrouded Fable", "082", "key") == (2.28, 15.38, 82)


def test_get_card_prices_no_graded_sales(monkeypatch):
    fake_api(monkeypatch, FakeResponse({"data": [{"prices": {"market": 0.18}, "ebay": {"salesByGrade": {}}}]}))
    assert get_card_prices("Kakuna", "151", "014", "key") == (0.18, None, None)


def test_get_card_prices_card_not_found(monkeypatch):
    fake_api(monkeypatch, FakeResponse({"data": []}))
    assert get_card_prices("Nobody", "Nowhere", "000", "key") is None


def test_get_card_prices_api_error_raises(monkeypatch):
    # pipeline.scan_image catches this and returns prices as null
    fake_api(monkeypatch, FakeResponse({}, status_code=401))
    with pytest.raises(requests.RequestException):
        get_card_prices("Okidogi ex", "Shrouded Fable", "082", "bad-key")
