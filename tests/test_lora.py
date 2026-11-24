import pytest

from umsh_sdr import lora


def test_symbol_time_sf10_bw500():
    assert lora.symbol_time() == pytest.approx(2.048e-3)


def test_airtime_matches_semtech_calculator():
    # SF10 / 500 kHz / CR4/5 / CRC / explicit header, 32-symbol preamble, LDRO off.
    # 36 bytes: 8 + ceil((288-40+28+16)/40)*5 = 48 payload symbols.
    assert lora.airtime(36) == pytest.approx((32 + 4.25 + 48) * 2.048e-3)
    assert lora.airtime(36, preamble=8) == pytest.approx((8 + 4.25 + 48) * 2.048e-3)


def test_cfo_conversion():
    assert lora.cfo_hz(1, 0.0) == pytest.approx(500e3 / 1024)
    assert lora.cfo_hz(-2, 0.5) == pytest.approx(-1.5 * 500e3 / 1024)
