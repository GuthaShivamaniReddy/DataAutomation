import pytest

from dataos.desktop.server import start_server_in_background, wait_for_health

_PORT = 8790


def test_server_starts_and_answers_health_check():
    start_server_in_background(port=_PORT)
    wait_for_health(port=_PORT, timeout=10)  # must not raise


def test_wait_for_health_times_out_when_nothing_is_listening():
    with pytest.raises(TimeoutError):
        wait_for_health(port=8799, timeout=1)
