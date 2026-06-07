import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_ip_to_int():
    from src.data_processing import ip_to_int
    assert ip_to_int("192.168.1.1") == 3232235777
    assert ip_to_int("0.0.0.0") == 0
    assert ip_to_int("invalid") == 0
    print("ip_to_int tests passed!")


def test_ip_to_int_known():
    from src.data_processing import ip_to_int
    result = ip_to_int("1.0.0.0")
    assert result == 16777216
    print("ip_to_int known value test passed!")
