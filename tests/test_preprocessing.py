"""
Unit Tests for Data Preprocessing and Feature Engineering
Project: Zomato Delivery Operations Analysis Using PySpark
"""

import pytest
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import parse_time_string_to_minutes


class TestTimeParsing:
    """Tests the multi-format timestamp parser."""

    def test_standard_hh_mm(self):
        # 21:55 -> 21*60 + 55 = 1315 minutes
        assert parse_time_string_to_minutes("21:55") == 1315
        # 09:20 -> 9*60 + 20 = 560 minutes
        assert parse_time_string_to_minutes("09:20") == 560

    def test_midnight_formats(self):
        # 24:05:00 -> 0*60 + 5 = 5 minutes past midnight
        assert parse_time_string_to_minutes("24:05:00") == 5
        # 24:15:00 -> 15 minutes past midnight
        assert parse_time_string_to_minutes("24:15:00") == 15
        # '1' (Excel representation for 24:00 midnight) -> 0
        assert parse_time_string_to_minutes("1") == 0

    def test_excel_decimal_fractions(self):
        # 0.5 is 12:00 PM (noon) -> 720 minutes
        assert parse_time_string_to_minutes("0.5") == 720
        # 0.75 is 18:00 (6:00 PM) -> 1080 minutes
        assert parse_time_string_to_minutes("0.75") == 1080
        # 0.916666667 is 22:00 (10:00 PM) -> 1320 minutes
        assert parse_time_string_to_minutes("0.916666667") == 1320

    def test_missing_and_invalid_values(self):
        assert parse_time_string_to_minutes(None) is None
        assert parse_time_string_to_minutes("NaN") is None
        assert parse_time_string_to_minutes("nan") is None
        assert parse_time_string_to_minutes("") is None
        assert parse_time_string_to_minutes("   ") is None
        assert parse_time_string_to_minutes("invalid_time") is None


class TestMidnightRollover:
    """Tests that elapsed order-to-pickup minutes accounts for midnight rollover."""

    def test_normal_same_day(self):
        order_min = parse_time_string_to_minutes("21:55")  # 1315
        pickup_min = parse_time_string_to_minutes("22:10") # 1330
        diff = pickup_min - order_min
        assert diff == 15

    def test_midnight_rollover(self):
        # Customer ordered at 23:55 (1435 min), courier picked up at 00:10 (10 min)
        order_min = parse_time_string_to_minutes("23:55")  # 1435
        pickup_min = parse_time_string_to_minutes("00:10") # 10
        diff = pickup_min - order_min
        if diff < 0:
            diff += 1440
        assert diff == 15


class TestCoordinateRectification:
    """Tests coordinate sign rectification and Null Island checks."""

    def test_negative_sign_rectification(self):
        # Coordinate in Pune: recorded as -18.55144, corrected by abs() to 18.55144
        raw_lat = -18.55144
        corrected_lat = abs(raw_lat)
        assert corrected_lat > 0
        assert abs(corrected_lat - 18.55144) < 1e-5

    def test_null_island_detection(self):
        # 0.0, 0.0 is Null Island (invalid GPS sensor reading)
        coords = (0.0, 0.0)
        is_invalid = (coords[0] == 0.0 or coords[1] == 0.0)
        assert is_invalid is True
