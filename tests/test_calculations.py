"""Tests for colony_counter.core.calculations."""
import numpy as np
import pytest

from colony_counter.core.calculations import (
    grand_total, px_per_mm, calc_cfu_ml, classify_morphology,
    make_annotated_image, format_result_row,
)


class TestGrandTotal:
    def test_empty_result(self):
        assert grand_total(None, set(), []) == (0, 0, 0)

    def test_no_edits(self):
        result = {'colonies': [
            {'ws_centers': [(10, 10), (20, 20)]},
            {'ws_centers': [(30, 30)]},
        ]}
        aa, mn, en = grand_total(result, set(), [])
        assert aa == 3
        assert mn == 0
        assert en == 0

    def test_with_exclusions(self):
        result = {'colonies': [
            {'ws_centers': [(10, 10), (20, 20)]},
        ]}
        excl = {(10, 10)}
        aa, mn, en = grand_total(result, excl, [(50, 50)])
        assert aa == 1  # only (20,20) active
        assert mn == 1  # one manual mark
        assert en == 1  # one excluded


class TestPxPerMm:
    def test_standard_dish(self):
        result = {'dish': (500, 500, 450)}  # radius 450px
        ppm = px_per_mm(result, 90.0)  # 90mm dish
        assert ppm == pytest.approx(10.0, abs=0.01)

    def test_none_result(self):
        assert px_per_mm(None, 90.0) is None

    def test_zero_diameter(self):
        result = {'dish': (500, 500, 450)}
        assert px_per_mm(result, 0) is None


class TestCalcCfuMl:
    def test_basic(self):
        # 50 colonies, 0.1ml, 1:100 → 50000
        assert calc_cfu_ml(50, 0.1, 100) == pytest.approx(50000.0)

    def test_no_dilution(self):
        # 200 colonies, 0.1ml, 1:1 → 2000
        assert calc_cfu_ml(200, 0.1, 1) == pytest.approx(2000.0)

    def test_zero_volume(self):
        assert calc_cfu_ml(50, 0, 100) is None

    def test_high_dilution(self):
        # 10 colonies, 0.1ml, 1:1000000 → 100_000_000
        assert calc_cfu_ml(10, 0.1, 1000000) == pytest.approx(1e8)


class TestClassifyMorphology:
    def test_empty(self):
        assert classify_morphology(None) == {}
        assert classify_morphology({'colonies': []}) == {}

    def test_mixed_colonies(self):
        result = {'colonies': [
            {'feat': {'area': 10, 'circularity': 0.95}},   # small, round
            {'feat': {'area': 100, 'circularity': 0.85}},  # medium, round
            {'feat': {'area': 100, 'circularity': 0.60}},  # medium, irregular
            {'feat': {'area': 500, 'circularity': 0.30}},  # large, irregular
        ]}
        m = classify_morphology(result)
        assert m['small'] >= 1
        assert m['large'] >= 1
        assert m['round'] >= 2
        assert m['irregular'] >= 1


class TestFormatResultRow:
    def test_basic(self):
        result = {
            'total': 42, 'colony_count': 40, 'cluster_count': 5,
            'avg_colony_area': 150.0,
        }
        row = format_result_row(
            '/path/to/img.jpg', result, 'img.jpg',
            auto_active=38, manual_n=4, excluded_n=2)
        assert row['name'] == 'img.jpg'
        assert row['total'] == 42
        assert row['singles'] == 35
        assert row['clusters'] == 5

    def test_tiff_path_stripped(self):
        row = format_result_row(
            '/path/to/img.tif::frame0',
            {'total': 1, 'colony_count': 1, 'cluster_count': 0, 'avg_colony_area': 10},
            'test', 1, 0, 0)
        assert '::frame' not in row['path']
