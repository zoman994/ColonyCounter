"""Tests for colony_counter.core.processing."""
import numpy as np
import cv2
import pytest

from colony_counter.core.processing import ImageProcessor


@pytest.fixture
def processor():
    return ImageProcessor()


class TestDishDetection:
    def test_fallback_on_blank_image(self):
        gray = np.zeros((500, 500), dtype=np.uint8)
        cx, cy, r = ImageProcessor.detect_dish(gray)
        assert 200 <= cx <= 300
        assert 200 <= cy <= 300
        assert r > 50

    def test_detects_bright_circle(self):
        gray = np.zeros((600, 600), dtype=np.uint8)
        cv2.circle(gray, (300, 300), 200, 200, -1)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        cx, cy, r = ImageProcessor.detect_dish(gray)
        assert abs(cx - 300) < 50
        assert abs(cy - 300) < 50
        assert abs(r - 200) < 80


class TestContourFeatures:
    def test_circle_features(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 30, 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        feat = ImageProcessor.contour_features(contours[0])
        assert feat['area'] > 2000
        assert feat['circularity'] > 0.8
        assert 0.8 < feat['aspect_ratio'] < 1.2
        assert feat['solidity'] > 0.9


class TestEstimateSingleColonyArea:
    def test_unimodal(self):
        areas = np.array([100, 110, 95, 105, 98, 102, 108, 97, 103, 101], dtype=float)
        est = ImageProcessor.estimate_single_colony_area(areas)
        assert 80 < est < 120

    def test_with_clusters(self):
        # 20 singles ~100px + 3 clusters ~500px
        singles = np.random.normal(100, 10, 20).clip(50, 200)
        clusters = np.array([500, 600, 550])
        areas = np.concatenate([singles, clusters]).astype(float)
        est = ImageProcessor.estimate_single_colony_area(areas)
        # Should find the single-colony peak, not be pulled by clusters
        assert 60 < est < 150

    def test_few_areas(self):
        areas = np.array([100.0, 200.0])
        est = ImageProcessor.estimate_single_colony_area(areas)
        assert est == pytest.approx(150.0)


class TestFillContourWithCircles:
    def test_single_point(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = ImageProcessor.fill_contour_with_circles(contours[0], 1, 5)
        assert len(pts) == 1
        assert 30 < pts[0][0] < 70

    def test_multiple_points(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 50, 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = ImageProcessor.fill_contour_with_circles(contours[0], 5, 10)
        assert len(pts) == 5

    def test_zero_points(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert ImageProcessor.fill_contour_with_circles(contours[0], 0, 5) == []


class TestSplitCluster:
    def test_single_blob(self):
        enhanced = np.zeros((200, 200), dtype=np.uint8)
        binary = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(enhanced, (100, 100), 20, 200, -1)
        cv2.circle(binary, (100, 100), 20, 255, -1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n, centers = ImageProcessor.split_cluster(contours[0], enhanced, binary, 1200.0, 200, 200)
        assert n >= 1
        assert len(centers) >= 1

    def test_two_blobs_merged(self):
        enhanced = np.zeros((200, 200), dtype=np.uint8)
        binary = np.zeros((200, 200), dtype=np.uint8)
        # Two bright circles bridged together
        cv2.circle(enhanced, (80, 100), 25, 200, -1)
        cv2.circle(enhanced, (130, 100), 25, 200, -1)
        cv2.rectangle(enhanced, (80, 90), (130, 110), 150, -1)
        cv2.circle(binary, (80, 100), 25, 255, -1)
        cv2.circle(binary, (130, 100), 25, 255, -1)
        cv2.rectangle(binary, (80, 90), (130, 110), 255, -1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n, centers = ImageProcessor.split_cluster(contours[0], enhanced, binary, 1800.0, 200, 200)
        assert n >= 2
        assert len(centers) >= 2
