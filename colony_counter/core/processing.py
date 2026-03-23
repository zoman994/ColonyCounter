"""Colony detection and image processing pipeline."""
from __future__ import annotations

import math
from typing import Optional, Callable

import cv2
import numpy as np
import numpy.typing as npt

from colony_counter.core.constants import C
from colony_counter.core.io_utils import cv_imread

try:
    from skimage.feature import peak_local_max
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class ImageProcessor:
    """All image processing algorithms — zero tkinter dependency."""

    # ── Hough helper ─────────────────────────────────────────────────────
    @staticmethod
    def _hough_circles(gray):
        h, w = gray.shape
        blurred = cv2.GaussianBlur(gray, C.HOUGH_BLUR_KERNEL, 0)
        min_r = int(min(h, w) * C.HOUGH_MIN_R_RATIO)
        max_r = int(min(h, w) * C.HOUGH_MAX_R_RATIO)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_r,
            param1=C.HOUGH_PARAM1, param2=C.HOUGH_PARAM2,
            minRadius=min_r, maxRadius=max_r)
        return circles, min_r, max_r

    @staticmethod
    def detect_dish(gray):
        circles, _, _ = ImageProcessor._hough_circles(gray)
        h, w = gray.shape
        if circles is not None:
            circles = sorted(np.round(circles[0]).astype(int).tolist(),
                             key=lambda c: c[2], reverse=True)
            return tuple(circles[0])
        return (w // 2, h // 2, int(min(h, w) * C.HOUGH_FALLBACK_R_RATIO))

    @staticmethod
    def detect_dishes(gray, max_dishes=2):
        circles, _, _ = ImageProcessor._hough_circles(gray)
        h, w = gray.shape
        if circles is None:
            return [(w // 2, h // 2, int(min(h, w) * C.HOUGH_FALLBACK_R_RATIO))]
        circles = sorted(np.round(circles[0]).astype(int).tolist(),
                         key=lambda c: c[2], reverse=True)
        result = []
        for cx, cy, r in circles:
            if all(((cx - rx)**2 + (cy - ry)**2)**0.5 >= max(r, rr) * 0.7
                   for rx, ry, rr in result):
                result.append((int(cx), int(cy), int(r)))
            if len(result) >= max_dishes:
                break
        return result or [(w // 2, h // 2, int(min(h, w) * C.HOUGH_FALLBACK_R_RATIO))]

    # ── Background ───────────────────────────────────────────────────────
    @staticmethod
    def normalize_background(gray, mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (C.BG_MORPH_KERNEL, C.BG_MORPH_KERNEL))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, k)
        bg = cv2.morphologyEx(bg, cv2.MORPH_ERODE, k)
        diff = cv2.subtract(bg, gray)
        return cv2.bitwise_and(diff, diff, mask=mask)

    # ── Label detection ──────────────────────────────────────────────────
    @staticmethod
    def detect_label_mask_dark(gray, dish_mask):
        masked = cv2.bitwise_and(gray, gray, mask=dish_mask)
        _, dark = cv2.threshold(masked, C.LABEL_DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
        dark = cv2.bitwise_and(dark, dark, mask=dish_mask)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        label_mask = np.zeros(gray.shape, dtype=np.uint8)
        dish_area = cv2.countNonZero(dish_mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < dish_area * C.LABEL_MIN_AREA or area > dish_area * C.LABEL_MAX_AREA:
                continue
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw * rh == 0:
                continue
            aspect = max(rw, rh) / (min(rw, rh) + 1e-5)
            fill = area / (rw * rh)
            if aspect > C.LABEL_MIN_ASPECT and fill > C.LABEL_MIN_FILL:
                cv2.fillConvexPoly(label_mask, np.intp(cv2.boxPoints(rect)), 255)
        if cv2.countNonZero(label_mask) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (C.LABEL_DILATE_K, C.LABEL_DILATE_K))
            label_mask = cv2.dilate(label_mask, k, iterations=2)
        return label_mask

    @staticmethod
    def detect_label_mask_light(gray, dish_mask):
        masked = cv2.bitwise_and(gray, gray, mask=dish_mask)
        _, bright = cv2.threshold(masked, C.LABEL_LIGHT_THRESH, 255, cv2.THRESH_BINARY)
        bright = cv2.bitwise_and(bright, bright, mask=dish_mask)
        blur = cv2.GaussianBlur(gray.astype(np.float32), (31, 31), 0)
        blur2 = cv2.GaussianBlur((gray.astype(np.float32))**2, (31, 31), 0)
        local_std = np.sqrt(np.maximum(blur2 - blur**2, 0))
        uniform = (local_std < C.LABEL_LIGHT_STD_THRESH).astype(np.uint8) * 255
        combined = cv2.bitwise_and(bright, uniform)
        combined = cv2.bitwise_and(combined, combined, mask=dish_mask)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        label_mask = np.zeros(gray.shape, dtype=np.uint8)
        dish_area = cv2.countNonZero(dish_mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < dish_area * C.LABEL_MIN_AREA or area > dish_area * C.LABEL_MAX_AREA:
                continue
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw * rh == 0:
                continue
            aspect = max(rw, rh) / (min(rw, rh) + 1e-5)
            fill = area / (rw * rh)
            if aspect > C.LABEL_LIGHT_MIN_ASPECT and fill > C.LABEL_LIGHT_MIN_FILL:
                cv2.fillConvexPoly(label_mask, np.intp(cv2.boxPoints(rect)), 255)
        if cv2.countNonZero(label_mask) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (C.LABEL_DILATE_K, C.LABEL_DILATE_K))
            label_mask = cv2.dilate(label_mask, k, iterations=2)
        return label_mask

    @staticmethod
    def detect_label_mask(gray, dish_mask, detect_light=False):
        dark = ImageProcessor.detect_label_mask_dark(gray, dish_mask)
        if detect_light:
            light = ImageProcessor.detect_label_mask_light(gray, dish_mask)
            return cv2.bitwise_or(dark, light)
        return dark

    # ── Colony area estimation ───────────────────────────────────────────
    @staticmethod
    def estimate_single_colony_area(areas):
        if len(areas) < 3:
            return float(np.median(areas))
        log_a = np.log(areas[areas > 0])
        n_bins = min(C.LOG_HIST_MAX_BINS, max(C.LOG_HIST_MIN_BINS, len(log_a) // 3))
        counts, edges = np.histogram(log_a, bins=n_bins)
        pk = int(np.argmax(counts))
        lo = edges[max(0, pk - 1)]
        hi = edges[min(len(edges) - 1, pk + 2)]
        pa = areas[(log_a >= lo) & (log_a <= hi)]
        if len(pa) == 0:
            return float(np.exp((edges[pk] + edges[pk + 1]) / 2))
        return float(np.median(pa))

    # ── Watershed ────────────────────────────────────────────────────────
    @staticmethod
    def watershed_per_component(cnt, avg_area: float, h: int, w: int) -> int:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() < 2:
            return 1
        er = max(3, int(np.sqrt(avg_area / np.pi)))
        md = max(3, int(er * C.WS_MIN_DIST_FACTOR))
        ta = max(2.0, float(dist.max()) * C.WS_THRESH_FACTOR)
        if HAS_SKIMAGE:
            coords = peak_local_max(dist, min_distance=md, threshold_abs=ta, labels=mask)
            return max(1, len(coords))
        # OpenCV fallback: distance threshold → connectedComponents
        _, thresh_map = cv2.threshold(dist, ta, 255, cv2.THRESH_BINARY)
        thresh_map = thresh_map.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, md), max(3, md)))
        thresh_map = cv2.erode(thresh_map, kernel, iterations=1)
        n_labels, _ = cv2.connectedComponents(thresh_map)
        return max(1, n_labels - 1)  # subtract background

    # ── Contour features ─────────────────────────────────────────────────
    @staticmethod
    def contour_features(cnt):
        area = cv2.contourArea(cnt)
        per = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area / (per * per)) if per > 0 else 0.0
        x, y, bw, bh = cv2.boundingRect(cnt)
        ar = bw / bh if bh > 0 else 1.0
        hull_a = cv2.contourArea(cv2.convexHull(cnt))
        sol = area / hull_a if hull_a > 0 else 1.0
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else x + bw // 2
        cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else y + bh // 2
        return dict(area=area, circularity=circ, aspect_ratio=ar,
                    solidity=sol, cx=cx, cy=cy)

    # ── Cluster grid fill ────────────────────────────────────────────────
    @staticmethod
    def fill_contour_with_circles(cnt, n, col_radius):
        if n <= 0:
            return []
        x, y, bw, bh = cv2.boundingRect(cnt)
        if n == 1:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                return [(int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))]
            return [(x + bw // 2, y + bh // 2)]
        r = max(2, col_radius)
        dy = max(1, int(r * 1.732))
        dx = max(1, int(r * 2.0))
        m = np.zeros((bh + 2, bw + 2), dtype=np.uint8)
        sh = cnt.copy()
        sh[:, :, 0] -= x
        sh[:, :, 1] -= y
        cv2.drawContours(m, [sh], -1, 255, -1)
        cands = []
        for row, yi in enumerate(range(r, bh, dy)):
            ox = r if row % 2 else 0
            for xi in range(ox, bw, dx):
                if yi < m.shape[0] and xi < m.shape[1] and m[yi, xi] > 0:
                    cands.append((xi + x, yi + y))
        if not cands:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                return [(int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))]
            return [(x + bw // 2, y + bh // 2)]
        if len(cands) <= n:
            return cands
        step = (len(cands) - 1) / max(1, n - 1)
        return [cands[min(int(i * step), len(cands) - 1)] for i in range(n)]

    # ── Color filter ─────────────────────────────────────────────────────
    @staticmethod
    def color_filter_mask(img_bgr, work_mask):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        wp = gray[work_mask > 0]
        if len(wp) == 0:
            return work_mask
        bg_med = np.median(wp)
        dark = (gray < bg_med * 0.7).astype(np.uint8) * 255
        sat = (hsv[:, :, 1] > C.HSV_S_LO).astype(np.uint8) * 255
        combined = cv2.bitwise_or(dark, sat)
        return cv2.bitwise_and(combined, combined, mask=work_mask)

    # ── Per-dish pipeline ────────────────────────────────────────────────
    def _process_single_dish(self, img, gray, cx, cy, r, params):
        h, w = img.shape[:2]
        dish_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(dish_mask, (cx, cy), max(1, int(r * C.DISH_MASK_RATIO)), 255, -1)

        # Calibrate min/max area from mm using REAL dish radius
        dish_mm = params.get('dish_diameter_mm', 90.0)
        if dish_mm > 0 and r > 0:
            real_px_per_mm = (2 * r) / dish_mm
        else:
            real_px_per_mm = 10.0  # fallback
        min_diam = params.get('min_diam_mm', 0.3)
        max_diam = params.get('max_diam_mm', 3.0)
        if min_diam > 0 and max_diam > 0:
            min_r_px = min_diam / 2.0 * real_px_per_mm
            max_r_px = max_diam / 2.0 * real_px_per_mm
            params = dict(params)  # don't mutate original
            params['min_area'] = max(1, int(math.pi * min_r_px * min_r_px))
            params['max_area'] = max(params['min_area'] + 1, int(math.pi * max_r_px * max_r_px))

        has_label, label_mask, hidden_est = False, None, 0
        if bool(params.get('detect_label', True)):
            detect_light = bool(params.get('detect_light_label', False))
            label_mask = self.detect_label_mask(gray, dish_mask, detect_light=detect_light)
            has_label = cv2.countNonZero(label_mask) > 0

        work_mask = dish_mask.copy()
        if has_label and label_mask is not None:
            work_mask = cv2.bitwise_and(work_mask, cv2.bitwise_not(label_mask))

        normalized = self.normalize_background(gray, work_mask)
        clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_TILE)
        enhanced = clahe.apply(normalized)

        if bool(params.get('use_otsu', False)):
            wp = enhanced[work_mask > 0]
            if len(wp) > 0:
                ov, _ = cv2.threshold(wp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh_val = max(5, int(ov * C.OTSU_SCALE))
            else:
                thresh_val = int(params['threshold'])
        else:
            thresh_val = int(params['threshold'])

        _, binary = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_and(binary, binary, mask=work_mask)

        if bool(params.get('use_color_filter', False)):
            cm = self.color_filter_mask(img, work_mask)
            k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.bitwise_and(binary, cv2.dilate(cm, k_dil, iterations=1))

        km = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (C.MORPH_KERNEL, C.MORPH_KERNEL))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, km, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, km, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a = max(1, int(params['min_area']))
        max_a = max(min_a + 1, int(params['max_area']))
        fb = bool(params['filter_bubbles'])
        uw = bool(params['use_watershed'])
        fe = bool(params['filter_elongated'])
        fn = bool(params['filter_nonconvex'])

        raw = []
        for cnt in contours:
            feat = self.contour_features(cnt)
            a = feat['area']
            if a < min_a:
                continue
            if fb and feat['circularity'] > C.BUBBLE_CIRC and a > max_a * C.BUBBLE_AREA_MULT:
                continue
            ar = feat['aspect_ratio']
            if ar > C.MAX_ASPECT or ar < C.MIN_ASPECT:
                continue
            if fe and a > min_a * 3:
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                if min(rw, rh) > 0 and max(rw, rh) / min(rw, rh) > C.ELONGATION_THRESH:
                    continue
            if fn and a > min_a * 5 and feat['solidity'] < C.SOLIDITY_THRESH:
                continue
            raw.append(dict(contour=cnt, feat=feat))

        all_areas = np.array([o['feat']['area'] for o in raw if o['feat']['area'] <= max_a], dtype=float)
        if len(all_areas) >= 3:
            avg_area = self.estimate_single_colony_area(all_areas)
        elif len(all_areas) > 0:
            avg_area = float(np.median(all_areas))
        else:
            avg_area = float(min_a * 5)
        avg_area = max(avg_area, float(min_a))
        ct = avg_area * C.CLUSTER_AREA_MULT
        col_r = max(4, int(np.sqrt(avg_area / np.pi)))

        colonies = []
        total = 0
        for obj in raw:
            cnt = obj['contour']
            feat = obj['feat']
            a = feat['area']
            if a <= ct:
                est = 1
                wsc = [(feat['cx'], feat['cy'])]
            else:
                ae = max(1, round(a / avg_area))
                if uw:
                    wc = self.watershed_per_component(cnt, avg_area, h, w)
                    est = wc if (wc >= 2 and wc <= ae * C.WS_SANITY_HI
                                 and wc >= ae * C.WS_SANITY_LO) else ae
                else:
                    est = ae
                wsc = self.fill_contour_with_circles(cnt, est, col_r)
            if not wsc:
                wsc = [(feat['cx'], feat['cy'])]
            total += est
            colonies.append(dict(
                contour=cnt, feat=feat,
                center=(feat['cx'], feat['cy']),
                ws_centers=wsc, estimated=est,
                is_cluster=(est > 1)))

        if has_label:
            da = cv2.countNonZero(dish_mask)
            wa = cv2.countNonZero(work_mask)
            if wa > 0 and wa < da * 0.95:
                hidden_est = int((total / wa) * (da - wa))

        return dict(
            total=total, colony_count=len(colonies),
            cluster_count=sum(1 for c in colonies if c['is_cluster']),
            avg_colony_area=avg_area, col_radius=col_r,
            dish=(cx, cy, r), colonies=colonies,
            binary=binary, enhanced=enhanced,
            has_label=has_label, hidden_estimate=hidden_est,
            label_mask=label_mask, work_mask=work_mask,
            threshold_used=thresh_val)

    # ── Main pipeline ────────────────────────────────────────────────────
    def process(self, image_path, params, progress_cb=None):
        if progress_cb:
            progress_cb(0.05, "Загрузка...")
        img = cv_imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось открыть: {image_path}")

        h0, w0 = img.shape[:2]
        scale = 1.0
        if max(h0, w0) > C.MAX_IMAGE_DIM:
            scale = C.MAX_IMAGE_DIM / max(h0, w0)
            img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)),
                             interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if progress_cb:
            progress_cb(0.15, "Чашки...")
        ov = params.get('dish_overrides')
        dishes = ([(int(x), int(y), int(r)) for x, y, r in ov]
                  if ov else self.detect_dishes(gray))

        if progress_cb:
            progress_cb(0.25, "Обработка...")
        annotated = img.copy()
        all_col = []
        dish_res = []
        sn = bool(params.get('show_numbers', True))

        for di, (dcx, dcy, dr) in enumerate(dishes):
            if progress_cb:
                progress_cb(0.25 + 0.55 * (di / max(1, len(dishes))),
                            f"Чашка {di + 1}...")
            dd = self._process_single_dish(img, gray, dcx, dcy, dr, params)
            for c in dd['colonies']:
                c['dish_idx'] = di
            all_col.extend(dd['colonies'])
            dish_res.append(dd)

            # Annotate
            cv2.circle(annotated, (dcx, dcy), dr, (80, 80, 220), 2)
            if dd['has_label'] and dd['label_mask'] is not None:
                lc = np.zeros_like(annotated)
                lc[dd['label_mask'] > 0] = (0, 60, 180)
                cv2.addWeighted(annotated, 1.0, lc, 0.35, 0, annotated)
                lcs, _ = cv2.findContours(dd['label_mask'], cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated, lcs, -1, (0, 100, 255), 2)

            cr = dd['col_radius']
            for col in dd['colonies']:
                if col['is_cluster']:
                    for cx_c, cy_c in col['ws_centers']:
                        cv2.circle(annotated, (cx_c, cy_c), cr, (20, 160, 255), 2)
                    if sn:
                        cc = col['center']
                        cv2.putText(annotated, str(col['estimated']),
                                    (cc[0] - 8, cc[1] - cr - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 100, 220), 2, cv2.LINE_AA)
                else:
                    cv2.drawContours(annotated, [col['contour']], -1,
                                     (30, 200, 30), 2)
                    if sn:
                        cv2.circle(annotated, col['center'], 3, (30, 200, 30), -1)

            if len(dishes) > 1:
                cv2.putText(annotated, f"#{di + 1}: {dd['total']}",
                            (dcx - 30, max(15, dcy - dr - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (80, 180, 255), 2, cv2.LINE_AA)

        if progress_cb:
            progress_cb(0.85, "Аннотация...")

        gt = sum(d['total'] for d in dish_res)
        lbl = f"Kolonii: {gt}"
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.rectangle(annotated, (6, 6), (lw + 18, lh + 18), (30, 30, 30), -1)
        cv2.putText(annotated, lbl, (12, lh + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 220, 60), 2, cv2.LINE_AA)

        avg_a = (float(np.mean([d['avg_colony_area'] for d in dish_res]))
                 if dish_res else float(params['min_area'] * 5))

        if progress_cb:
            progress_cb(1.0, "Готово")

        return dict(
            total=gt, colony_count=len(all_col),
            cluster_count=sum(1 for c in all_col if c['is_cluster']),
            avg_colony_area=avg_a,
            col_radius=max(4, int(np.sqrt(avg_a / np.pi))),
            dish=dishes[0] if dishes else (w // 2, h // 2, int(min(h, w) * C.HOUGH_FALLBACK_R_RATIO)),
            dishes=dishes, dish_results=dish_res,
            colonies=all_col, annotated=annotated, img_clean=img.copy(),
            binary=dish_res[0]['binary'] if dish_res else np.zeros((h, w), np.uint8),
            enhanced=dish_res[0]['enhanced'] if dish_res else np.zeros((h, w), np.uint8),
            scale=scale,
            has_label=any(d['has_label'] for d in dish_res),
            hidden_estimate=sum(d['hidden_estimate'] for d in dish_res),
            label_mask=dish_res[0]['label_mask'] if dish_res else None)
