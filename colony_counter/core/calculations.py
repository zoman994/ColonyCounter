"""Pure computation helpers — zero tkinter dependency."""
from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt


def grand_total(
    result: dict | None,
    excluded_set: set[tuple[int, int]],
    manual_marks_list: list[tuple[int, int]],
) -> tuple[int, int, int]:
    """Returns (auto_active, manual_count, excluded_count)."""
    if not result:
        return (0, 0, 0)
    aa = sum(
        len([c for c in col['ws_centers'] if c not in excluded_set])
        for col in result['colonies'])
    return (aa, len(manual_marks_list), len(excluded_set))


def px_per_mm(result: dict | None, dish_diameter_mm: float) -> float | None:
    """Pixels per mm based on detected dish radius and user-set diameter."""
    if not result:
        return None
    dish = result.get('dish')
    if not dish:
        return None
    r_px = dish[2]
    if dish_diameter_mm > 0 and r_px > 0:
        return (r_px * 2) / dish_diameter_mm
    return None


def calc_cfu_ml(colony_count: int, plating_volume_ml: float, dilution_factor: float) -> float | None:
    """Calculate CFU/ml.
    dilution_factor is the denominator: 1:100 → dilution_factor=100
    Formula: CFU/ml = count / (volume_ml * (1 / dilution_factor))
           = count * dilution_factor / volume_ml
    """
    if plating_volume_ml > 0 and dilution_factor >= 1:
        return colony_count * dilution_factor / plating_volume_ml
    return None


def classify_morphology(result: dict | None) -> dict[str, int]:
    """Classify colonies into size/shape categories."""
    if not result or not result.get('colonies'):
        return {}
    areas = [c['feat']['area'] for c in result['colonies']]
    circs = [c['feat']['circularity'] for c in result['colonies']]
    med_a = np.median(areas) if areas else 1
    return dict(
        small=sum(1 for a in areas if a < med_a * 0.5),
        medium=sum(1 for a in areas if med_a * 0.5 <= a <= med_a * 2.0),
        large=sum(1 for a in areas if a > med_a * 2.0),
        round=sum(1 for c in circs if c > 0.7),
        irregular=sum(1 for c in circs if c <= 0.7))


def make_annotated_image(
    result: dict | None,
    excluded_set: set[tuple[int, int]],
    marks_list: list[tuple[int, int]],
    annotations_list: list[str],
    dish_overrides: list | None,
    clean_img: npt.NDArray[np.uint8] | None,
) -> npt.NDArray[np.uint8] | None:
    """Rebuild annotated image with manual edits — pure OpenCV, no tkinter."""
    if result is None or clean_img is None:
        return None

    has_edits = bool(excluded_set or marks_list or annotations_list)
    if not has_edits:
        return None  # caller should use cached annotated

    base = clean_img.copy()
    cr = result.get('col_radius', 10)

    # Dish circles
    dishes_to_draw = dish_overrides or result.get('dishes', [result['dish']])
    for dcx, dcy, dr in dishes_to_draw:
        cv2.circle(base, (int(dcx), int(dcy)), int(dr), (80, 80, 220), 2)

    # Label overlays
    for dd in result.get('dish_results', []):
        if dd.get('has_label') and dd.get('label_mask') is not None:
            lc = np.zeros_like(base)
            lc[dd['label_mask'] > 0] = (0, 60, 180)
            cv2.addWeighted(base, 1.0, lc, 0.35, 0, base)
            lcs, _ = cv2.findContours(dd['label_mask'], cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base, lcs, -1, (0, 100, 255), 2)

    # Colony drawing
    auto_total = 0
    for col in result['colonies']:
        ac = [c for c in col['ws_centers'] if c not in excluded_set]
        ic = [c for c in col['ws_centers'] if c in excluded_set]
        if ac:
            is_cl = len(ac) > 1 or col['is_cluster']
            color = (20, 160, 255) if is_cl else (30, 200, 30)
            if not col['is_cluster']:
                cv2.drawContours(base, [col['contour']], -1, color, 2)
                cv2.circle(base, ac[0], 3, color, -1)
            else:
                for (ax, ay) in ac:
                    cv2.circle(base, (ax, ay), cr, color, 2)
            auto_total += len(ac)
        for (ex, ey) in ic:
            cv2.drawMarker(base, (ex, ey), (0, 0, 200),
                           cv2.MARKER_TILTED_CROSS, cr * 2, 2, cv2.LINE_AA)

    # Manual marks
    mark_r = max(2, cr // 3)
    for (mx, my) in marks_list:
        cv2.circle(base, (mx, my), mark_r, (0, 215, 255), 1)
        cv2.drawMarker(base, (mx, my), (0, 215, 255),
                       cv2.MARKER_CROSS, mark_r * 2, 1, cv2.LINE_AA)

    # Grand total label
    g = auto_total + len(marks_list)
    lbl = f"Kolonii: {g}"
    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(base, (6, 6), (lw + 18, lh + 18), (30, 30, 30), -1)
    cv2.putText(base, lbl, (12, lh + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 220, 60), 2, cv2.LINE_AA)

    # Text annotations
    h_img = base.shape[0]
    for ai, atxt in enumerate(annotations_list):
        y_pos = h_img - 30 - ai * 28
        cv2.putText(base, atxt, (12, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return base


def format_result_row(
    path: str, result: dict, display_name: str,
    auto_active: int, manual_n: int, excluded_n: int,
    ppm: float | None = None, cfu: float | None = None,
    dilution_group: str = "", dilution_factor: float = 1,
) -> dict:
    """Standard data row for export (Excel/CSV/PDF)."""
    singles = result['colony_count'] - result['cluster_count']
    grand = auto_active + manual_n
    area_mm2 = ""
    if ppm and ppm > 0.01:
        area_mm2 = round(result['avg_colony_area'] / (ppm * ppm), 4)
    cfu_str = f"{cfu:.0f}" if cfu and cfu > 0 else ""
    return dict(
        name=display_name, auto=result['total'],
        excluded=excluded_n, manual=manual_n, total=grand,
        singles=singles, clusters=result['cluster_count'],
        avg_area_px=round(result['avg_colony_area'], 1),
        avg_area_mm2=area_mm2, cfu=cfu_str,
        group=dilution_group, dilution=dilution_factor,
        path=path.split('::frame')[0] if '::frame' in path else path)
