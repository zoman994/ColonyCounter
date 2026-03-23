"""Named constants for the colony detection pipeline."""
from colony_counter import VERSION


class C:
    VERSION = VERSION
    MAX_IMAGE_DIM = 2000            # Max image side (px); larger → resize

    # Hough Circle Transform (dish detection)
    HOUGH_BLUR_KERNEL = (21, 21)    # Gaussian blur before Hough
    HOUGH_MIN_R_RATIO = 0.28        # Min radius = 28% of min(h, w)
    HOUGH_MAX_R_RATIO = 0.56        # Max radius = 56% of min(h, w)
    HOUGH_FALLBACK_R_RATIO = 0.44   # Fallback if Hough fails
    HOUGH_PARAM1 = 60               # Canny upper threshold
    HOUGH_PARAM2 = 35               # Accumulator threshold

    # Dish mask
    DISH_MASK_RATIO = 0.96          # Use 96% of radius (margin from edge)

    # Background normalisation
    BG_MORPH_KERNEL = 71            # Morphology kernel for background estimate (px)

    # CLAHE (contrast)
    CLAHE_CLIP = 3.0
    CLAHE_TILE = (8, 8)

    # Binary morphology cleanup
    MORPH_KERNEL = 3                # Open/Close kernel after thresholding

    # Dark label detection
    LABEL_DARK_THRESH = 60          # Brightness < 60 → dark region
    LABEL_MIN_AREA = 0.02           # Min label area = 2% of dish
    LABEL_MAX_AREA = 0.40           # Max label area = 40% of dish
    LABEL_MIN_ASPECT = 2.5          # Min aspect ratio (labels are elongated)
    LABEL_MIN_FILL = 0.4            # Min fill ratio in bounding rect
    LABEL_DILATE_K = 30             # Dilation kernel for label mask (px)

    # Light/transparent label detection
    LABEL_LIGHT_THRESH = 200        # Brightness > 200 → bright region
    LABEL_LIGHT_STD_THRESH = 15     # Local std < 15 → uniform area
    LABEL_LIGHT_MIN_ASPECT = 2.0
    LABEL_LIGHT_MIN_FILL = 0.45

    # Colony filtering
    BUBBLE_CIRC = 0.90              # Circularity > 0.9 + large = air bubble
    BUBBLE_AREA_MULT = 1.8          # Bubble if area > max_area × 1.8
    MAX_ASPECT = 6.0                # Max aspect ratio for colony
    MIN_ASPECT = 0.16               # Min aspect ratio (= 1/6.25)
    ELONGATION_THRESH = 3.5         # minAreaRect ratio for elongation filter
    SOLIDITY_THRESH = 0.45          # Min solidity (area / convexHull)

    # Cluster splitting
    CLUSTER_AREA_MULT = 1.8         # area > avg × 1.8 → cluster candidate

    # Single colony area estimation (log-histogram)
    LOG_HIST_MAX_BINS = 30
    LOG_HIST_MIN_BINS = 10

    # Watershed (legacy, still used in split_cluster)
    WS_MIN_DIST_FACTOR = 0.5        # min_distance = radius × 0.5
    WS_THRESH_FACTOR = 0.2          # threshold_abs = dist.max() × 0.2
    WS_SANITY_LO = 0.5              # ws_count ≥ area_estimate × 0.5
    WS_SANITY_HI = 1.5              # ws_count ≤ area_estimate × 1.5

    # Cluster splitting (new: image-based, replaces hex grid)
    SPLIT_ADAPTIVE_C = 3             # Subtracted from adaptive threshold
    SPLIT_MIN_FRAGMENT = 0.15        # Min fragment area as fraction of avg_area
    SPLIT_DIST_THRESH = 0.25         # Distance transform threshold (fraction of max)

    # Adaptive learning (EMA)
    LEARN_ALPHA = 0.30               # EMA coefficient
    LEARN_MIN_AUTO = 5               # Min auto-colonies to learn from
    LEARN_MIN_RATIO = 0.04           # Min |excluded−added|/auto for correction
    LEARN_MAX_DELTA = 5              # Max threshold change per iteration
    LEARN_DELTA_K = 12               # Ratio → delta scaling factor

    # Zoom
    ZOOM_MIN = 0.25
    ZOOM_MAX = 10.0
    ZOOM_FACTOR = 1.15               # Zoom step per scroll tick

    # HSV color filter
    HSV_S_LO = 30                    # Min Saturation for color filter

    # Otsu scaling
    # Empirical: ~48% of Otsu threshold on CLAHE-enhanced image gives
    # optimal FP/FN balance on standard agar.  For dark media (starch agar)
    # may need 0.55–0.65.
    OTSU_SCALE = 0.48
