"""PDF export via matplotlib — summary table + per-image pages."""
import datetime

import cv2

from colony_counter.core.constants import C

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def export_pdf(filepath, rows, get_annotated_fn, results_dict, version=C.VERSION):
    """Export PDF report.
    rows: list of dicts from format_result_row()
    get_annotated_fn(path) -> BGR numpy array
    results_dict: {path: result_dict}
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib not installed")

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(filepath) as pdf:
        # Summary page
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis('off')
        ax.set_title(
            f"Colony Counter v{version}\n"
            f"{datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}",
            fontsize=16, fontweight='bold')

        td = [[i + 1, r['name'][:28], r['auto'], r['manual'], r['total'],
               r['avg_area_px']] for i, r in enumerate(rows)]
        col_labels = ["#", "Файл", "Авто", "Ручн.", "ИТОГО", "Ср.пл."]
        t = ax.table(cellText=td, colLabels=col_labels,
                     cellLoc='center', loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.5)
        for k, cell in t.get_celld().items():
            if k[0] == 0:
                cell.set_facecolor('#2E4DA0')
                cell.set_text_props(color='white', fontweight='bold')
        pdf.savefig(fig)
        plt.close(fig)

        # Per-image pages
        for r in rows:
            path = r['path']
            ann = get_annotated_fn(path)
            result = results_dict.get(path)
            if ann is None or result is None:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            fig.suptitle(r['name'], fontsize=14, fontweight='bold')

            axes[0].imshow(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Результат")
            axes[0].axis('off')

            areas = [c['feat']['area'] for c in result['colonies']]
            if areas:
                axes[1].hist(areas,
                             bins=min(30, max(5, len(areas) // 3)),
                             color='#2E4DA0', edgecolor='white', alpha=0.8)
                axes[1].set_title("Площади")
                axes[1].set_xlabel("пикс.")
                axes[1].set_ylabel("кол-во")
                axes[1].axvline(result['avg_colony_area'], color='red',
                                linestyle='--',
                                label=f"Ср.={result['avg_colony_area']:.0f}")
                axes[1].legend()
            else:
                axes[1].text(0.5, 0.5, "Нет", ha='center')
                axes[1].axis('off')

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
