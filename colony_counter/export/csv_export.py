"""CSV export — semicolon-delimited, UTF-8-BOM for Excel compatibility."""
import csv


def export_csv(filepath, rows):
    """Export to CSV. rows = list of dicts from format_result_row()."""
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(["#", "Файл", "Авто", "Исключено", "Ручных", "ИТОГО",
                    "Одиночных", "Кластеров", "Ср.площадь_px",
                    "Ср.площадь_мм2", "CFU/мл", "Группа", "Разведение"])
        for idx, r in enumerate(rows, 1):
            w.writerow([idx, r['name'], r['auto'], r['excluded'],
                        r['manual'], r['total'], r['singles'], r['clusters'],
                        r['avg_area_px'], r['avg_area_mm2'], r['cfu'],
                        r['group'], r['dilution']])
