# Colony Counter

Автоматический подсчёт колоний бактерий на чашках Петри.
Портативное десктоп-приложение (Windows .exe, не требует установки Python).

## Что умеет

- **Детекция чашки** — Hough circles, авто-определение границ
- **Подсчёт колоний** — бинаризация + CLAHE + морфология + watershed
- **Кластеры** — разделение слипшихся колоний (watershed через skimage)
- **Этикетки** — обнаружение и маскирование наклеек на чашках
- **Калибровка** — пиксели → мм через диаметр чашки
- **CFU/мл** — расчёт концентрации (объём посева × разведение)
- **Серийные разведения** — кривая разведений с графиком
- **Undo/Redo** — Ctrl+Z / Ctrl+Y для всех ручных правок
- **Экспорт** — Excel (.xlsx), CSV, PDF (с графиками), PNG/JPG
- **Сессии** — сохранение/загрузка всего состояния работы
- **Тёмная/светлая тема** — переключение одной кнопкой

## Скриншот

Тёмная тема, обработанное изображение с колониями:

```
◉ ColonyCounter v2.1
┌──────────┬──────────────────────┬──────────────┐
│ Файлы    │  [Результат]         │  Обнаружение │
│          │                      │  Фильтры     │
│ img1 234 │   🟢 одиночные       │  Калибровка  │
│ img2 --- │   🔵 кластеры        │  CFU/мл      │
│          │   ❌ исключённые      │  Пресеты     │
│ ИТОГО    │   🟡 ручные          │  Статистика  │
└──────────┴──────────────────────┴──────────────┘
```

## Запуск

### Портативный .exe (Windows, без Python)

Скачайте `ColonyCounterV2.exe` из [Releases](../../releases) и запустите.

### Из исходников

```bash
pip install -e .
python -m colony_counter
```

### Зависимости

```
opencv-python >= 4.8
numpy >= 1.24
Pillow >= 10.0
openpyxl >= 3.1
scikit-image >= 0.22
matplotlib >= 3.8
```

## Сборка .exe

```bash
pip install pyinstaller
python -m PyInstaller --onefile --windowed --name ColonyCounterV2 \
  --collect-submodules=matplotlib --collect-submodules=skimage \
  --collect-submodules=colony_counter colony_counter/__main__.py
```

Результат: `dist/ColonyCounterV2.exe` (~115 MB)

## Структура проекта

```
colony_counter/
  __init__.py          — версия
  __main__.py          — точка входа
  app.py               — контроллер UI
  core/
    constants.py       — именованные константы
    io_utils.py        — Unicode-safe imread/imwrite + TIFF
    processing.py      — ImageProcessor (детекция, watershed, фильтрация)
    cache.py           — ленивый кэш изображений на диске
    learning.py        — адаптивная коррекция порога (EMA)
    calculations.py    — CFU, калибровка, морфология
    session.py         — сохранение/загрузка сессий
  export/
    excel_export.py    — .xlsx
    csv_export.py      — .csv
    pdf_export.py      — PDF с графиками
    image_export.py    — PNG/JPG/BMP
  ui/
    theme.py           — тёмная/светлая тема
    widgets.py         — DarkButton, DarkCheck, DarkSlider, DarkSection
    logo.py            — встроенный логотип (base64)
```

`core/` не импортирует tkinter — можно тестировать и использовать без GUI.

## Горячие клавиши

| Клавиша | Действие |
|---------|----------|
| Ctrl+O | Добавить файлы |
| Ctrl+S | Экспорт Excel |
| Ctrl+E | Сохранить изображение |
| Ctrl+Z / Ctrl+Y | Undo / Redo |
| Space | Обработать текущее |
| Left / Right | Навигация по списку |
| Delete | Удалить из списка |
| F1-F4 | Пресеты параметров |

## Алгоритм обработки

1. Загрузка и масштабирование (макс. 2000px)
2. Детекция чашки Петри (Hough circles)
3. Обнаружение этикетки (тёмные + опционально светлые)
4. Вычитание фона (морфологическое открытие 71×71)
5. CLAHE — локальный контраст
6. Бинаризация по порогу (ручной или Otsu)
7. Морфологическая очистка (OPEN + CLOSE)
8. Фильтрация контуров (площадь, округлость, аспект, выпуклость)
9. Оценка площади одиночной колонии (log-гистограмма)
10. Watershed для разделения кластеров (skimage peak_local_max)
11. Оценка скрытых колоний под этикеткой

## Лицензия

GPL-3.0 — см. [LICENSE](LICENSE)

## RES lab — Fungal Biotechnology
