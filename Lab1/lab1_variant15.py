"""
Лабораторная работа 1: Основы работы с цифровыми изображениями в OpenCV
Вариант V = 15
"""

import cv2
import numpy as np
import os

# ============================================================================
# 5.1. ПОДГОТОВКА СРЕДЫ И ИМПОРТ
# ============================================================================
print("="*70)
print("ЛАБОРАТОРНАЯ РАБОТА 1: ОБРАБОТКА ЦИФРОВЫХ ИЗОБРАЖЕНИЙ")
print("Вариант V = 15")
print("="*70)

print("\n5.1. ПОДГОТОВКА СРЕДЫ И ИМПОРТ")
print("-" * 70)

print(f"OpenCV версия: {cv2.__version__}")
print(f"NumPy версия: {np.__version__}")


# ============================================================================
# 5.2. ЗАГРУЗКА/ЧТЕНИЕ ЦВЕТНОГО ИЗОБРАЖЕНИЯ
# ============================================================================
print("\n5.2. ЗАГРУЗКА/ЧТЕНИЕ ЦВЕТНОГО ИЗОБРАЖЕНИЯ")
print("-" * 70)

# Параметры для варианта 15
V = 15

# Вычисляем параметры по формулам
fx_small = 0.30 + 0.01 * V
fy_small = 0.30 + 0.005 * V
fx_big = 1.20 + 0.01 * V
fy_big = 1.20 + 0.005 * V

# Поворот: V mod 3
rotation_mod = V % 3
if rotation_mod == 0:
    rotation_angle = cv2.ROTATE_90_CLOCKWISE
    rotation_text = "90° по часовой стрелке"
elif rotation_mod == 1:
    rotation_angle = cv2.ROTATE_180
    rotation_text = "180°"
else:
    rotation_angle = cv2.ROTATE_90_COUNTERCLOCKWISE
    rotation_text = "90° против часовой стрелки"

# Beta для яркости: V mod 5
beta = (V % 5) * 20

# Размеры холста
H = 240 + 5 * V
W = 320 + 3 * V

print(f"\nВычисленные параметры варианта {V}:")
print(f"  fx_small = 0.30 + 0.01×{V} = {fx_small}")
print(f"  fy_small = 0.30 + 0.005×{V} = {fy_small}")
print(f"  fx_big = 1.20 + 0.01×{V} = {fx_big}")
print(f"  fy_big = 1.20 + 0.005×{V} = {fy_big}")
print(f"  Поворот: {rotation_text}")
print(f"  Beta (яркость) = ({V} mod 5) × 20 = {beta}")
print(f"  H (высота холста) = 240 + 5×{V} = {H}")
print(f"  W (ширина холста) = 320 + 3×{V} = {W}")

# Чтение изображения
image_path = "b1dbe0982164625d9272f3895c4c132e.jpg"

if not os.path.exists(image_path):
    print(f"\nОшибка: файл '{image_path}' не найден!")
    exit(1)

img = cv2.imread(image_path, cv2.IMREAD_COLOR)

if img is None:
    print(f"\nОшибка: не удалось прочитать изображение '{image_path}'!")
    exit(1)

print(f"\nИсходное изображение успешно загружено: {image_path}")


# ============================================================================
# 5.3. КОНТРОЛЬ СВОЙСТВ ИЗОБРАЖЕНИЯ
# ============================================================================
print("\n5.3. КОНТРОЛЬ СВОЙСТВ ИЗОБРАЖЕНИЯ")
print("-" * 70)

print(f"Shape исходного изображения: {img.shape}")
print(f"Dtype исходного изображения: {img.dtype}")

# Проверяем пиксель в центре изображения
y, x = img.shape[0] // 2, img.shape[1] // 2
pixel_value = img[y, x]
print(f"\nПиксель в средней позиции [{y}, {x}]:")
print(f"  Значение: {pixel_value}")
print(f"  Порядок каналов в OpenCV: BGR (Blue, Green, Red)")
print(f"  B={pixel_value[0]}, G={pixel_value[1]}, R={pixel_value[2]}")


# ============================================================================
# 5.4. ОТОБРАЖЕНИЕ ИСХОДНОГО ИЗОБРАЖЕНИЯ
# ============================================================================
print("\n5.4. ОТОБРАЖЕНИЕ ИСХОДНОГО ИЗОБРАЖЕНИЯ")
print("-" * 70)

# Сохраняем исходное изображение для отчета
cv2.imwrite(f"out_original_{V}.png", img)
print(f"Исходное изображение сохранено: out_original_{V}.png")


# ============================================================================
# 5.5. ПРЕОБРАЗОВАНИЕ В ОТТЕНКИ СЕРОГО И СОХРАНЕНИЕ
# ============================================================================
print("\n5.5. ПРЕОБРАЗОВАНИЕ В ОТТЕНКИ СЕРОГО И СОХРАНЕНИЕ")
print("-" * 70)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Shape grayscale изображения: {gray.shape}")
print(f"Dtype grayscale изображения: {gray.dtype}")
print("Grayscale имеет форму (H, W), то есть 1 канал (двумерный массив)")

cv2.imwrite(f"out_gray_{V}.png", gray)
print(f"\nРезультат сохранён: out_gray_{V}.png")


# ============================================================================
# 5.6. МАСШТАБИРОВАНИЕ (УМЕНЬШЕНИЕ И УВЕЛИЧЕНИЕ)
# ============================================================================
print("\n5.6. МАСШТАБИРОВАНИЕ (УМЕНЬШЕНИЕ И УВЕЛИЧЕНИЕ)")
print("-" * 70)

# Уменьшение
img_small = cv2.resize(img, None, fx=fx_small, fy=fy_small,
                       interpolation=cv2.INTER_LINEAR)
print(f"\nУменьшение изображения:")
print(f"  Коэффициенты: fx={fx_small}, fy={fy_small}")
print(f"  Исходный размер: {img.shape[:2]}")
print(f"  Новый размер: {img_small.shape[:2]}")
print(f"  Эффект: потеря деталей, сглаживание")

cv2.imwrite(f"out_resize_small_{V}.png", img_small)
print(f"  Сохранено: out_resize_small_{V}.png")

# Увеличение
img_big = cv2.resize(img, None, fx=fx_big, fy=fy_big,
                     interpolation=cv2.INTER_LINEAR)
print(f"\nУвеличение изображения:")
print(f"  Коэффициенты: fx={fx_big}, fy={fy_big}")
print(f"  Исходный размер: {img.shape[:2]}")
print(f"  Новый размер: {img_big.shape[:2]}")
print(f"  Эффект: интерполяция, возможное размытие, артефакты")

cv2.imwrite(f"out_resize_big_{V}.png", img_big)
print(f"  Сохранено: out_resize_big_{V}.png")


# ============================================================================
# 5.7. ГЕОМЕТРИЧЕСКОЕ ПРЕОБРАЗОВАНИЕ (ПОВОРОТ)
# ============================================================================
print("\n5.7. ГЕОМЕТРИЧЕСКОЕ ПРЕОБРАЗОВАНИЕ (ПОВОРОТ)")
print("-" * 70)

img_rotated = cv2.rotate(img, rotation_angle)

print(f"\nПоворот изображения:")
print(f"  Угол поворота: {rotation_text}")
print(f"  Исходный shape: {img.shape}")
print(f"  Shape после поворота: {img_rotated.shape}")

if rotation_angle == cv2.ROTATE_90_CLOCKWISE or rotation_angle == cv2.ROTATE_90_COUNTERCLOCKWISE:
    print(f"  Примечание: при повороте на 90° высота и ширина поменялись местами")

cv2.imwrite(f"out_rotate_{V}.png", img_rotated)
print(f"  Сохранено: out_rotate_{V}.png")


# ============================================================================
# 5.8. РАБОТА С КАНАЛАМИ
# ============================================================================
print("\n5.8. РАБОТА С КАНАЛАМИ")
print("-" * 70)

# Разделение на каналы
b, g, r = cv2.split(img)

print(f"\nРазделение на каналы:")
print(f"  Canal B (Blue) shape: {b.shape}")
print(f"  Canal G (Green) shape: {g.shape}")
print(f"  Canal R (Red) shape: {r.shape}")

# Собираем с перестановкой каналов (BGR -> RGB)
img_swap = cv2.merge([r, g, b])

print(f"\nПеречастановка каналов (BGR -> RGB):")
print(f"  Исходный порядок: B, G, R")
print(f"  Новый порядок: R, G, B")
print(f"  Shape после merge: {img_swap.shape}")
print(f"  Результат: цвета инвертированы в RGB (синий становится красным и наоборот)")

cv2.imwrite(f"out_swap_channels_{V}.png", img_swap)
print(f"  Сохранено: out_swap_channels_{V}.png")


# ============================================================================
# 6. ГЕНЕРАЦИЯ УНИКАЛЬНОГО ЦВЕТНОГО ИЗОБРАЖЕНИЯ (ХОЛСТ)
# ============================================================================
print("\n6. ГЕНЕРАЦИЯ УНИКАЛЬНОГО ЦВЕТНОГО ИЗОБРАЖЕНИЯ (ХОЛСТ)")
print("-" * 70)

# Создание белого холста
canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

print(f"\nПараметры холста:")
print(f"  Размер (H x W): {H} x {W}")
print(f"  Shape: {canvas.shape}")
print(f"  Тип данных: {canvas.dtype}")
print(f"  Цвет фона: белый (255, 255, 255)")

# Рисование примитивов
# Линия (синяя)
pt1 = (50, 50)
pt2 = (W - 50, H - 50)
color_blue = (255, 0, 0)  # BGR
thickness_line = 2
cv2.line(canvas, pt1, pt2, color_blue, thickness_line)
print(f"\n1. Линия:")
print(f"   От {pt1} до {pt2}")
print(f"   Цвет BGR: {color_blue} (синий)")
print(f"   Толщина: {thickness_line}")

# Прямоугольник (зелёный)
x1, y1 = 100, 100
x2, y2 = W - 100, H // 2
color_green = (0, 255, 0)  # BGR
thickness_rect = 2
cv2.rectangle(canvas, (x1, y1), (x2, y2), color_green, thickness_rect)
print(f"\n2. Прямоугольник:")
print(f"   Левый верхний угол: ({x1}, {y1})")
print(f"   Правый нижний угол: ({x2}, {y2})")
print(f"   Цвет BGR: {color_green} (зелёный)")
print(f"   Толщина: {thickness_rect}")

# Окружность (красная)
center = (W // 2, H // 2)
radius = 60
color_red = (0, 0, 255)  # BGR
thickness_circle = 2
cv2.circle(canvas, center, radius, color_red, thickness_circle)
print(f"\n3. Окружность:")
print(f"   Центр: {center}")
print(f"   Радиус: {radius}")
print(f"   Цвет BGR: {color_red} (красный)")
print(f"   Толщина: {thickness_circle}")

# Текстовая надпись
text = f"Variant {V}"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
color_text = (0, 0, 0)  # BGR (чёрный)
thickness_text = 2
text_pos = (W // 2 - 70, H - 50)
cv2.putText(canvas, text, text_pos, font, font_scale, color_text, thickness_text)
print(f"\n4. Текстовая надпись:")
print(f"   Текст: '{text}'")
print(f"   Позиция: {text_pos}")
print(f"   Цвет BGR: {color_text} (чёрный)")
print(f"   Шрифт: FONT_HERSHEY_SIMPLEX")
print(f"   Размер шрифта: {font_scale}")
print(f"   Толщина: {thickness_text}")

cv2.imwrite(f"out_canvas_{V}.png", canvas)
print(f"\nХолст с примитивами сохранён: out_canvas_{V}.png")


# ============================================================================
# 7. РАСШИРЕННАЯ ЧАСТЬ - ЯРКОСТЬ (Вариант A)
# ============================================================================
print("\n7. РАСШИРЕННАЯ ЧАСТЬ - ЯРКОСТЬ И КОНТРАСТ")
print("-" * 70)

# Увеличение яркости
alpha_brightness = 1.0
brighter = cv2.convertScaleAbs(img, alpha=alpha_brightness, beta=beta)

print(f"\nУвеличение яркости:")
print(f"  Формула: dst = |src × {alpha_brightness} + {beta}|")
print(f"  Beta (параметр яркости) = ({V} mod 5) × 20 = {beta}")
print(f"  Использование convertScaleAbs предотвращает переполнение значений")
print(f"  (при выходе за диапазон 0-255 результат насыщается)")

cv2.imwrite(f"out_brightness_{V}.png", brighter)
print(f"  Сохранено: out_brightness_{V}.png")

# Изменение контраста
alpha_contrast = 1.5
contrast = cv2.convertScaleAbs(img, alpha=alpha_contrast, beta=0)

print(f"\nИзменение контраста:")
print(f"  Формула: dst = |src × {alpha_contrast} + 0|")
print(f"  Alpha = {alpha_contrast} увеличивает контраст")
print(f"  Светлые области становятся светлее, тёмные - темнее")
print(f"  convertScaleAbs обеспечивает безопасность при выходе за границы диапазона")

cv2.imwrite(f"out_contrast_{V}.png", contrast)
print(f"  Сохранено: out_contrast_{V}.png")

print(f"\nПримечание о convertScaleAbs:")
print(f"  - Безопасен от переполнения целого типа")
print(f"  - Автоматически насыщает значения в диапазоне [0, 255]")
print(f"  - Предпочтительнее, чем простое сложение img + beta")
print(f"  - Обеспечивает предсказуемые результаты")


# ============================================================================
# ОБОБЩЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "="*70)
print("ОБРАБОТКА ЗАВЕРШЕНА")
print("="*70)

print("\nСохраненные файлы:")
output_files = [
    f"out_original_{V}.png",
    f"out_gray_{V}.png",
    f"out_resize_small_{V}.png",
    f"out_resize_big_{V}.png",
    f"out_rotate_{V}.png",
    f"out_swap_channels_{V}.png",
    f"out_canvas_{V}.png",
    f"out_brightness_{V}.png",
    f"out_contrast_{V}.png",
]

for i, filename in enumerate(output_files, 1):
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        print(f"  {i}. {filename} ({file_size} байт)")

print("\nВариант: 15")
print("Все пункты задания выполнены успешно!")
print("="*70)
