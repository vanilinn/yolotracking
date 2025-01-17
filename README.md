# Трекинг При Помощи YOLO Ultralytics

Этот проект включает скрипты для трекинга объектов на видеопотоке с использованием модели YOLO и различных методов трекинга.

## Архитектура

### 1. `yolotrack.py`
Скрипт для трекинга определенных объектов на видеопотоке. Отслеживает пересечение двух настраиваемых линий (например, используется для отслеживания количества машин, повернувших в определенном направлении на конкретном перекрестке). Имеется возможность указывать, какие объекты трекать, а какие игнорировать.

### 2. `yolotrackgpu.py`
Скрипт для трекинга всех объектов YOLO на видеопотоке. Инференс производится на GPU.

### 3. `tracking.py`
Попытка реализовать трекинг при помощи YOLO и DeepSort.

## Стек Технологий

- Python 3.x
- OpenCV
- YOLO (You Only Look Once)
- DeepSort (для трекинга)
- GPU для ускорения инференса

## Установка и Запуск

### 1. Клонирование Репозитория

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
