import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# URL HLS потока
hls_url = 'hls_url'

# Загрузка модели YOLOv8 с использованием CPU
# device = torch.device('cpu')
model = torch.model('yolov8n.pt', pretrained=True)

# Инициализация Deep SORT
tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0, max_iou_distance=0.9)

# Цвет для следов
track_colors = {}


# Функция для получения цвета по ID трека
def get_color(track_id):
    if track_id not in track_colors:
        track_colors[track_id] = (
        int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
    return track_colors[track_id]


# Захват видео с HLS потока
cap = cv2.VideoCapture(hls_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Применение модели YOLO для детекции
    results = model(frame)

    # Извлечение детекций
    detections = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        if cls == 2:  # Класс 2 соответствует "car" в модели YOLOv5
            detections.append((box, conf))

    # Обновляем трекер
    tracks = tracker.update_tracks(detections, frame=frame)

    # Отображаем треки и следы
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        bbox = track.to_tlbr()  # Получаем координаты bounding box (left, top, right, bottom)
        color = get_color(track_id)

        # Рисуем bounding box
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

        # Рисуем ID трека
        cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                    2)

    # Отображаем результат
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
