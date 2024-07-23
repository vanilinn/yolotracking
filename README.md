## Трекинг при помощи YOLO Ultralytics

* yolotrack.py - скрипт для трекинга определенных объектов на видеопотоке, отслеживается персечение двух настраиваемых линий (например, используется для отслеживания количества машин, повернувших в определенном направлении на конкретном перекрестке), имеется возможность указывать, какие объекты трекать, а какие игнорировать
* yolotrackgpu.py - скрипт для трекинга всех объектов yolo на видеопотоке, инференс производится на gpu
* tracking.py - попытка реализовать трекинг при помощи yolo и DeepSort