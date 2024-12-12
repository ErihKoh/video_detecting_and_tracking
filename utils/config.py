deep_sort_params = {
    # Відстеження

    'max_cosine_distance': 0.3,  # Залишаємо достатню дискримінацію для уникнення помилкових збігів
    'nn_budget': 1000,  # Обмежуємо кількість ознак для швидшої обробки

    # Рух
    'max_iou_distance': 1,  # Збалансоване значення для високошвидкісних об'єктів
    'max_age': 30,  # Швидке видалення треків, які зникають
    'n_init': 3,  # Мінімальна кількість кадрів для підтвердження треку

    # Фільтрація
    'nms_max_overlap': 0.7,  # Підтримка об'єктів у густих сценах
    'half': True,
}

classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog']

# Параметри для ObjectDetection ndTracking
detection_params = ["/Users/erihkoh/GitHub/video_detecting_and_tracking/yolov8m.pt", 0.8]
video_source = 1

# color
colors = {'RED': (0, 0, 255),
          'GREEN': (0, 255, 0),
          'BLUE': (255, 0, 0),
          'WHITE': (255, 255, 255),
          }
