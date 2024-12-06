deep_sort_params = {
    # Відстеження
    'max_cosine_distance': 0.3,  # Залишаємо достатню дискримінацію для уникнення помилкових збігів
    'nn_budget': 100,  # Обмежуємо кількість ознак для швидшої обробки

    # Рух
    'max_iou_distance': 0.7,  # Збалансоване значення для високошвидкісних об'єктів
    'max_age': 30,  # Швидке видалення треків, які зникають
    'n_init': 3,  # Мінімальна кількість кадрів для підтвердження треку

    # Фільтрація
    'nms_max_overlap': 0.7,  # Підтримка об'єктів у густих сценах
    'half': True,
}

classes = ['person', 'bicycle', 'car', 'motorbike', 'bus', 'truck', 'bird', 'cat', 'dog']

# Параметри для ObjectDetection ndTracking
detection_params = ["yolov8n.pt", 1, 0.7]

