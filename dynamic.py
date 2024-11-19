import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO('yolov8x.pt')

# 加载类别名称
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

print("Press 'q' to exit.")

while True:
    # 从摄像头读取帧
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image from camera.")
        break

    height, width, channels = frame.shape

    # 执行推理
    results = model(frame)[0]

    # 初始化检测结果
    class_ids = []
    confidences = []
    boxes = []

    # 解析检测结果
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = result

        # 过滤置信度低的检测
        if confidence > 0.3:
            # 计算边界框参数
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)

            # 保存检测结果
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))

    # 使用非最大抑制（NMS）去除重复框
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 初始化一个计数器来记录类别出现次数
    detected_objects = []

    # 输出检测结果
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            detected_objects.append(label)

            # 绘制边界框和类别标签
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 显示检测结果
    cv2.imshow("Real-Time Object Detection", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
