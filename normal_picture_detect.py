import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO('yolov8x.pt')


# 加载类别名称
with open("coco2.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 加载输入图像
image_path = "plane_count.jpg"  # 替换为你的图片路径
image = cv2.imread(image_path)
height, width, channels = image.shape

# 执行推理
results = model(image)[0]

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
    print("Detected objects in the image:")
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        print(f"Class: {label}, Confidence: {confidence:.2f}, Box: ({x}, {y}, {w}, {h})")
        detected_objects.append(label)

        # 绘制边界框和类别标签
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 显示图像
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No objects detected in the image.")

# 输出识别到的类别和数量
if detected_objects:
    object_counts = Counter(detected_objects)
    print("\nSummary of detected objects:")
    for obj, count in object_counts.items():
        print(f"{obj}: {count}")
else:
    print("\nNo objects detected in the image.")
