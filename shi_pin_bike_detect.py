import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO

# # 加载 YOLOv8 模型
# model = YOLO(r'C:\Users\Micha\Desktop\p2\runs\detect\train25\weights\best.pt')

model = YOLO('yolov8x.pt')

# 加载类别名称
with open("coco2.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 视频文件路径
video_path = r"BJP1.mp4"  # 输入视频文件路径
output_path = "BJ_output"  # 输出视频保存路径

# 打开视频文件
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video {video_path}")
    exit()

# 获取视频属性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 定义视频输出格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"Processing video: {video_path}")
print("Press 'q' to stop processing.")

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

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
        if confidence > 0.7:
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

    # 绘制检测结果
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

    # 保存处理后的帧到输出视频
    out.write(frame)

    # 显示处理后的帧
    cv2.imshow("Video Detection", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to: {output_path}")
