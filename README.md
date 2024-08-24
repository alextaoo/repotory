YOLO，全称为“You Only Look Once”（你只看一眼），是一种流行的实时目标检测系统，由Joseph Redmon等人在2015年提出。
YOLO模型的核心思想是将目标检测任务视为一个单一的回归问题，通过一个卷积神经网络（CNN）直接从图像像素到边界框坐标和类别概率的映射。
YOLO模型经过了多次迭代，包括YOLOv2（YOLO9000）、YOLOv3和YOLOv4等版本，每个版本都在性能和速度上有所提升，同时也引入了一些新的技术，如更深的网络结构、更好的锚框机制、多尺度特征融合等。

YOLO使用的标注格式是每张图像一个文本文件，文件名与图像文件名相对应。文本文件中每一行对应一个边界框，格式为：<class> <x_center> <y_center> <width> <height>。
其中，<class>是类别索引，<x_center>和<y_center>是边界框中心点相对于图像宽度和高度的比例，<width>和<height>是边界框的宽度和高度相对于图像宽度和高度的比例。

# 读取训练集视频
for anno_path, video_path in zip(train_annos[:5], train_videos[:5]):
    print(video_path)
    anno_df = pd.read_json(anno_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0 
    # 读取视频所有画面
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_height, img_width = frame.shape[:2]
        
        # 将画面写为图
        frame_anno = anno_df[anno_df['frame_id'] == frame_idx]
        cv2.imwrite('./yolo-dataset/train/' + anno_path.split('/')[-1][:-5] + '_' + str(frame_idx) + '.jpg', frame)

        # 如果存在标注
        if len(frame_anno) != 0:
            with open('./yolo-dataset/train/' + anno_path.split('/')[-1][:-5] + '_' + str(frame_idx) + '.txt', 'w') as up:
                for category, bbox in zip(frame_anno['category'].values, frame_anno['bbox'].values):
                    category_idx = category_labels.index(category)
                    
                    # 计算yolo标注格式
                    x_min, y_min, x_max, y_max = bbox
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    if x_center > 1:
                        print(bbox)
                    up.write(f'{category_idx} {x_center} {y_center} {width} {height}\n')
        
        frame_idx += 1


Ultraalytics 是一个提供多种计算机视觉模型的库，包括 YOLO 系列。这段代码是一个简单的训练启动示例。


from ultralytics import YOLO

# 设置模型版本
model = YOLO("yolov8n.pt") 

# 设定数据集和训练参数
results = model.train(data="yolo-dataset/yolo.yaml", epochs=2, imgsz=1080, batch=16)


- box_loss 是边界框回归损失，用于评估预测的边界框与真实边界框之间的差异。
- cls_loss 是分类损失，用于评估类别预测的准确性。
- dfl_loss 是防御性损失，用于提高模型的泛化能力。

从输出结果来看，经过两个训练周期后，模型的边界框损失、分类损失和防御性损失都有所下降，这表明模型在训练过程中学习了如何更好地预测边界框和分类。同时，模型的 mAP50 和 mAP50-95 指标也有所提高，这表明模型的整体性能有所提升






        
