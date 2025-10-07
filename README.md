# mnist_project
这是基于上一个项目的CNN项目的完整补充版，包含了全流程的训练、推理、模型保存、加载预测。这也是首个完整的项目，也证明了我对于深度学习框架和推理部署有了一定的理解和心得，对于后续的学习有至关重要的作用

Conv2d(1, 32, 3, 1) → Conv2d(32, 64, 3, 1) → MaxPool2d(2)
→ Linear(9216, 128) → Dropout(0.25) → Linear(128, 10)


# 项目功能说明
使用 torchvision.datasets.MNIST 加载 MNIST 数据集

构建一个两层卷积网络（CNN）：
Conv2d(1, 32, 3, 1) → Conv2d(32, 64, 3, 1) → MaxPool2d(2)
→ Linear(9216, 128) → Dropout(0.25) → Linear(128, 10)

使用交叉熵损失函数 nn.CrossEntropyLoss()

使用 Adam 优化器训练模型

训练 5 个 epoch，打印平均损失

保存训练好的模型参数为 mnist_cnn.pth

编写独立的推理脚本 inference.py，可对单张图片进行分类预测

# 使用方法

训练模型
python train.py
训练完成后会生成模型文件：
mnist_cnn.pth

推理测试
python inference.py sample.png
预测类别：5

# 项目结构

```bash
mnist_project/
├── train.py          # 模型训练脚本
├── inference.py      # 推理脚本
├── model.py          # CNN 模型定义
├── mnist_cnn.pth     # 训练好的模型权重
├── sample.png        # 推理测试样例
└── README.md         # 项目说明文档
```


# 运行结果展示 

<img width="1194" height="245" alt="image" src="https://github.com/user-attachments/assets/abdd4b0c-ae9b-40c6-9271-63feb85134aa" />

<img width="524" height="40" alt="image" src="https://github.com/user-attachments/assets/8a9b59e5-3122-487f-a410-00e276242e31" />


