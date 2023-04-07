# 基于pLSA的报告分析系统

## Brief Intro

​	pLSA（probabilistic latent semantic analysis）是一种基于概率模型的文本主题模型。它可以将文本数据转换成一组主题，每个主题由一组单词构成，从而可以对文本数据进行聚类、分类、推荐等操作。

---

## 文件声明

### 程序文件

vect_div_0322.pkl：是将文本段分词向量化的模型文件，基于[jieba]([jieba · PyPI](https://pypi.org/project/jieba/))的[paddle]([飞桨PaddlePaddle-源于产业实践的开源深度学习平台](https://www.paddlepaddle.org.cn/feature))模式

pLSA_div_0322.pkl：进行潜在语义分析的模型文件，基于sklearn.decomposition.LatentDirichletAllocation包实现

FNN_div_0322.pkl：三层全连接神经网络模型，input文档-主题矩阵，output关于标签label的数据

stopwords.txt：程序运行需要的文件，内含中文常见分割词，用于文本处理

dependence.txt：含有我conda环境的包配置

test/：用于放置待分析的pdf文档（目前可能只支持放一个pdf的情况）

new_extra.py：从test/目录读取pdf文档，在./目录下生成一个predict.txt，里面是文本化的pdf

plsa_ana2.py：从./目录读取predict.txt和三个模型文件

​	plsa_ana2.py *workflow*:

1. 首先对predict.txt中每一条（来自pdf每一页）进行分词(依赖vect_div_0322.pkl)
2. 再将文档--词向量信息转化为文档--主题信息（依赖pLSA_div_0322.pkl）
3. 最后将文档--主题信息投入FNN预测标签（依赖FNN_div_0322.pkl）
4. 整理结果，生成result_dict.json，里面包含了结果信息

main.py：会依次调用new_extra.py和plsa_ana2.py

### 进行分析需要的文件

将一个pdf文件放在test/目录下即可

### 结果分析生成的文件

./predict.txt：文本化的pdf文件

**./result_dict.json**：结果文件

---

## 文件结果说明

### 字典中的内容即为程序结果，存储在result_dict.json中

result_dict示例:

{0: [[], []], 

1: [[6], [1.0]], 

2: [[28, 29, 30, 35, 36, 38, 41, 89], [1.0, 0.7, 0.3, 1.0, 0.9, 0.9, 1.0, 0.0]],

3: [[28, 34, 36, 38], [1.0, 1.0, 1.0, 0.9]], 

4: [[15, 17, 18, 19, 20, 24, 44, 49], [1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 1.0, 1.0]], 

5: [[15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 46, 63, 65, 88], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.3, 0.5, 0.7]], 

6: [[52, 79], [0.7, 1.0]], 

7: [[], []], 

8: [[], []], 

9: [[], []], 

10: [[], []], 

11: [[6], [1.0]]}

### 字典中每一个pair代表： 内容标签:[[有关此内容的页码],[这些页码属于此内容标签的confidence(0~1.0)]]

0：内容标签释义

1：是否存在利益相关方对企业的评价

2：董事会是否发挥监控作用

3：是否对员工健康及安全的保护措施

4：是否有完善的人才培养措施

5：生态保护措施

6：节能减排措施

7：公益项目参与

8：乡村振兴项目参与

9：群体列表

10：与利益相关方沟通情况

11：管治基本架构