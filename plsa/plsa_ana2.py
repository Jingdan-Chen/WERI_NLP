import re
import os
# import glob
# print("numpy import")
import numpy as np
# print("jieba import")
import jieba
# print("sklearn import")
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import joblib
# print("torch import")
import torch
import torch.nn as nn
import torch.optim as optim

# print("paddle jieba")
#use jieba_paddle
import paddle
paddle.enable_static()
jieba.enable_paddle()
import json
print("Doing Initialization!")

model_path = "./strategy/plsa/"
auf_path = model_path
encoder = "utf-8"
sep = "_!_"
file_path = model_path
train_name = "train.txt"
test_name = "test.txt"
predict_name = "predict.txt"
# val_name = "val.txt"
joblib_file = "pLSA_div_0322.pkl"
FNN_file = "FNN_div_0322.pkl"
vect_file = "vect_div_0322.pkl"
div_flag=False
# PLSA args
num_topics = 30
max_iterations = 10000 
load_lda_flag=True
pop_none_flag=True
load_fnn_flag=True
doing_predict = True
        
def divlize_label(lis,div=False):
    if div:
        if lis==[12]:
            return [0]
        else:
            return [1]
    else:
        return  lis

def de_pun(text):
    # 只保留中文
    pattern = r'[^\u4e00-\u9fff]'
    # Use the pattern to remove all matches from the text
    res_text = re.sub(pattern, '', text)
    return res_text

def dec2lis(num):
    num = int(num)
    bin_num = bin(num)[2:]
    return [i for i in range(len(bin_num)) if bin_num[::-1][i]=="1"]

def lis2dec(lis):
    res=0
    for i in lis:
        i=int(i)
        res+=2**i
    return res

# 定义分词函数
def tokenizer(text):
    # return list(jieba.cut(text,use_paddle=False))
    return list(jieba.cut(text,use_paddle=True))
    # return list(jieba.cut(text))

def load_data(file,div=False,predict=False):
    with open(file,"r",encoding=encoder) as f:
        cont = list(map(lambda a:a.strip("\n"),f.readlines()))
    text = list(map(lambda a:de_pun(a.split(sep)[0]),cont))
    if predict:
        label = []
    else:
        label = list(map(lambda a:divlize_label(dec2lis(a.split(sep)[1]),div=div),cont))
    return text,label



def doc2mat(corpus): # corpus: list of string

    # 使用CountVectorizer将语料库转换为文档-单词矩阵
    if not doing_predict:
        vectorizer = CountVectorizer(tokenizer=tokenizer)
        X = vectorizer.fit_transform(corpus)
        joblib.dump(vectorizer, model_path+vect_file)
    else:
        vectorizer = joblib.load(model_path+vect_file)
        X = vectorizer.transform(corpus)

    word_lis0 = vectorizer.get_feature_names_out()
    with open(auf_path+'stopwords.txt',encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())
    word_lis_delidx = [1 if word_lis0[i] in stopwords else 0 for i in range(len(word_lis0))]
    # 将元素为1的列从二维数组中删除
    cols_to_delete = np.where(np.array(word_lis_delidx) == True)
    new_X = np.delete(X.toarray(), cols_to_delete, axis=1)
    word_lis1 = np.delete(np.array(word_lis0), cols_to_delete)
    return new_X,word_lis1

def to_one_hot(lst,pop_none=False):
    unique_labels = sorted(list(set([item for sublist in lst for item in sublist])))
    num_labels = len(unique_labels)
    one_hot_arr = np.zeros((len(lst), num_labels))
    for i, sublst in enumerate(lst):
        sublst_one_hot = np.zeros(num_labels)
        for label in sublst:
            label_idx = unique_labels.index(label)
            sublst_one_hot[label_idx] = 1
        one_hot_arr[i] = sublst_one_hot
    if pop_none:
        return one_hot_arr[:, :-1]
    return one_hot_arr

def lda_fit(X):
    # 创建pLSA模型
    model = LatentDirichletAllocation(n_components=num_topics, max_iter=max_iterations)
    # 训练模型
    train_X = X[:,:]# 94x4991文档-单词矩阵/94*4544
    # test_X = X.toarray()[-1,:].reshape(1,4991)
    model.fit(train_X)

    # 输出主题词汇
    print("----------------输出主题分类结果----------------")
    for i, topic in enumerate(model.components_):
        top_words_indices = topic.argsort()[:-11:-1]
        # top_words = ' '.join([str(index) for index in top_words_indices])
        top_words = ' '.join([word_lis[index] for index in top_words_indices])
        print(f"Topic {i}: {top_words}")

    joblib.dump(model, model_path+joblib_file)

    return model.transform(train_X)

def load_lda(X):
    # print(model_path+joblib_file)
    model = joblib.load(model_path+joblib_file)
    res = model.transform(X)
    return res

class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

def FNN_classification(X, y, num_epochs=100000, batch_size=96, learning_rate=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_dim = X.shape[1]
    hidden_dim = 40
    output_dim = y.shape[1]

    model = FNN(input_dim, hidden_dim, output_dim)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print("---------Training FNN!-------------")
    for epoch in range(num_epochs):
        for i in range(0, X_train.shape[0], batch_size):
            batch_x = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y.float())
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    with torch.no_grad():
        y_pred_train = model(X_train)
        y_pred_test = model(X_test)

    train_acc = ((y_pred_train > 0.5) == y_train).float().mean()
    test_acc = ((y_pred_test > 0.5) == y_test).float().mean()
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    joblib.dump(model, model_path+FNN_file)

    return model

def write_label_txt(filename,lis):
    with open(filename,"w") as f:
        for i in lis:
            if len(i)==1 and int(i[0])==12:
                f.write(str([])+"\n")
            else:
                f.write(str(i)+"\n")
    return 

    

def run_plsa_ana2():
    # 定义语料库
    # print("Processing PDF imported")
    if not doing_predict:
        corpus,labels = load_data(file_path+train_name,div=div_flag,predict=doing_predict)
        y_onehot = to_one_hot(labels,pop_none=pop_none_flag)
        y_tensor = torch.from_numpy(y_onehot).type(torch.float32)
    else:
        corpus,labels = load_data(file_path+predict_name,div=div_flag,predict=doing_predict)
    # print(corpus)
    X,word_lis = doc2mat(corpus)
    # print(X)
    # print(word_lis)
    if load_lda_flag:
        # print("loading lda!")
        print("Extracting information from pdf!")
        lda_X = load_lda(X)
    else:
        print("train lda")
        lda_X = lda_fit(X)


    lda_X_tensor = torch.from_numpy(lda_X).type(torch.float32)
    
    if load_fnn_flag:
        # print("load FNN")
        print("Using Neural Network to analyze pdf content!")
        FNN_model = joblib.load(model_path+FNN_file)
    else:
        print("train FNN")
        FNN_model = FNN_classification(lda_X_tensor,y_tensor)

    print("generating results!")
    # 将y_pred_test > 0.5转换为numpy数组
    # print(lda_X_tensor)
    FNN_res = FNN_model(lda_X_tensor)
    # print(FNN_res)
    y_pred_test_np = ( FNN_res > 0.5).numpy()
    FNN_res_np = FNN_res.detach().numpy()

    # 遍历每个样本
    predict_labels_list = []
    predict_confidence_list = []
    for i in range(y_pred_test_np.shape[0]):
        # 获取该样本预测为1的标签编号
        prelabels = np.where(1-y_pred_test_np[i] < 1e-5)[0]
        confidence_np = FNN_res_np[i]
        predict_labels_list.append(prelabels.tolist())
        predict_confidence_list.append([confidence_np[j] for j in prelabels.tolist()])

    # write_label_txt("./label.txt",labels)
    # write_label_txt("./predicted_label.txt",predict_labels_list)
    # write_label_txt("./predicted_confidence.txt",predict_confidence_list)
    
    result_dict = {i:[[],[]] for i in range(12)}
    for i in range(len(predict_labels_list)):
        page = i+1
        for j in range(len(predict_labels_list[i])):
            result_dict[predict_labels_list[i][j]][0].append(page)
            result_dict[predict_labels_list[i][j]][1].append(round(1-np.sin(predict_confidence_list[i][j]*np.pi),1))
    
    print("process complete!")
    # print(result_dict)
    # print(predict_labels_list)
    with open("result_dict.json", "w") as outfile:
        json.dump(result_dict, outfile)

    print("result in result_dict!") 
    
# 字典中的内容即为程序结果，存储在result_dict.json中
# result dict:
# {0: [[], []], 
# 1: [[6], [1.0]], 
# 2: [[28, 29, 30, 35, 36, 38, 41, 89], [1.0, 0.7, 0.3, 1.0, 0.9, 0.9, 1.0, 0.0]],
# 3: [[28, 34, 36, 38], [1.0, 1.0, 1.0, 0.9]], 
# 4: [[15, 17, 18, 19, 20, 24, 44, 49], [1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 1.0, 1.0]], 
# 5: [[15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 46, 63, 65, 88], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.3, 0.5, 0.7]], 
# 6: [[52, 79], [0.7, 1.0]], 
# 7: [[], []], 
# 8: [[], []], 
# 9: [[], []], 
# 10: [[], []], 
# 11: [[6], [1.0]]}
# 字典中每一个pair代表： 内容标签:[[有关此内容的页码],[这些页码属于此内容标签的confidence(0~1.0)]]

# 0：内容标签释义
# 1：是否存在利益相关方对企业的评价
# 2：董事会是否发挥监控作用
# 3：是否对员工健康及安全的保护措施
# 4：是否有完善的人才培养措施
# 5：生态保护措施
# 6：节能减排措施
# 7：公益项目参与
# 8：乡村振兴项目参与
# 9：群体列表
# 10：与利益相关方沟通情况
# 11：管治基本架构




