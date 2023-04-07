import sys

sys.path.append('../')
from ..model import BertForSentenceClassification
from ..model import BertConfig
from ..utils import LoadSingleSentenceClassificationDataset
from ..utils import logger_init
from transformers import BertTokenizer
import logging
import torch
import os
import time

do_predict_flag=True

class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'SingleSentenceClassification')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'train.txt')
        self.val_file_path = os.path.join(self.dataset_dir, 'val.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'test.txt')
        self.predict_file_path = os.path.join(self.dataset_dir, 'predict.txt')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.split_sep = '_!_'
        self.is_sample_shuffle = True
        self.batch_size = 32
        self.max_sen_len = None
        self.num_labels = 8191 #8191
        self.epochs = 10
        self.model_val_per_epoch = 2
        self.max_position_embeddings=1024
        logger_init(log_file_name='single', log_level=logging.INFO,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")


def train(config):
    model = BertForSentenceClassification(config,
                                          config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path,map_location=config.device)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5) #5e-5 def
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                            tokenizer=bert_tokenize,
                                                            batch_size=config.batch_size,
                                                            max_sen_len=config.max_sen_len,
                                                            split_sep=config.split_sep,
                                                            max_position_embeddings=config.max_position_embeddings,
                                                            pad_index=config.pad_token_id,
                                                            is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)
    max_acc = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_iter):
            sample = sample.to(config.device)  # [src_len, batch_size]
            label = label.to(config.device)
            padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
            loss, logits = model(
                input_ids=sample,
                attention_mask=padding_mask,
                token_type_ids=None,
                position_ids=None,
                labels=label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            acc = (logits.argmax(1) == label).float().mean()
            if idx % 10 == 0:
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            acc = evaluate(val_iter, model, config.device, data_loader.PAD_IDX)
            logging.info(f"Accuracy on val {acc:.3f}")
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), model_save_path)


def inference(config,eval_flag=True):
    model = BertForSentenceClassification(config,
                                          config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path,map_location=config.device)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行预测......")
    model = model.to(config.device)
    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                          tokenizer=BertTokenizer.from_pretrained(
                                                              config.pretrained_model_dir).tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          split_sep=config.split_sep,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)
    if eval_flag:
        acc = evaluate(test_iter, model, device=config.device, PAD_IDX=data_loader.PAD_IDX)
        logging.info(f"Acc on test:{acc:.3f}")

def predict(config):
    model = BertForSentenceClassification(config,
                                          config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path,map_location=config.device)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行预测......")
    model = model.to(config.device)
    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                            tokenizer=BertTokenizer.from_pretrained(
                                                                config.pretrained_model_dir).tokenize,
                                                            batch_size=1,
                                                            max_sen_len=config.max_sen_len,
                                                            split_sep=config.split_sep,
                                                            max_position_embeddings=config.max_position_embeddings,
                                                            pad_index=config.pad_token_id,
                                                            is_sample_shuffle=False)
    test_iter = data_loader.load_train_val_test_data(test_file_path=config.predict_file_path,
                                                                           only_test=True)

    res_lis = eval_pred(test_iter, model, device=config.device, PAD_IDX=data_loader.PAD_IDX)
    return res_lis
    

def evaluate(data_iter, model, device, PAD_IDX):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            padding_mask = (x == PAD_IDX).transpose(0, 1)
            logits = model(x, attention_mask=padding_mask)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
            

        model.train()
        return acc_sum / n


def eval_pred(data_iter, model, device, PAD_IDX):
    model.eval()
    res_lis = []
    with torch.no_grad():
        # print("len of data iter: ",len(list(data_iter)))
        for x, i in data_iter:
            print(i)
            x = x.to(device)
            padding_mask = (x == PAD_IDX).transpose(0, 1)
            logits = model(x, attention_mask=padding_mask)
            y=logits.argmax(1)
            res_lis.append(y)

        model.train()
    return list(map(lambda a:a.item(),res_lis))

def dec2lis(num):
    num = int(num)
    bin_num = bin(num)[2:]
    return [i for i in range(len(bin_num)) if bin_num[::-1][i]=="1"]

def run_bert_classification():
    model_config = ModelConfig()
    if do_predict_flag:
        
        # 遍历目标路径下的所有文件和子目录
        for root, dirs, files in os.walk(model_config.dataset_dir):
            for file in files:
                # 如果文件以 .pt 扩展名结尾，则删除该文件
                if file.endswith(".pt"):
                    os.remove(os.path.join(root, file))
                    # print(f"{file} has been removed.")
        # print(model_config.device)
        res_lis = predict(model_config)
        res_lis = list(map(dec2lis,res_lis))
        record_ = set()
        for i in range(len(res_lis)):
            for j in range(len(res_lis[i])):
                record_.add(res_lis[i][j])
        result_ = [True if i in record_ else False for i in range(12)]
        return result_
    else:
        # train(model_config)
        inference(model_config)
        return
