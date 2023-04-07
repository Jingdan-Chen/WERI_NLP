import shutil
import os

def empty_dir(dir_name):
    """
    清空目标文件夹下的所有文件和文件夹。
    
    参数：
    dir_name：字符串，目标文件夹的名称。
    """
    for root, dirs, files in os.walk(dir_name, topdown=False):
        # 删除文件
        for file in files:
            os.remove(os.path.join(root, file))
        # 删除文件夹
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))

def find_Strategy(pdf_name,bert_flag=1): # bert_flag: 1 for using BERT; 0 for using plsa
    default_EP= -1
    default_EC= -1
    default_PW= -1
    default_CE= -1
    default_Sf= -1
    default_Tr= -1
    curr_d = "./"  #D:\Source\citi\Citicup-Web\py\NLP\strategy
    # empty_dir(curr_d+"strategy/cache")
    # print(pdf_name)
    # f=open(pdf_name,"r")
    # f.close()
    print(pdf_name)
    shutil.copy(pdf_name, curr_d+"/strategy/cache")
    # os.system("python "+curr_d+"new_extra.py")
    from .new_extra import run_extra
    run_extra()
    # try:
    if bert_flag:
        # 使用bert预测
        shutil.copy(curr_d + "predict.txt", curr_d + "strategy/BERT/data/SingleSentenceClassification")
        current_dir = os.getcwd()
        os.chdir(curr_d + "strategy/BERT/Tasks")
        from .BERT.Tasks.TaskForSingleSentenceClassification import run_bert_classification
        res_lis = run_bert_classification()
        os.chdir(current_dir)
    else:  # 使用plsa
        shutil.copy(curr_d + "predict.txt", curr_d + "strategy/plsa")
        from .plsa.plsa_main import plsa_analyze
        res_lis = plsa_analyze()

        assert len(res_lis) == 12
        # os.rename(curr_d + "result_dict.json", curr_d + "strategy/plsa/cache/result_dict.json")
        # os.rename(curr_d + "predict.txt", curr_d + "strategy/plsacache/predict.txt")
    default_EP = int(res_lis[4])  # True for 1, False for 0
    default_EC = int(res_lis[5])
    default_PW = int(res_lis[6])
    default_CE = int(res_lis[8])
    default_Sf = int(res_lis[2])
    default_Tr = int(res_lis[3])
    # except Exception as e:
    #     print(f'Failed to analyze. Reason: {e}')



    return default_EP, default_EC, default_PW, default_CE, default_Sf, default_Tr
