import re
import glob
import PyPDF2
# from PyPDF2 import PdfReadWarning
# import os

# import warnings
# warnings.filterwarnings('ignore', category=PdfReadWarning)



directory = "./strategy/cache/"
pdf_files = glob.glob(directory + "*.pdf")
# for i in pdf_files:
#     print(i)
csv_files = glob.glob(directory + "*.csv")
# note = "./note_Res.txt"
# file=open(note,"w")
# file.close()
if_depun = False # default for "False"

def get_base(path):
    basename_regex = re.compile(r'[^/\\\\]+(?=[.][a-zA-Z]+$)')
    match = basename_regex.search(path)
    if match:
        return match.group()
    else:
        return ""


def de_pun(text):
    # Define the regular expression pattern to match all digits and punctuation marks and space and \n
    #pattern = r'[\d\s+' + re.escape(string.punctuation + '，。、！？：；‘’“”【】《》——·') + r']'
    # 只保留中文
    pattern = r'[^\u4e00-\u9fff]'
    # Use the pattern to remove all matches from the text
    res_text = re.sub(pattern, '', text)

    return res_text

def extra_pdf(obj_dir,basename,if_depun):
    # 打开PDF文件
    pdf_file = open(obj_dir+basename+".pdf", 'rb')
    # basename = filename.split(".")[0]
    # 创建PDF读取器对象
    pdf_reader = PyPDF2.PdfReader(pdf_file,strict=False)
    # 提取每一页的文本并写入txt文件, page by page
    num_of_pages = len(pdf_reader.pages)
    res_list = []
    for page_num in range(num_of_pages):
        page = pdf_reader.pages[page_num]
        # with open(pagedir+"/"+str(page_num)+".txt", "w") as text_file:
        cont = page.extract_text()
            # res_text = "".join(cont)
            # text_file.write(res_text)
        if if_depun:
            temp = "".join(list(map(de_pun,cont))) #method 1, using de_pun
        else:
            temp = "".join(list(map(lambda a:a.strip("\n"),cont))) #method 2, simply cat
        res_list.append(temp)

    # 关闭文件
    pdf_file.close()
    # text_file.close()
    return res_list

def label_process(string):
    string = string.strip("\n")
    temp = list(string.split("."))
    i=0
    while i<len(temp):
        if temp[i]=="N":
            temp[i] = "12"
        if len(temp[i])==0:
            temp.pop(i)
        else:
            i+=1

    if len(temp)==0:
        return 
    else:
        return temp

def extra_csv(obj_dir,basename):
    with open(obj_dir+basename+".csv","r") as f:
        cont = f.readlines()
        cont = [label_process(cont[i]) for i in range(len(cont))]
    i=0
    while i<len(cont):
        if cont[i]==None:
            cont.pop(i)
        else:
            i+=1
    return cont

def string_code(s,encoder="utf-8"):
    s = " ".join(s.split())
    s = s.encode(encoder, errors='ignore') # 这会将 s 转换为 bytes 类型，并忽略无法编码的字符
    s = s.decode(encoder, errors='ignore') # 这会将 s 转换回字符串类型，并忽略无法解码的字节
    return s



def run_extra():
    obj_dir = directory
    pdf_files = glob.glob(directory + "*.pdf")
    for k in range(len(pdf_files)):
        basename = get_base(pdf_files[k])
        # print("run_EXTRA base:"+basename)
        cont_lis = extra_pdf(obj_dir,basename,if_depun)
        # label_lis = extra_csv(obj_dir,basename)
        # try:
        with open("predict"+".txt","w",encoding="utf-8") as f:
            for i in range(len(cont_lis)):
                f.write(string_code(cont_lis[i]+"_!_"+"-1")+"\n")
                # if i < len(label_lis):
                #     for j in range(len(label_lis[i])):
                #         try:
                #             f.write(string_code(cont_lis[i]+"_!_"+label_lis[i][j])+"\n")
                #         except Exception as e:
                #             print("ERROR!","TYPE:"+str(e))
                #             print("filename:"+basename,"[i,j]=[{},{}]".format(i,j),sep="\t")
                #             continue
        # except:
        #     print(cont_lis)
        #     print("---------------------")
        #     print(label_lis)

