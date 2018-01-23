
# coding: utf-8

# In[21]:

import pickle
import re
import jieba


# In[5]:

model = pickle.load(open('model','rb'))
model


# In[22]:

stopwordset = set()
with open('jieba_dict/stopwords.txt','r',encoding = 'utf-8') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))
        
def seg_sent(art):
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'  
    art = art.strip()
    art = re.sub(r, '', art)
    l = []
    seg_list = list(jieba.cut(art, cut_all = False))
    for word in seg_list:
        if word not in stopwordset and word != ' ' and word != "\n" and word != "\n\n":
            l.append(word)
    corpus = " ".join(l)
    return corpus


# In[39]:


def main(title):
    file = open(title,"r",encoding = 'utf-8')
    
    article = file.read()
    
    cp = seg_sent(article)
    
    result = model.predict([cp])
    output = open('file_sa.txt','w')
    if result == 0:
        print ("这是一篇负面新闻")
        output.write("这是一篇负面新闻")
    else:
        print("这是一篇正面新闻")
        output.write("这是一篇正面新闻")
    print('负面值:\t\t','正面值:')
    rate = model.predict_proba([cp])
    print(rate)
    output.write("负面值："+str(rate[0][0])+" 正面值： "+str(rate[0][1]))


# In[ ]:

if __name__ == "__main__":
    import sys
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    print(file_path)
    main(file_path)


# In[ ]:



