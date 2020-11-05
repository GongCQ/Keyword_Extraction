#!/usr/bin/env python
# coding: utf-8

# # “神策杯”2018高校算法大师赛 有监督特征生成
# 
# 特征 | 解释
# ---|---
# id | 候选关键词所属样本id
# tags | 候选关键词
# cixing | 词性
# tfidf |jieba库算出的tfidf值
# ske | 共现矩阵偏度，From 论文《大数据时代基于统计特征的情报关键词提取方法》
# occur_in_title | 是否出现在标题
# occur_in_first_sentence | 是否出现在正文第一句
# occur_in_last_sentence | 是否出现在正文最后一句
# occur_in_other_sentence | 是否出现在正文中间
# len_tags | 词长
# num_tags | 当前样本候选关键词个数
# num_sen | 当前样本句子数量
# classes | 聚类特征，doc2vec+Kmeans
# len_text | 标题+正文长度
# textrank | jieba库计算的textrank值
# word_count | 词频
# tf | 词频 / 总单词数
# num_head_words | 头词频，候选关键词在前1/4文本里面出现频次
# hf | 头词频 / 前1/4文本单词数
# pr | tf / tf.sum()
# has_num | 候选关键词是否包含数字
# has_eng | 候选关键词是否包含英文
# is_TV | 是否是《..》里面的词
# sim | 文本doc2vec与候选关键词word2vec的余弦相似度
# sim_euc | 文本doc2vec与候选关键词word2vec的欧氏距离
# mean_l2 | 关键词所在句子平均长度
# meaxl2 | 关键词所在句子最大长度
# min_l2 | 关键词所在句子最小长度
# min_pos | 关键词最早出现词位置
# diff_min_pos_bili | 词位置相关统计特诊
# diff_kurt_pos_bili | 词位置相关统计特诊
# diff_max_min_sen_pos | 关键词所在句子位置相关统计特征
# diff_var_sen_pos_bili | 关键词所在句子位置相关统计特征
# mean_sim_tags | 单词平均相似度，基于word2vec
# diff_mean_sim_tags | 单词平均相似度相关统计特征
# kurt_sim_tags_256 | 单词相似度峰度，基于另一word2vec模型
# diff_max_min_sim_tags_256 | 单词平均相似度相关统计特征
# var_gongxian | 共现矩阵方差
# kurt_gongxian | 共现矩阵峰度
# diff_min_gongxian | 共现矩阵相关统计特征
# cixing_\*_num | 当前样本候选关键词词性\*的数量
# cixing_\*_bili | 当前样本候选关键词词性\*的比例
# label | 标签
# 
# - 参考论文
# 
# [1] 常耀成, 张宇翔, 王红, 等. 特征驱动的关键词提取算法综述[J]. 软件学报, 2018, 7: 015.
# 
# [2] 李跃鹏, 金翠, 及俊川. 基于 word2vec 的关键词提取算法[J]. 科研信息化技术与应用, 2015, 6(4): 54-59.
# 
# [3] 罗繁明, 杨海深. 大数据时代基于统计特征的情报关键词提取方法[J]. 情报资料工作, 2013, 34(3): 19r20.

# In[1]:


import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from sklearn.preprocessing import MinMaxScaler
import re
import pickle
from operator import itemgetter
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import gc
import math
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import skew, kurtosis
from collections import Counter
from tqdm import tqdm

get_ipython().magic('matplotlib inline')

pd.options.display.max_rows = 700


# ## 加载自定义词典

# In[2]:


## 搜狗+百度词典 深蓝词典转换
jieba.load_userdict('./字典/明星.txt')
jieba.load_userdict('./字典/实体名词.txt')
jieba.load_userdict('./字典/歌手.txt')
jieba.load_userdict('./字典/动漫.txt')
jieba.load_userdict('./字典/电影.txt')
jieba.load_userdict('./字典/电视剧.txt')
jieba.load_userdict('./字典/流行歌.txt')
jieba.load_userdict('./字典/创造101.txt')
jieba.load_userdict('./字典/百度明星.txt')
jieba.load_userdict('./字典/美食.txt')
jieba.load_userdict('./字典/FIFA.txt')
jieba.load_userdict('./字典/NBA.txt')
jieba.load_userdict('./字典/网络流行新词.txt')
jieba.load_userdict('./字典/显卡.txt')

## 爬取漫漫看网站和百度热点上面的词条
jieba.load_userdict('./字典/漫漫看_明星.txt')
jieba.load_userdict('./字典/百度热点人物+手机+软件.txt')
jieba.load_userdict('./字典/自定义词典.txt')

## 实体名词抽取之后的结果 有一定的人工过滤 
## origin_zimu 这个只是把英文的组织名过滤出来
jieba.load_userdict('./字典/person.txt')
jieba.load_userdict('./字典/origin_zimu.txt')

## 第一个是所有《》里面出现的实体名词
## 后者是本地测试集的关键词加上了 
jieba.load_userdict('./字典/出现的作品名字.txt')
jieba.load_userdict('./字典/val_keywords.txt')

## 网上随便找的停用词合集
jieba.analyse.set_stop_words('./stopword.txt')


# ## 加载数据

# In[3]:


all_docs = pd.read_csv('all_docs.txt', sep='\001', header=None)
all_docs.columns = ['id', 'title', 'content']
all_docs.fillna('', inplace=True)

val = pd.read_csv('train_docs_keywords.txt', sep='\t', header=None)
val.columns = ['id', 'kw']
val.kw = val.kw.apply(lambda x: x.split(','))


# ## 数据清理
# 在比赛的过程中，通过分析文本中还是存在挺多噪声字符的，例如```&amp;```等，在这里就直接将这些字符清洗了。

# In[4]:


all_docs['title_cut'] = all_docs['title'].apply(lambda x:''.join(filter(lambda ch: ch not in ' \t◆#%', x)))
all_docs['content_cut'] = all_docs['content'].apply(lambda x:''.join(filter(lambda ch: ch not in ' \t◆#%', x)))

all_docs['title_cut'] = all_docs['title_cut'].apply(lambda x: re.sub('&amp;', ' ', x))
all_docs['title_cut'] = all_docs['title_cut'].apply(lambda x: re.sub('&quot;', ' ', x))
all_docs['title_cut'] = all_docs['title_cut'].apply(lambda x: re.sub('&#34;', ' ', x))

all_docs['content_cut'] = all_docs['content_cut'].apply(lambda x: re.sub('&amp;', ' ', x))
all_docs['content_cut'] = all_docs['content_cut'].apply(lambda x: re.sub('&quot;', ' ', x))
all_docs['content_cut'] = all_docs['content_cut'].apply(lambda x: re.sub('&#34;', ' ', x))

all_docs['content_cut'] = all_docs['content_cut'].apply(lambda x: re.sub('&nbsp;', ' ', x))
all_docs['title_cut'] = all_docs['title_cut'].apply(lambda x: re.sub('&nbsp;', ' ', x))

all_docs['content_cut'] = all_docs['content_cut'].apply(lambda x: re.sub('&gt;', ' ', x))
all_docs['title_cut'] = all_docs['title_cut'].apply(lambda x: re.sub('&gt;', ' ', x))

all_docs['content_cut'] = all_docs['content_cut'].apply(lambda x: re.sub('&lt;', ' ', x))
all_docs['title_cut'] = all_docs['title_cut'].apply(lambda x: re.sub('&lt;', ' ', x))

all_docs['content_cut'] = all_docs['content_cut'].apply(lambda x: re.sub('hr/', ' ', x))
all_docs['title_cut'] = all_docs['title_cut'].apply(lambda x: re.sub('hr/', ' ', x))

strinfo = re.compile('······')

all_docs['content_cut'] = all_docs['content_cut'].apply(lambda x: re.sub(strinfo, ' ', x))
all_docs['title_cut'] = all_docs['title_cut'].apply(lambda x: re.sub(strinfo, ' ', x))


# ## 预处理之句子拆分
# - 目的是考虑词的位置，加大关键词在标题/首局/末句的词频（权重）

# In[5]:


def split_sentences(x):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    ret = pd.Series()

    ret['id'] = x['id']
    if x['content_cut'] == '' or len(x['content_cut']) < 2:
        ret['first_sentence'] = ''
        ret['other_sentence'] = ''
        ret['last_sentence'] = ''
        return ret

    sentence_delimiters = re.compile(u'[。？！；!?]')
    sentences =[i for i in sentence_delimiters.split(x['content_cut']) if i != '']
    num_sen = len(sentences)

    if num_sen == 1:
        ret['first_sentence'] = sentences[0]
        ret['other_sentence'] = ''
        ret['last_sentence'] = sentences[0]
    elif num_sen == 2:
        ret['first_sentence'] = sentences[0]
        ret['other_sentence'] = ''
        ret['last_sentence'] = sentences[-1]
    else:
        ret['first_sentence'] = sentences[0]
        ret['other_sentence'] = ''.join(sentences[1:-1])
        ret['last_sentence'] = sentences[-1]
    return ret


# In[6]:


tmp = all_docs.apply(split_sentences, axis=1)

all_docs = pd.merge(all_docs, tmp, on='id', how='left')


# In[7]:


# 取出第一句的作品名字 《。。。》
# 感谢豆腐的baseline，我们可以知道假如标题中存在《。。。》，那么里面的内容直接就是关键词

all_docs['first_sentence_reg'] = all_docs['first_sentence'].apply(lambda x:re.findall(r"《(.+?)》",x))


# In[8]:


# 提取《》里面的实体名词加进jieba词库并持久化

# reg = []
# for word in all_docs['content_cut'].apply(lambda x:re.findall(r"《(.+?)》",x)).values:
#     reg += word

# with open('出现的作品名字.txt', 'w', encoding='utf-8') as f:
#     for word in set(reg):
#         f.write(word+'\n')


# In[9]:


## 历史作品名字 作为下面的一个特征
TV = []
with open('./字典/出现的作品名字.txt', 'r', encoding='utf-8') as f:
    for word in f.readlines():
        TV.append(word.strip())


# In[10]:


## 这是根据本数据集算的idf文件
idf = {}
with open('my_idf.txt', 'r', encoding='utf-8') as f:
    for i in f.readlines():
        if len(i.strip().split()) == 2:
            v = i.strip().split()
            idf[v[0]] = float(v[1])


# ## 线下评估函数

# In[ ]:


def evaluate(df):
    def get_score(x):
        score = 0
        if x['label1'] in x['kw']:
            score += 0.5
        if x['label2'] in x['kw']:
            score += 0.5 
        return score
    
    pred = df[df.id.isin(val.id)]
    tmp  = pd.merge(pred, val, on='id', how='left')
    tmp['score'] = tmp.apply(get_score, axis=1)
    print('Score: ',tmp.score.sum())
    return tmp


# ## Doc2Vec 聚类

# In[ ]:


from gensim.models import Doc2Vec
from gensim.models import Word2Vec


# In[ ]:


## classes_doc2vec.npy 文件是先算DOC2VEC向量 然后用Kmeans简单聚成10类
classes = np.load('classes_doc2vec.npy')

all_docs['classes'] = classes


# ## Doc2vec word2vec 计算主题相似性
# 将Doc2vec word2vec两个模型的向量长度定为一样长就可以直接计算余弦相似度、欧式距离等

# In[ ]:


doc2vec_model = Doc2Vec.load('doc2vec.model')

word2vec_model = Word2Vec.load('word2vec.model')


# In[ ]:


wv= word2vec_model.wv


# In[ ]:


def Cosine(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return npvec1.dot(npvec2)/(math.sqrt((npvec1**2).sum()) * math.sqrt((npvec2**2).sum()))


def Euclidean(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1-npvec2)**2).sum())


# In[ ]:


## 后面加大窗口和迭代又算了一次word2vec 模型 主要是用来算候选关键词之间的相似度

word2vec_model_256 = Word2Vec.load('word2vec_iter10_sh_1_hs_1_win_10.model')


# In[ ]:


all_docs['idx'] = all_docs.index.values


# ## 特征提取

# In[ ]:


def get_train_df(df, train=True):
    res = []
    for index in tqdm(df.index):
        
        x = df.loc[index]
        # TF-IDF
        first_sentence_reg = ' '.join(x['first_sentence_reg'])
        ## 这里主要是提取jieba默认的tf-idf值 我这边jieba自带的idf文件效果比自己提取的要好 所以也顺便用来筛选候选词 
        ## 假如把topK设置为None的话 数据量会增大10倍 但是非常难跑 没有验证过效果
        ## PS：这里我稍微修改了一下jieba的源码 allowpPOS实际是不允许出现的词性 即allowPOS = NotAllowPOS
        
        text = 19*(x['title_cut']+'。')+ 3*(x['first_sentence']+'。') + 1*(x['other_sentence']+'。')+                3*(x['last_sentence']+ '。') + 7*first_sentence_reg
        jieba_tags = jieba.analyse.extract_tags(sentence=text, topK=20, allowPOS=('r','m','d', 'p', 'q', 'ad', 'u', 'f'), withWeight=True,                                          withFlag=True)

        tags = []
        cixing = []
        weight = []
        for tag in jieba_tags:
            tags.append(tag[0].word)
            cixing.append(tag[0].flag)
            weight.append(tag[1])

        sentence_delimiters = re.compile(u'[。？！；!?]')
        sentences =[i for i in sentence_delimiters.split(text) if i != '']
        num_sen = len(sentences)

        words = []
        num_words = 0
        for sen in sentences:
            cut = jieba.lcut(sen)
            words.append(cut)
            num_words += len(cut)
        
        new_tags = []
        new_cixing = []
        new_weight = []
        len_tags = []
        for i in range(len(tags)):
            if tags[i].isdigit() and tags[i] not in ['985', '211']:
                continue
            if ',' in tags[i]:
                continue
            new_tags.append(tags[i])
            new_weight.append(weight[i])
            new_cixing.append(cixing[i])
            len_tags.append(len(tags[i]))
            
            
        ## 位置特征： 1. 是否出现在标题 2.是否出现在第一句 3.是否出现在最后一句 4.出现在正文中间部分
        occur_in_title = np.zeros(len(new_tags))
        occur_in_first_sentence = np.zeros(len(new_tags))
        occur_in_last_sentence = np.zeros(len(new_tags))
        occur_in_other_sentence = np.zeros(len(new_tags))
        for i in range(len(new_tags)):
            if new_tags[i] in x['title_cut']:
                occur_in_title[i] = 1
            if new_tags[i] in x['first_sentence']:
                occur_in_first_sentence[i] = 1
            if new_tags[i] in x['last_sentence']:
                occur_in_last_sentence[i] = 1
            if new_tags[i] in x['other_sentence']:
                occur_in_other_sentence[i] = 1
        
        
        ## 共现矩阵及相关统计特征 这里我一开始统计了好多 例如均值、方差、偏度等 得到新特征后贪心验证只保留以下三个 下面的统计特征同理
        num_tags = len(new_tags)
        arr = np.zeros((num_tags, num_tags))
        for i in range(num_tags):
            for j in range(i+1, num_tags):
                count = 0
                for word in words:
                    if new_tags[i] in word and new_tags[j] in word:
                        count += 1
                arr[i, j] = count
                arr[j, i] = count
        ske = stats.skew(arr)
        # cols += ['var_gongxian']
        # cols += ['kurt_gongxian']
        # cols += ['diff_min_gongxian']   
        var_gongxian = np.zeros(len(new_tags))
        kurt_gongxian = np.zeros(len(new_tags))
        diff_min_gongxian = np.zeros(len(new_tags))
        for i in range(len(new_tags)):
            var_gongxian[i] = np.var(arr[i])
            kurt_gongxian[i] = stats.kurtosis(arr[i])
            diff_sim = np.diff(arr[i])
            if len(diff_sim) > 0:
                diff_min_gongxian[i] = np.min(diff_sim)

                
        ## textrank特征
        textrank_tags = dict(jieba.analyse.textrank(sentence=text, allowPOS=('r','m','d', 'p', 'q', 'ad', 'u', 'f'), withWeight=True))
        
        textrank = []
        for tag in new_tags:
            if tag in textrank_tags:
                textrank.append(textrank_tags[tag])
            else:
                textrank.append(0)
                
        all_words = np.concatenate(words).tolist()
        
        ## 词频
        tf = []
        for tag in new_tags:
            tf.append(all_words.count(tag))
        tf = np.array(tf)
        
        ## hf: 头词频，文本内容前1/4候选词词频
        hf = []
        head = len(words) // 4 + 1
        head_words = np.concatenate(words[:head]).tolist()
        for tag in new_tags:
            hf.append(head_words.count(tag))
        
        ## has_num：是否包含数字
        ## has_eng: 是否包含字母
        def hasNumbers(inputString):
            return bool(re.search(r'\d', inputString))
        def hasEnglish(inputString):
            return bool(re.search(r'[a-zA-Z]', inputString))
        has_num = []
        has_eng = []
        for tag in new_tags:
            if hasNumbers(tag):
                has_num.append(1)
            else:
                has_num.append(0)
            if hasEnglish(tag):
                has_eng.append(1)
            else:
                has_eng.append(0)
                
        ## is_TV:是否为作品名称
        is_TV = []
        for tag in new_tags:
            if tag in TV:
                is_TV.append(1)
            else:
                is_TV.append(0)
                
        ## idf: 用训练集跑出的逆词频
        v_idf = []
        for tag in new_tags:
            v_idf.append(idf.get(tag, 0))
        
        ## 计算文本相似度，这里直接用doc2vec跟每个单词的word2vec做比较
        ## sim: 余弦相似度
        ## sim_euc：欧氏距离
        default = np.zeros(100)
        doc_vec = doc2vec_model.docvecs.vectors_docs[x['idx']]
        sim = []
        sim_euc = []
        for tag in new_tags:
            if tag in wv:
                sim.append(Cosine(wv[tag], doc_vec))
                sim_euc.append(Euclidean(wv[tag], doc_vec))
            else:
                sim.append(Cosine(default, doc_vec))
                sim_euc.append(Euclidean(default, doc_vec))
                
        ## 关键词所在句子长度 L2，记录为列表，然后算统计特征 
        mean_l2 = np.zeros(len(new_tags))
        max_l2 = np.zeros(len(new_tags))
        min_l2 = np.zeros(len(new_tags))
        for i in range(len(new_tags)):
            tmp = []
            for word in words:
                if new_tags[i] in word:
                    tmp.append(len(word))
            if len(tmp) > 0:
                mean_l2[i] = np.mean(tmp)
                max_l2[i] = np.max(tmp)
                min_l2[i] = np.min(tmp)
                
# cols += ['min_pos']
# cols += ['diff_min_pos_bili']
# cols += ['diff_kurt_pos_bili']  

        ## 关键词所在位置，记录为列表，然后算统计特征 

        min_pos = [np.NaN for _ in range(len(new_tags))]
        diff_min_pos_bili = [np.NaN for _ in range(len(new_tags))]
        diff_kurt_pos_bili = [np.NaN for _ in range(len(new_tags))]
        
        for i in range(len(new_tags)):
            pos = [a for a in range(len(all_words)) if all_words[a] == new_tags[i]]
            pos_bili = np.array(pos) / len(all_words)
            
            if len(pos) > 0:
                min_pos[i] = np.min(pos)
                diff_pos = np.diff(pos)
                diff_pos_bili = np.diff(pos_bili)
                if len(diff_pos) > 0:
                    diff_min_pos_bili[i] = np.min(diff_pos_bili)
                    diff_kurt_pos_bili[i] = stats.kurtosis(diff_pos_bili)
                    
        ## 关键词所在句子位置特征，也是做成列表，做统计特征
        # cols += ['diff_max_min_sen_pos']
        # cols += ['diff_var_sen_pos_bili']

        diff_max_min_sen_pos =  [np.NaN for _ in range(len(new_tags))]
        diff_var_sen_pos_bili =  [np.NaN for _ in range(len(new_tags))]  
        for i in range(len(new_tags)):
            pos = [a for a in range(len(words)) if new_tags[i] in words[a]]
            pos_bili = np.array(pos) / len(all_words)
            
            if len(pos) > 0:
                diff_pos = np.diff(pos)
                diff_pos_bili = np.diff(pos_bili)
                if len(diff_pos) > 0:
                    diff_max_min_sen_pos[i] = np.max(diff_pos) - np.min(diff_pos)
                    diff_var_sen_pos_bili[i] = np.var(diff_pos_bili)
                    
#         ## 左右信息熵 没用
#         left_entropy = []
#         right_entropy = []
#         for tag in new_tags:
#             left = []
#             right = []
#             for word in words:
#                 if len(word) < 3:
#                     continue
#                 for i in range(len(word)):
#                     if word[i] == tag:
#                         if i < 1:
#                             left.append('None')
#                             right.append(word[i+1])
#                         if i == (len(word) - 1):
#                             left.append(word[i-1])
#                             right.append('None')
#             left_entropy.append(calc_ent(np.array(left)))
#             right_entropy.append(calc_ent(np.array(right)))
                
        ## 候选关键词之间的相似度 word2vec gensim 窗口默认 迭代默认 向量长度100
        ## sim_tags_arr：相似度矩阵
        sim_tags_arr = np.zeros((len(new_tags), len(new_tags)))
        for i in range(len(new_tags)):
            for j in range(i+1, len(new_tags)):
                if new_tags[i] in wv and new_tags[j] in wv:
                    sim_tags_arr[i, j] = word2vec_model.similarity(new_tags[i], new_tags[j])
                    sim_tags_arr[j, i] = sim_tags_arr[i, j]
            # cols += ['mean_sim_tags']
# cols += ['diff_mean_sim_tags']        
#         max_sim_tags = np.zeros(len(new_tags))
#         min_sim_tags = np.zeros(len(new_tags))
        mean_sim_tags = np.zeros(len(new_tags))
#         var_sim_tags = np.zeros(len(new_tags))
#         skew_sim_tags = np.zeros(len(new_tags))
#         kurt_sim_tags = np.zeros(len(new_tags))
#         max_min_sim_tags = np.zeros(len(new_tags))
#         diff_max_sim_tags = np.zeros(len(new_tags))
#         diff_min_sim_tags = np.zeros(len(new_tags))
        diff_mean_sim_tags = np.zeros(len(new_tags))
#         diff_var_sim_tags = np.zeros(len(new_tags))
#         diff_skew_sim_tags = np.zeros(len(new_tags))
#         diff_kurt_sim_tags = np.zeros(len(new_tags))
#         diff_max_min_sim_tags = np.zeros(len(new_tags))       
        for i in range(len(new_tags)):
#             max_sim_tags[i] = np.max(sim_tags_arr[i])
#             min_sim_tags[i] = np.min(sim_tags_arr[i])
            mean_sim_tags[i] = np.mean(sim_tags_arr[i])
#             var_sim_tags[i] = np.var(sim_tags_arr[i])
#             skew_sim_tags[i] = stats.skew(sim_tags_arr[i])
#             kurt_sim_tags[i] = stats.kurtosis(sim_tags_arr[i])
#             max_min_sim_tags[i] = np.max(sim_tags_arr[i]) - np.min(sim_tags_arr[i])
            diff_sim = np.diff(sim_tags_arr[i])
            if len(diff_sim) > 0:
#                 diff_max_sim_tags[i] = np.max(diff_sim)
#                 diff_min_sim_tags[i] = np.min(diff_sim)
                diff_mean_sim_tags[i] = np.mean(diff_sim)
#                 diff_var_sim_tags[i] = np.var(diff_sim)
#                 diff_skew_sim_tags[i] = stats.skew(diff_sim)
#                 diff_kurt_sim_tags[i] = stats.kurtosis(diff_sim)
#                 diff_max_min_sim_tags[i] = np.max(diff_sim) - np.min(diff_sim)

        ## 候选关键词之间的相似度 word2vec gensim 窗口10 迭代10 向量长度256 
    
        sim_tags_arr_255 = np.zeros((len(new_tags), len(new_tags)))
        for i in range(len(new_tags)):
            for j in range(i+1, len(new_tags)):
                if new_tags[i] in word2vec_model_256 and new_tags[j] in word2vec_model_256:
                    sim_tags_arr_255[i, j] = word2vec_model_256.similarity(new_tags[i], new_tags[j])
                    sim_tags_arr_255[j, i] = sim_tags_arr_255[i, j]
# cols += ['diff_max_min_sim_tags_256']
# cols += ['kurt_sim_tags_256']

        kurt_sim_tags_256 = np.zeros(len(new_tags))
        diff_max_min_sim_tags_256 = np.zeros(len(new_tags))       
        for i in range(len(new_tags)):
            kurt_sim_tags_256[i] = stats.kurtosis(sim_tags_arr_255[i])
            diff_sim = np.diff(sim_tags_arr_255[i])
            if len(diff_sim) > 0:
                diff_max_min_sim_tags_256[i] = np.max(diff_sim) - np.min(diff_sim)   
        
        
        ## label 训练集打标签
        if train:
            label = []
            for tag in new_tags:
                if tag in x.kw:
                    label.append(1)
                else:
                    label.append(0)
                    
        ## 不同词性的比例
        cixing_counter = Counter(new_cixing)
        
        fea = pd.DataFrame()
        fea['id'] = [x['id'] for _ in range(len(new_tags))]
        fea['tags'] = new_tags
        fea['cixing'] = new_cixing


        fea['tfidf'] = new_weight
        fea['ske'] = ske
        
        fea['occur_in_title'] = occur_in_title
        fea['occur_in_first_sentence'] = occur_in_first_sentence
        fea['occur_in_last_sentence'] = occur_in_last_sentence
        fea['occur_in_other_sentence'] = occur_in_other_sentence
        fea['len_tags'] = len_tags
        fea['num_tags'] = num_tags
        fea['num_words'] = num_words
        fea['num_sen'] = num_sen
        fea['classes'] = x['classes']

        fea['len_text'] = len(x['title_cut'] + x['content_cut'])
        fea['textrank'] = textrank
        fea['word_count'] = tf
        fea['tf'] = tf / num_words
        fea['num_head_words'] = len(head_words)
        fea['head_word_count'] = hf
        fea['hf'] = np.array(hf) / len(head_words)
        fea['pr'] = tf / tf.sum()
        fea['has_num'] = has_num
        fea['has_eng'] = has_eng
        fea['is_TV'] = is_TV
        fea['idf'] = v_idf
        fea['sim'] = sim
        fea['sim_euc'] = sim_euc

        fea['mean_l2'] = mean_l2
        fea['meaxl2'] = max_l2
        fea['min_l2'] = min_l2
        
        fea['min_pos'] = min_pos
        fea['diff_min_pos_bili'] = diff_min_pos_bili
        fea['diff_kurt_pos_bili'] = diff_kurt_pos_bili
    
        fea['diff_max_min_sen_pos'] = diff_max_min_sen_pos
        fea['diff_var_sen_pos_bili'] = diff_var_sen_pos_bili

        fea['mean_sim_tags'] = mean_sim_tags
        fea['diff_mean_sim_tags'] = diff_mean_sim_tags

        fea['kurt_sim_tags_256'] = kurt_sim_tags_256
        fea['diff_max_min_sim_tags_256'] = diff_max_min_sim_tags_256
        fea['var_gongxian'] = var_gongxian
        fea['kurt_gongxian'] = kurt_gongxian
        fea['diff_min_gongxian'] = diff_min_gongxian
        
        ## 当前文本候选关键词词性比例
        for c in ['x', 'nz', 'l', 'n', 'v', 'ns', 'j', 'a', 'vn', 'nr', 'eng', 'nrt',
                  't', 'z', 'i', 'b', 'o', 'nt', 'vd', 'c', 's', 'nrfg', 'mq', 'rz',
                  'e', 'y', 'an', 'rr']:
            fea['cixing_{}_num'.format(c)] = cixing_counter[c]
            fea['cixing_{}_bili'.format(c)] = cixing_counter[c] / len(new_cixing)

        if train:
            fea['label'] = label
        res.append(fea)
    return res


# In[ ]:


train_doc = pd.merge(val, all_docs, on='id', how='left')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'res = get_train_df(train_doc, train=True)\n\ntrain_df = pd.concat(res, axis=0).reset_index(drop=True)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'res = get_train_df(all_docs, train=False)\n\ntest_df = pd.concat(res, axis=0).reset_index(drop=True)')


# In[9]:


train_df = pd.to_csv('train_df_v7.csv', index=False)
test_df = pd.to_csv('train_df_v7.csv', index=False)

# train_df = pd.read_csv('train_df_v7.csv')
# test_df = pd.read_csv('train_df_v7.csv')

