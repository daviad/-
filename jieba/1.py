# https://github.com/fxsjy/jieba

# -*- coding: utf-8 -*-
import jieba
import jieba.posseg as pseg
import jieba.analyse
'''

jieba分词的

Feature

–

支持三种分词模式 
1 精确模式，试图将句子最精确地切开，适合文本分析； 
2 全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义； 
3 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
支持繁体分词
支持自定义词典

'''


# str1 = "我来到北京清华大学"
# str2 = 'python的正则表达式是好用的'
# str3 = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"

# seg_list = jieba.cut(str1,cut_all = True)  ##全模式
# result = pseg.cut(str1)  ##词性标注，标注句子分词后每个词的词性
# result2 = jieba.cut(str2)  ##默认是精准模式
# result3 = jieba.analyse.extract_tags(str1,2) ##关键词提取，参数setence对应str1为待提取的文本,topK对应2为返回几个TF/IDF权重最大的关键词，默认值为20
# result4 = jieba.cut_for_search(str3)  ##搜索引擎模式

# print (" /".join(seg_list))

# for w in  result:
#     print(w.word, "/", w.flag, ", ",)

# for t in result2:
#     print (t)

# for s in result3:
#     print (s)

# print (" ,".join(result4))

# 1.使用jieba分词对中文文档进行分词

import jieba
 
 
# jieba.load_userdict('userdict.txt')
# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
 
 
# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('./test/stopwords.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr
 
 
inputs = open('./test/input.txt', 'r', encoding='utf-8')
outputs = open('./test/output.txt', 'w')
for line in inputs:
    line_seg = seg_sentence(line)  # 这里的返回值是字符串
    outputs.write(line_seg + '\n')
outputs.close()
inputs.close()



# 2.停用词表去重

# 从网上收集来的停用词可能有重复的，下面的代码去重
# 停用词表按照行进行存储，每一行只有一个词语
# python3
def stopwd_reduction(infilepath, outfilepath):
    infile = open(infilepath, 'r', encoding='utf-8')
    outfile = open(outfilepath, 'w')
    stopwordslist = []
    for str in infile.read().split('\n'):
        if str not in stopwordslist:
            stopwordslist.append(str)
            outfile.write(str + '\n')
 
 
stopwd_reduction('./test/stopwords.txt', './test/stopword.txt')

