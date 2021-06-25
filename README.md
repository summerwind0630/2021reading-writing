# 2021reading-writing
论文阅读与写作实践课程


建立并维护某领域论文清单。
  报告项目简介（3-5min）
  维护更新到本学期结束
    --补充领域介绍文档
    --每篇论文入选理由，1-3句话介绍论文亮点
    -随时更新
    
Sentiment analysis Survey<br>
---
  1、Deep Learning for Sentiment Analysis：A Survey(2018年)<br>该文首先对深度学习的技术进行了概述，然后对基于深度学习的情感分析研究进行了全面的综述<br>  
  2、Recent advances in deep learning based sentiment analysis(2020年）<br>老师推荐，从粗粒度情感分类任务到细粒度情感挖掘任务，从表层情感检测到隐性情感分析任务，从理解文本情感到生成有情感的文本任务。本文简要介绍了基于深度学习的情感分析方法在情感分析任务中的最新进展，包括方法总结和数据集分析。这个调查很适合研究这个领域的研究者和进入这个领域的研究者<br>  
  3、Current Challenges and New Directions in Sentiment Analysis Research（2020）<br>在情感分析领域目前的挑战以及新的方向<br>  
4、文本情感分析方法研究综述（2021计算机工程与应用）  

  书籍
  ------
  《情感分析》刘兵<br>
  《文本情感分析》林政<br>
  
  USEFUL URL
  ------
  https://paperswithcode.com/task/sentiment-analysis<br>
  http://nlpprogress.com/english/sentiment_analysis.html<br>
  https://github.com/changwookjun/nlp-paper
  
Paper List
------
硕博论文
------
<br>《多层次文本情感分析研究_王业全》<br>
本文对多个层次的文本进行了情感分析方面的研究。在要素级别，我们探索了特定要素情感分类和要素情感协同分析两个任务，分别利用基于要素的注意力机制和胶囊网络进行情感分析；在文档级别，我们利用胶囊网络，提出了具有可解释性的方案；在多轮次对话级别，我们针对目前缺乏高质量数据集的现状，构建了数据集，并初步提出解决架构<br>
<br>《面向短文本情感分析的方法及应用研究_李扬》<br>从词向量模型入手，研究了如何在词向量模型中加入情感先验，提高词向量的表示能力；其次探索了如何解决短文本数据不足的问题；接着探索了如何在短文本中挖掘其潜在变量，将数据的特征明显的表示出来；最后本文在短文本情感分析的基础上探索了如何进行社交媒体上敏感信息的识别。
<br>《面向数据特性的文本情感分析方法研究_李旸》<br>本文以社会媒体文本数据为研究对象，重点围绕文本情感分析领域的文本情感分类、反问句识别和反讽句识别、方面项级情感分析及其在可解释性推荐中的运用
等任务，聚焦于类别分布非平衡、标签数据缺乏、情感表达方式隐晦和情感载体多样等数据特性给情感分析任务带来的挑战，分别单独或融合运用数据采样、半监督学习、深度学习、文本嵌入表示和图神经网络等理论与技术，研究并提出稠密混合区域采样、初始种子选择、训练数据集更新、特征自动抽取、多种情感信息融合、用户 -产品关系建模等关键问题的解决思路、方法或算法，并进行实验验证。
<br>《细粒度文本情感分析问题研究_杨骏》<br>我看看第一章绪论，SA的研究背景，给出了代表性语料库，情感词典。

会议期刊
-----
A Unified Generative Framework for Aspect-Based Sentiment Analysis （ACL2021.6）<br>
总结了ABSA的7个子任务，指出现有的一些work还没有很好的有一个统一的框架去一块解决这7类子任务，现在大多是1-3个子任务一起做。而本文构造了一个序列生成任务（预测aspect-term与opinion-term的位置），可以一次性解决这7类子任务。非常好的一个新想法。  
SKEP: Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis (ACL2020)<br>
百度提出了一种情感知识增强的语言模型预训练方法，在通用预训练的基础上，设计了面向情感知识建模的Masking策略和情感多目标学习算法，融合了情感词、极性、主体评论关系等多层情感知识，首次实现了情感任务统一的文本预训练表示学习。  
Sentiment Analysis by Capsules (www2018)<br>
模型的关键思想是设计一个简单的胶囊结构，并使用每个胶囊专注于特定的情感类别。每个胶囊输出其状态概率和重建表示!实验表明，在没有借助语言学知识的情况下，这个简单的基于胶囊的模型取得了迄今为止最优秀的情感分类性能,胶囊能够注意到具有可解释性的词语  

A Hierarchical Model of Reviews for Aspect-based Sentiment Analysis（EMNLP2016）  
除了各句子内部的信息以外，句子与句子间的相关性对于情感分析也是十分重要的，故而本文同时获取句子级别信息(sentence-level)和评论级别信息(review-level)。.提出hierarchical bidirectional long short-term memory(H-LSTM)模型，综合评论的句子内部信息特征、句子间信息特征、句子的实体属性特征(eg.FOOD#QUALITY)，进行情感分析。最终证明本文提出模型具有更好的效果且不需要文本以外信息的结论。与Best、XRCE、IIT-TUDA这些综合了文本之外的信息的模型比较，本文的H-LSTM只需要使用文本本身信息，能获得势均力敌(competitive performance)的结果。  

Convolutional Neural Networks for Sentence Classification(2014)
元老级文章。本文从Embedding层进行验证实验，论证强调预训练(pre-train)和微调(fine-tune)对模型结果的巨大改观作用。br<>
Sentiment Analysis
Multi-Task Deep Neural Networks for Natural Language Understanding - Xiaodong Liu(2019)

Aspect-level Sentiment Analysis using AS-Capsules - Yequan Wang(2019)

On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis - Jose Camacho-Collados(2018)

Learned in Translation: Contextualized Word Vectors - Bryan McCann(2018)

Universal Language Model Fine-tuning for Text Classification - Jeremy Howard(2018)

Convolutional Neural Networks with Recurrent Neural Filters - Yi Yang(2018)

Information Aggregation via Dynamic Routing for Sequence Encoding - Jingjing Gong(2018)

Learning to Generate Reviews and Discovering Sentiment - Alec Radford(2017)

A Structured Self-attentive Sentence Embedding - Zhouhan Lin(2017)
