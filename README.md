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
  https://aclanthology.org/ 顶会大全
  
  https://paperswithcode.com/task/sentiment-analysis<br>
  
  http://nlpprogress.com/english/sentiment_analysis.html<br>
  
  https://github.com/changwookjun/nlp-paper
  
  https://github.com/declare-lab/awesome-sentiment-analysis#robust-sentiment-analysis 按照不同的情感分析任务总结的论文
  
  https://zhuanlan.zhihu.com/p/376734949 搜集了2014-2020的顶会
  
Paper List
------

Beginner's Guide (Must-Read Papers)
-----
Effects of adjective orientation and gradability on sentence subjectivity

Word sense and subjectivity

Thumbs up?: sentiment classification using machine learning techniques

Thumbs up or thumbs down? semantic orientation applied to unsupervised classification of reviews

A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts

Mining and summarizing customer reviews

Recursive deep models for semantic compositionality over a sentiment treebank

Convolutional neural networks for sentence classification

Contextual valence shifters

SENTIWORDNET: A publicly available lexical resource for opinion mining


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

Aspect Sentiment Classification with Document-level Sentiment Preference Modeling（ACL2020）
本文构建了句子之间的相关网络，其他句子为所预测句子的情感分析任务提供了支持信息。这一方法的假设是短文本（如商品评价）中针对同一问题的情感表述较为一致，甚至整个文本的情感基调都较连贯，因此其他句子的信息可以提供有益的指导。

Target-Guided Structured Attention Network for Target-Dependent Sentiment Analysis
不同于以往将单词作为基本分析单元的研究，本文提出模型分析（如注意力机制）的基本单位应该是语义群（片段）而非单词，并基于这个想法构建了针对对象的语义群注意力机制。最终的结果也表明这样的方法尤其在复杂句子中能更准确地捕捉情感信息。

Modelling Context and Syntactical Features for Aspect-based Sentiment Analysis（ACL2020）
无论是从应用还是理论角度看，对象级情感分析都不应单独进行，而要与对象抽取任务结合起来进行。该文章构建了这样的一体化工具，能充分利用上下文和句法信息，有效地提升了对象级情感分类成绩。


Multi-Task Deep Neural Networks for Natural Language Understanding - Xiaodong Liu(2019)

Aspect-level Sentiment Analysis using AS-Capsules - Yequan Wang(2019)

On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis - Jose Camacho-Collados(2018)

Learned in Translation: Contextualized Word Vectors - Bryan McCann(2018)

Universal Language Model Fine-tuning for Text Classification - Jeremy Howard(2018)

Convolutional Neural Networks with Recurrent Neural Filters - Yi Yang(2018)

Information Aggregation via Dynamic Routing for Sequence Encoding - Jingjing Gong(2018)

Learning to Generate Reviews and Discovering Sentiment - Alec Radford(2017)

A Structured Self-attentive Sentence Embedding - Zhouhan Lin(2017)

North American Chapter of the Association for Computational Linguistics (2021)

NAACL
------

Does syntax matter? A strong baseline for Aspect-based Sentiment Analysis with RoBERTa
Junqi Dai | Hang Yan | Tianxiang Sun | Pengfei Liu | Xipeng Qiu

ASAP: A Chinese Review Dataset Towards Aspect Category Sentiment Analysis and Rating Prediction
Jiahao Bu | Lei Ren | Shuang Zheng | Yang Yang | Jingang Wang | Fuzheng Zhang | Wei Wu

Domain Adaptation for Arabic Cross-Domain and Cross-Dialect Sentiment Analysis from Contextualized Word Embedding
Abdellah El Mekki | Abdelkader El Mahdaouy | Ismail Berrada | Ahmed Khoumsi

Multi-task Learning of Negation and Speculation for Targeted Sentiment Classification
Andrew Moore | Jeremy Barnes

Graph Ensemble Learning over Multiple Dependency Trees for Aspect-level Sentiment Classification
Xiaochen Hou | Peng Qi | Guangtao Wang | Rex Ying | Jing Huang | Xiaodong He | Bowen Zhou

Aspect-based Sentiment Analysis with Type-aware Graph Convolutional Networks and Layer Ensemble
Yuanhe Tian | Guimin Chen | Yan Song

Grey-box Adversarial Attack And Defence For Sentiment Classification
Ying Xu | Xu Zhong | Antonio Jimeno Yepes | Jey Han Lau

Adapting BERT for Continual Learning of a Sequence of Aspect Sentiment Classification Tasks
Zixuan Ke | Hu Xu | Bing Liu

Towards Sentiment and Emotion aided Multi-modal Speech Act Classification in Twitter
Tulika Saha | Apoorva Upadhyaya | Sriparna Saha | Pushpak Bhattacharyya

When and Why a Model Fails? A Human-in-the-loop Error Detection Framework for Sentiment Analysis
Zhe Liu | Yufan Guo | Jalal Mahmud

On the logistical difficulties and findings of Jopara Sentiment Analysis
Marvin Agüero-Torales | David Vilares | Antonio López-Herrera

Unsupervised Self-Training for Sentiment Analysis of Code-Switched Data
Akshat Gupta | Sargam Menghani | Sai Krishna Rallabandi | Alan W Black

Multi-input Recurrent Independent Mechanisms for leveraging knowledge sources: Case studies on sentiment analysis and health text mining
Parsa Bagherzadeh | Sabine Bergler

Improving Cross-Lingual Sentiment Analysis via Conditional Language Adversarial Nets
Hemanth Kandula | Bonan Min

Selective Attention Based Graph Convolutional Networks for Aspect-Level Sentiment Classification
Xiaochen Hou | Jing Huang | Guangtao Wang | Peng Qi | Xiaodong He | Bowen Zhou

ACL
-----
Bridge-Based Active Domain Adaptation for Aspect Term Extraction

Aspect-Category-Opinion-Sentiment Quadruple Extraction with Implicit Aspects and Opinions

Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction

Towards Generative Aspect-Based Sentiment Analysis

Dynamic and Multi-Channel Graph Convolutional Networks for Aspect-Based Sentiment Analysis

Making Flexible Use of Subtasks: A Multiplex Interaction Network for Unified Aspect-based Sentiment Analysis

CTFN: Hierarchical Learning for Multimodal Sentiment Analysis Using Coupled-Translation Fusion Network

Towards Generative Aspect-Based Sentiment Analysis

Cross-Domain Review Generation for Aspect-Based Sentiment Analysis

KinGDOM: Knowledge-Guided DOMain Adaptation for Sentiment Analysis

Modelling Context and Syntactical Features for Aspect-based Sentiment Analysis

Relational Graph Attention Network for Aspect-based Sentiment Analysis

Relation-Aware Collaborative Learning for Unified Aspect-Based Sentiment Analysis

 Adversarial and Domain-Aware BERT for Cross-Domain Sentiment Analysis
 
 Sentiment and Emotion help Sarcasm? A Multi-task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis
 
 EMNLP
 ------
 A Multi-Task Incremental Learning Framework with Category Name Embedding for Aspect-Category Sentiment Analysis
会议：EMNLP 2020. Long Paper.
作者：Zehui Dai, Cheng Peng, Huajie Chen, Yadong Ding
链接：https://www.aclweb.org/anthology/2020.emnlp-main.565.pdf


SentiLARE: Linguistic Knowledge Enhanced Language Representation for Sentiment Analysis
会议：EMNLP 2020. Long Paper.
作者：Pei Ke, Haozhe Ji, Siyang Liu, Xiaoyan Zhu, Minlie Huang
链接：https://www.aclweb.org/anthology/2020.emnlp-main.567.pdf


Aspect-Based Sentiment Analysis by Aspect-Sentiment Joint Embedding
会议：EMNLP 2020. Long Paper.
作者：Jiaxin Huang, Yu Meng, Fang Guo, Heng Ji, Jiawei Han
链接：https://www.aclweb.org/anthology/2020.emnlp-main.568.pdf


 Convolution over Hierarchical Syntactic and Lexical Graphs for Aspect Level Sentiment Analysis
会议：EMNLP 2020. Long Paper.
作者：Mi Zhang, Tieyun Qian
链接：https://www.aclweb.org/anthology/2020.emnlp-main.286.pdf


Multi-Instance Multi-Label Learning Networks for Aspect-Category Sentiment Analysis
会议：EMNLP 2020. Long Paper.
作者：Yuncong Li, Cunxiang Yin, Sheng-hua Zhong, Xu Pan
链接：https://www.aclweb.org/anthology/2020.emnlp-main.287.pdf


 Sentiment Analysis of Tweets Using Heterogeneous Multi-layer Network Representation and Embedding
会议：EMNLP 2020. Long Paper.
作者：Loitongbam Gyanendro Singh, Anasua Mitra, Sanasam Ranbir Singh
链接：https://www.aclweb.org/anthology/2020.emnlp-main.718.pdf


 Unified Feature and Instance Based Domain Adaptation for End-to-End Aspect-based Sentiment Analysis
会议：EMNLP 2020. Long Paper.
作者：Chenggong Gong, Jianfei Yu, Rui Xia
链接：https://www.aclweb.org/anthology/2020.emnlp-main.572.pdf


Public Sentiment Drift Analysis Based on Hierarchical Variational Auto-encoder
会议：EMNLP 2020. Short Paper.
作者：Wenyue Zhang, Xiaoli Li, Yang Li, Suge Wang, Deyu Li, Jian Liao, Jianxing Zheng
链接：https://www.aclweb.org/anthology/2020.emnlp-main.307.pdf

Aspect Based Sentiment Analysis with Aspect-Specific Opinion Spans
会议：EMNLP 2020. Short Paper.
作者：Lu Xu, Lidong Bing, Wei Lu, Fei Huang
链接：https://www.aclweb.org/anthology/2020.emnlp-main.288.pdf


 GRACE: Gradient Harmonized and Cascaded Labeling for Aspect-based Sentiment Analysis
会议：EMNLP 2020. Findings Short Paper.
作者：Huaishao Luo, Lei Ji, Tianrui Li, Daxin Jiang, Nan Duan
链接：https://www.aclweb.org/anthology/2020.findings-emnlp.6.pdf


 Sentiment Analysis with Weighted Graph Convolutional Networks
会议：EMNLP 2020. Findings Short Paper.
作者：Fanyu Meng, Junlan Feng, Danping Yin, Si Chen, Min Hu
链接：https://www.aclweb.org/anthology/2020.findings-emnlp.52.pdf


 Octa: Omissions and Conflicts in Target-Aspect Sentiment Analysis
会议：EMNLP 2020. Findings Short Paper.
作者：Zhe Zhang, Chung-Wei Hang, Munindar Singh
链接：https://www.aclweb.org/anthology/2020.findings-emnlp.149.pdf


 DomBERT: Domain-oriented Language Model for Aspect-based Sentiment Analysis
会议：EMNLP 2020. Findings Short Paper.
作者：Hu Xu, Bing Liu, Lei Shu, Philip Yu
链接：https://www.aclweb.org/anthology/2020.findings-emnlp.156.pdf


 A Shared-Private Representation Model with Coarse-to-Fine Extraction for Target Sentiment Analysis
会议：EMNLP 2020. Findings Short Paper.
作者：Peiqin Lin, Meng Yang
链接：https://www.aclweb.org/anthology/2020.findings-emnlp.382.pdf


Improving Aspect-based Sentiment Analysis with Gated Graph Convolutional Networks and Syntax-based Regulation

AAAI
------
 Word-Level Contextual Sentiment Analysis with Interpretability
 
 Adversarial Training Based Multi-Source Unsupervised Domain Adaptation for Sentiment Analysis
 
 Knowing What, How and Why: A Near Complete Solution for Aspect-Based Sentiment Analysis
 
 Target-Aspect-Sentiment Joint Detection for Aspect-Based Sentiment Analysis
 
 
