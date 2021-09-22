
### Pre-train Prompt and Predict A Systematic Survey of Prompting Methods in Natural Language Processing 

https://arxiv.org/pdf/2107.13586

一篇关于 prompt-learning 的综述文章。

本文首先梳理了NLP的发展历程，第一阶段是手动构建特征让模型进行学习， 可以总结为 feature engineering ， 第二阶段引入了深度神经网络，模型可以自己学习特征，但是仍然需要对模型架构进行设计，称为architecture engineering。  第三阶段监督学习的方式遇到了瓶颈，形成了新的范式是 pre-train and fine-tune paradigm， 在这个阶段需要去做的是objective engineering，即需要去设置fine-tune的目标函数。 在最近出现了一个更新的小阶段， 可以被称为pre-train, prompt, and predict ， 即首先预训练模型，然后设计prompt，之后去预测。 这个阶段强调的就是 prompt engineering 。

第二个部分作者对prompt的基本概念进行介绍，把使用prompt的方法总结成了三个步骤。 第一个步骤是 Prompt Addition,  即对输入X使用一个模版，模版中包含了一些slot。 如果slot在中间，任务就被称为 cloze prompt(完形填空)， 如果prompt在最后就是 prefix prompt。  并且值得注意的是模版word不一定是自然语言词，也可能是连续向量。  第二个步骤是 Answer Search, 即找到一个词填充上 prompt 使得LM的得分最高。 一般会预先定义一个答案集合，如果是分类任务，会选出特定的词来对应类别，如果是生成任务，可能是全词表。  第三个步骤是Answer Mapping 
， 生成的词是词表空间的表示，需要mapping到label到空间。  之后作者分成了几个方面去讲解了prompting方法需要注意的问题，分别是 pre-train model choice,  prompt engineering, answer engineering, 	 Expanding the Paradigm,  Prompt-based Training Strategies。  

Prompt engineering： 这一章节列举了一些产生prompt的方式。  第一种是Manual Template Engineering，即人工设计模版。  但人工很难设计出最优的 prompt。 第二种是Automated Template Learning， Automated Template Learning 是目前研究的主流了，可以从多个视角进行分析， 首先可以分成离散的prompt和连续的prompt 设计，还可以分成静态和动态（即对于不同的输入有不同的prompt）。  对于离散的 prompt,  几种相对主流的方法分别是1.  Prompt Mining,  这是一种数据挖掘的方式，在很大的数据集（比如 wikipedia） 中找到x和y的最佳依赖路径。 2. Prompt Paraphrasing， 这种方法的原理是先找到一个人工设计的prompt，然后用一系列自动的方法对其进行改写，比如back-translation 等等，从这当中找出最好的一个 prompt。3. Gradient-based Search ， 对于给定的预测， 使得模型去优化prompt的词选择，通过一种迭代的方式最终选出prompt。  4. Prompt Generation  ,  有两篇相关的工作是使用 T5 这样的生成式预训练模型来生成模版，感觉是类似于定义了一个模版的模版，定义了在哪里插入输入数据，然后使用预训练模型来生成模版。   5. Prompt Scoring ，  一种给 prompt打分的方法，首先给 人工设计出一些prompt， 然后使用语言模型来对这些 prompt进行打分。  然后是 Continuous Prompts， 其实 prompt并没有必要必须要使用人类能能够读懂的语言，只要是能辅助语言模型完成具体的任务就可以， 所以可以直接建模 prompt 为编码空间中的向量。 这样做带来了两个好处， 第一个是取消了模版的单词必须是自然语言这个设定。 二是模版的编码和语言模型的参数无关，由自己的参数进行驱动，有自己的参数，可以使用下游数据进行fine-tune。    Continuous Prompts 的方法可以分成下面几类， 1. Prefix Tuning， 在输入的 x 前加入一些连续的 prompt (个人理解是正常的token id需要经过embedding层变成向量然后经过transformer编码， 而连续的prompt需要直接给一个 embedding，不需要transformer的embedding层了)， 把预训练模型的参数冻结，只去调prompt的embedding。  2.  Tuning Initialized with Discrete Prompts， 使用已经搜索好的离散 prompt进行初始化，然后再进行优化。   3. Hard-Soft Prompt Hybrid Tuning, 把hard prompt 和 soft prompt结合起来进行优化，P-tuning 在learnable prompt template 中增加了一点 anchor word 提升了效果。

Answer Engineering:  首先是 answer 的形式，一般来说有三种 token, span(multi-token), sentence, 一般 token, span会用在分类任务上， sentence会用在生成任务上。  比较关键的是从词空间到label的映射过程，即Answer Space Design Methods， 有三种方法。 1.  Manual Design ， 一般生成任务都是 identity mapping， 在全词表上生成，没有限制。  2. 很多分类任务会对输出的词表进行限制，把相关的词和一些类别进行对应， 一般也是一一对应的。  2.  Discrete Answer Search   人工设计很难达到最优，所以有一些工作研究自动搜索mapping关系， 经典的离散搜索方法包括  Answer Paraphrasing （使用 back-translation 对答案进行各种改写，改写后的作为词表词）， Prune-then-Search （大概先匹配出可能的token集合，然后在数据集上去最大似然？）， Label Decomposition （把label拆成好多个词，这些词就是需要的token）。 Continuous Answer Search（也是类似于连续prompt， 使用一个连续的embedding作为answer词的表达）。

Multi-Prompt Learning: 使用多个 prompt的方法， prompt ensembling(对同一个任务同时采用多个 prompt进行集成学习)， Prompt augmentation （其实就是 few-shot learning 了， prompt就是一些该数据集的例子）。 Prompt Composition （把一个任务拆解成多个子prompt任务，然后再集合起来）， Prompt Composition （把一个比较长的 prompt拆解成子prompt分开去解决）
。
Training Strategies for Prompting Methods， 主要是参数的更新方法 1. Promptless Fine-tuning， 就是最经典的 pretrian-finetune。 2. Tuning-free Prompting （像GPT3那样的完全不需要调节任何参数）3. Fixed-LM Prompt Tuning （只调节 prompt相关的参数，不去调节语言模型的参数）3.  Fixed-prompt LM Tuning （固定prompt参数，去调节语言模型） 4. Prompt+LM Tuning （对于prompt和语言模型全都调节）。

Applications: 目前 prompt方法相关的应用，只列举几个比较关心的， semantic parsing(Constrained Language Models Yield Few-Shot Semantic Parsers)： 看成了一个 text-to-text的任务，默认了每个方法提供了validNextTokens，来限制decode的输出。  Text Generation（使用 prefix prompt对于text-generation还是比较自然的）。

几个和prompt比较相关的话题， Ensemble Learning （集成学习）， Few-shot Learning ， Larger-context Learning （在输入中增加更多的context信息）， Query Reformulation， QA-based Task Formulation， Controlled Generation ， Data Augmentation。

挑战：  1. prompt 设计，目前大多数应用还是在 分类和生成任务，对于信息抽取的等任务仍然很少有做。  使用prompt来生成结构化信息也很少有工作在做。 如何同时考虑 prompt和answer的设计也是一个比较大的挑战。  2. Answer Engineering, 分类任务的两个挑战是如何选取最优answer space和答案有多个词的时候如何去生成（X-FACTR: Multilingual factual knowledge retrieval from pretrained language models. ）。 生成任务的挑战，多reference的学习如何设计？  


### Constrained Language Models Yield Few-Shot Semantic Parsers 

https://arxiv.org/pdf/1902.09492.pdf

这篇文章探索了 大规模预训练模型能否在 semantic parsing 任务上做  Few-Shot  learning， 由于semantic parsing 任务生成的都是结构化表示，但预训练模型生成的都是自然语言，所以本文首先建立出了一种可控标准的英文表达形式（这种形式和结构化表示是一一对应的），可以通过规则化的代码相互转化。  然后控制预训练模型去生成这种可控的标准英文表达形式。  

本文的方法主要有两个点，1是Dynamic Prompt Creation， 其实就是用example作为prompt, 使用了GPT3动态选取了example。 2. 是Constrained Decoding， 每一步decode并不是从全词表生成，而是设置了一个validNextTokens 函数，这个函数来判断可以生成的下一个token范围，来确保生成的句子是满足固定格式的。 本文没有在GPT3上没有fine-tune, 在一些其他相对比较小的模型上也使用了一些方法进行fine-tune。

实验结果，在Overnight,Break , SMCalFlow 数据集上都在few-shot setting下进行了实验， 在overnight数据集上能取得和全量数据类似的结果，在其他两个数据集上离全量数据集SOTA方法差距还比较大。


