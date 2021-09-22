
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



### Language Models are unsupervised multitask learners

https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

GPT2论文初看好像改动不是很大， 但总结下来有两点， 1是提供了一种新的思路，生成式的预训练模型本身可以直接去解决多种任务。 2. 模型的规模大大提升之后取得了非常优异的效果。


新的理念： 之前的大多数任务都会被建模成 P(output|input) ， 但 GPT2的目标是使用同一个无监督的模型去学习多个任务， 具体来讲模型的输出就会变成 P(output|input, task)。  给模型同时输入任务描述（条件），这样就可以针对相同的输入产生不同的输出。 这种建模也是能进行 zero-shot learning 的根本。 

理念的实现方式： 在输入中加入不同的prompt可以达成这样的效果。 比如(translate to french, english text, french text)这样的文本串中，输入translate to french, english text 让模型输出后面的 french text就完成翻译的任务。  刚看到这里的很困惑，为什么GPT2可以清楚的解释不同的prompt并且产生输出？ 可能的解释是，GPT2的模型性能非常强，有语义的prompt可以给GPT2非常强的提示，并且即使这样zero-shot效果也不是特别好的，还是需要给几个例子，即few-shot learning 能产生更好的效果。

数据集： 在 40G 的相对比较高质量的 WebText 数据集上进行了预训练。

模型： 用了48 层的 transformer decoder, 50,257 词表大小， 512的batch_size,  最终参数量是 1.5B。

实验效果： 在zero-shot setting下，在8个语言建模中的7个取得了最好效果。 在阅读理解，翻译，摘要等任务的zero-setting也取得了还不错的效果，但是没达到SOTA。


### Language models are few shot learners

https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf

Nip2020 最佳论文

GPT3模型仍然是相对于GPT2模型的一个增量改进， 其实并没有新的理念提出，但是由于模型的性能更加强大了，也由此让人们对于GPT2提出的理念理解更加深刻了？

理念更新： In context learning,  文章中说，在训练时任务是给context word预测下一个词，但模型在这个过程中也会对于文本的模式进行自动的学习，来更好的对下一个词进行预测。 这样就引出了一种新的模式： 不需要梯度更新的few-shot learning。 直接把few-shot 的case作为提示放在文本前面（比较长的prompt）,GPT3会自动学习这种模型然后产生对应的输出。  由于GPT3非常强大的模型性能和prompt多样性，GPT3真实现了一个效果还不错的 zero-shot learner。  

数据集： 在Common Crawl, WebText2, Books1, Books2 and Wikipedia 5个数据集上进行了预训练。

模型： 使用了96层的 transformer decoder，每个decoder有96个attention heads。  最终有  175 billion 参数。 

实验结果： 在 语言建模任务上， 在 zero-shot setting 下就能超过SOTA，在非常多其他的任务上比如翻译，问答也能使用  zero-shot setting 或者  one-shot setting  达到最优效果或者接近最优效果。


### Its Not Just Size That Matters Small Language Models Are Also Few-Shot Learners

https://aclanthology.org/2021.naacl-main.185.pdf

NAACL 2021

GPT3 模型在很多任务的 few-shot learning 设定上取得了非常出色的效果， 但是GPT3需要的参数量十分巨大，一般的研究者很难去使用。 这篇文章提出了一个使用cloze question 配合梯度更新的方法， 在只有 GPT参数量 0.1% 的情况下在一些任务上取得了比GPT3更好的效果。 

先介绍本文的前序工作， PET（pattern exploiting training）。 PET把很多NLP任务建模成了以下步骤：
1.  通过一个 pattern P 把 输入文本 X 变成 T*, T* 中有 cloze question ，包含一个mask。 
2. 通过 预训练语言模型 预测其中的mask, 产生输出 Y。 
3. 使用一个 verbalizer V 把Y映射到T，其中 T 是该NLP任务的特定符号，比如情感分析的两个类别。 
这样的一个 pattern-verbalizer pairs 就是 PVPs。
同时PET中还使用了多个 PVP，使其相互学习，从无监督数据中增强了模型的效果（有点类似self-training）。

本文在PET上做了一点改进，之前的PET输出只能是一个token, 不能满足多种NLP任务的需要，其实就很类似 seq 的生成了，先预测第一个token,取概率最大再去预测下一个token。 作者分成 inference 和 train 来去介绍，很类似seq2seq learning的基本 setting。

实验： 在QA任务， Text entailment, 问答的多个数据集上做了实验，在这些数据集大都能取得比GPT3更好的效果。

### KnowPrompt Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction 

https://arxiv.org/pdf/2104.07650.pdf

近期，prompt-tuning 方法在 few-shot setting 下取得了不错的进展，但是在关系抽取领域中，如何更好的设计prompt仍然是一个需要领域专家的工作。 本文提出了一个将知识引入 prompt-tuning 的方法来去解决关系抽取问题。

方法，首先还是按照经典的 prompt 方式转换输入，把句子变成 X  E1 【mask】 E2, 这种形式， X是输入的句子，E1，E2是实体，【mask】是要预测的关系，之前的prompt方法是把mask预测出的词和目标label做一个一对一映射，这样做就没有用到关系标签的丰富语义。 于是本文做了如下的改变
1. 在对mask进行预测的时候，在输出层中，把输入的维度进行扩展，维度从词表的大小扩展到词表的大小+关系的数量。 直接看输出结果在后面 关系数量大小的维度上 logit 来判断类别。这样mask language 的loss就可以直接作为一个交叉熵。
2. 在实体前后加入特殊符号 [sub] 和 [obj], 使用实体类型对应的向量来进行初始化。  把关系向量使用其中包含单词的向量的来初始化。 然后设置了一个 KE loss， 就是把构成三元组的实体 |h+r -t|尽量小，再负采用一些数据，这些数据的 |h+r -t|尽量大， 有点对比学习的意思。
实验效果，在5个数据集的标准设定和低资源设定下都取得了不错的效果。


### Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference 

EACL 2021

https://aclanthology.org/2021.eacl-main.20.pdf

有一些工作通过给预训练模型一些任务描述来无监督的解决这些问题，但是这种方法一般是比相对应的监督学习方法差的。 本文就提出了一种Pattern- 
Exploiting Training (PET) 方法，是一种半监督学习方法，通过把问题建模成完形填空来增强模型对于问题本身的理解。

PET把很多NLP任务建模成了以下步骤：
1.  通过一个 pattern P 把 输入文本 X 变成 T*, T* 中有 cloze question ，包含一个mask。 
2. 通过 预训练语言模型 预测其中的mask, 产生输出 Y。 
3. 使用一个 verbalizer V 把Y映射到T，其中 T 是该NLP任务的特定符号，比如情感分析的两个类别。 
这样的一个 pattern-verbalizer pairs 就是 PVPs。
同时PET中还使用了多个 PVP，使其相互学习，从无监督数据中增强了模型的效果（有点类似self-training）。

本文在 Yelp， AG’s News ， Yahoo ，MNLI 的 few-shot setting 下都取得了不错的效果。
