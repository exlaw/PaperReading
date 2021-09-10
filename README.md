## 论文阅读

阅读论文后自己写总结还是很重要的，一方面提升阅读效果，二是在今后再阅读文章时可以帮助快速回忆，节省时间。

- [论文阅读](#----)
  * [Exploring Auxiliary Reasoning Tasks for Task-oriented Dialog Systems with Meta Cooperative Learning](#exploring-auxiliary-reasoning-tasks-for-task-oriented-dialog-systems-with-meta-cooperative-learning)
  * [Awakening Latent Grounding from Pretrained Language Models for Semantic Parsing](#awakening-latent-grounding-from-pretrained-language-models-for-semantic-parsing)
  * [Value-Agnostic Conversational Semantic Parsing](#value-agnostic-conversational-semantic-parsing)
  * [Vocabulary Learning via Optimal Transport for Neural Machine Translation](#vocabulary-learning-via-optimal-transport-for-neural-machine-translation)
  * [SMBOP Semi-autoregressive Bottom-up Semantic Parsing](#smbop-semi-autoregressive-bottom-up-semantic-parsing)
  * [ConSERT A Contrastive Framework for Self-Supervised Sentence Representation Transfer](#consert-a-contrastive-framework-for-self-supervised-sentence-representation-transfer)
  * [SimCSE Simple Contrastive Learning of Sentence Embeddings](#simcse-simple-contrastive-learning-of-sentence-embeddings)
  * [Keep the Structure: A Latent Shift-Reduce Parser for Semantic Parsing](#keep-the-structure--a-latent-shift-reduce-parser-for-semantic-parsing)
  * [MEDA Meta-Learning with Data Augmentation for Few-Shot Text Classification](#meda-meta-learning-with-data-augmentation-for-few-shot-text-classification)
  * [All NLP Tasks Are Generation Tasks: A General Pretraining Framework](#all-nlp-tasks-are-generation-tasks--a-general-pretraining-framework)
  * [Joint Verification and Reranking for Open Fact Checking Over Tables](#joint-verification-and-reranking-for-open-fact-checking-over-tables)
  * [Towards Table-to-Text Generation with Numerical Reasoning](#towards-table-to-text-generation-with-numerical-reasoning)
  * [Towards Robustness of Text-to-SQL Models against Synonym Substitution](#towards-robustness-of-text-to-sql-models-against-synonym-substitution)
  * [LGESQL Line Graph Enhanced Text-to-SQL Model with Mixed Local and Non-Local Relations](#lgesql-line-graph-enhanced-text-to-sql-model-with-mixed-local-and-non-local-relations)
  * [Optimizing Deeper Transformers on Small Datasets](#optimizing-deeper-transformers-on-small-datasets)
  * [From Paraphrasing to Semantic Parsing: Unsupervised Semantic Parsing via Synchronous Semantic Decoding](#from-paraphrasing-to-semantic-parsing--unsupervised-semantic-parsing-via-synchronous-semantic-decoding)
  * [Span-based Semantic Parsing for Compositional Generalization](#span-based-semantic-parsing-for-compositional-generalization)
  * [Compositional Generalization and Natural Language Variation: Can a Semantic Parsing Approach Handle Both?](#compositional-generalization-and-natural-language-variation--can-a-semantic-parsing-approach-handle-both-)
  * [On the Sentence Embeddings from Pre-trained Language Models](#on-the-sentence-embeddings-from-pre-trained-language-models)
  * [All That’s ‘Human’ Is Not Gold: Evaluating Human Evaluation of Generated Text](#all-that-s--human--is-not-gold--evaluating-human-evaluation-of-generated-text)
  * [KILT a Benchmark for Knowledge Intensive Language Tasks](#kilt-a-benchmark-for-knowledge-intensive-language-tasks)
  * [WIKITABLET A Large-Scale Data-to-Text Dataset for Generating Wikipedia Article Sections](#wikitablet-a-large-scale-data-to-text-dataset-for-generating-wikipedia-article-sections)
  * [Describing a Knowledge Base](#describing-a-knowledge-base)
  * [GenWiki A Dataset of 1.3 Million Content-Sharing Text and Graphs for Unsupervised Graph-to-Text Generation](#genwiki-a-dataset-of-13-million-content-sharing-text-and-graphs-for-unsupervised-graph-to-text-generation)
  * [WikiGraphs A Wikipedia Text Knowledge Graph Paired Dataset](#wikigraphs-a-wikipedia-text-knowledge-graph-paired-dataset)
  * [Text-to-Table A New Way of Information Extraction](#text-to-table-a-new-way-of-information-extraction)
  * [BART Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](#bart-denoising-sequence-to-sequence-pre-training-for-natural-language-generation--translation--and-comprehension)
  * [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](#exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


### Exploring Auxiliary Reasoning Tasks for Task-oriented Dialog Systems with Meta Cooperative Learning 

AAAI 2021

https://ojs.aaai.org/index.php/AAAI/article/view/17615

AAAI 2021 的一篇论文，做的是任务型对话系统。   任务型对话的目标是通过和用户的对话达到某种特定的目的。 具体到任务上，输入是用户每一轮的问题和知识库（KB），输出是系统产生的回答。
尽管近期的很多采用seq2seq的方法取得了不错的效果，但是 seq2seq的方法也存在着两个问题 1. 是很难在知识库上进行推断（reasoning), 这导致了系统容易产生无关的回答。 2 是seq2seq方法很容易忽略语境信息，这导致了系统经常忽略关键信息。 

针对以上问题，本文采用了多任务学习的方式对seq2seq方法的不足进行了一些修正，新增了两个辅助任务，分别是 KB reasoning task 和 dialogue reasoning task。 并且采用了一个 meta cooperative learning 的方式来对三个任务进行统一的学习。

本文的baseline model 采用了经典模型，包括了dialogue encoder, dialogue memory(用来保存历史对话语句和状态)， KB Memory(用来保存历史知识库信息和状态)，dialogue decoder（用来产生最后的结果）。  输出方式采用类似 copy net的方式进行，用动态概率控制是从词汇表中选择、KB中选择还是历史对话中选择词汇。
第一个辅助任务是KB reasoning, 不使用 dialogue memory， 增加一个新的 KB reasoning network模块，目标是更好的学习  KB Memory， 最终的loss是和 标准主网络输出logit的KL散度比较。
第二个辅助任务是 dialogue reasoning, 不使用 KB memory, 新增一个 dialogue  reasoning 模块，目标是更好的学习 dialogue memory, 最终的loss 是和标准主网络logit的KL散度比较。
由于两个辅助任务的网络loss都是和主网络的KL散度比较，所有自然就有了梯度更新顺序问题，本文采用的算法是类似MAML的一个meta learning思想，取名为 meta cooperative learning 。

从最终的实验效果来看，在 CamRest， In-Car Assistant  和 Multi-WOZ 的各项指标上都取得了最优效果。 


### Awakening Latent Grounding from Pretrained Language Models for Semantic Parsing 

ACL2021 findings

https://aclanthology.org/2021.findings-acl.100.pdf

探索 大规模预训练模型 grounding capabilities  的工作比较少，本文就来探索  的 grounding capabilities , 具体来说就是 token和概念的对应关系，找到句子中每个token和相应概念的对应关系就是  grounding， 比如 text-to-sql 中的找到文本和数据库schema， 知识库 entity 之间的对应关系。  本文通过应用一个简单的 erase-Awaken 机制，在不需要标注（token级别）的情况下学到了grounding信息，在后续的实验中，应用到 text-to-sql 领域，可以取得最高9.8%的提升。

方法分成三个步骤，第一个步骤是训练一个概念预测模块，这个模块的目的是判断一个句子是否提到了某个概念。 在 text-to-sql 领域中，对应的是数据库中的schema信息是否在 SQL语句中出现，这个标注信息已经存在，不需要再进行额外的标注。  第二个步骤是 擦除token,  然后来判断擦除token后对某个概念的 预测分数差别，如果差别很大，那么说明这个token对这个概念能产生非常大的影响，否则就不会产生影响，这样也就产生了一个相关矩阵。 第三个步骤是用上一步产生的相关矩阵作为监督信号再去训练一个Latent Grounding的预测模块（是根据实验效果直接使用相关矩阵作为预测结果效果并不好，本文认为Latent Grounding的预测模块能够加大不同模块之间的差别）。

实验主要分成两个部分，第一个部分是对 Grounding 的预测准确率判断，和之前人工标注的Grounding信息进行对比，根据作者的实验结果，效果大幅度超过了其他的模型，并且能够取得和完全监督学习类似的效果。 第二个部分是应用到 text-to-sql任务中作为 linking模块的数据，对SLSQL 等比较简单的模型都能产生相对较大的提升。

### Value-Agnostic Conversational Semantic Parsing 

ACL 2021

https://aclanthology.org/2021.acl-long.284.pdf

对话场景下的 semantic parsing 目前大多数做法都把之前轮次的对话全部都进行编码，这样会导致计算效率低，并且产生一些没有必要的依赖。 本文认为，只需要之前预测结果 program 的类型信息，不需要具体的值 （Value-Agnostic）就可以产生非常好的效果。 本文的贡献： 1.  使用了一个紧编码，这个编码只包含了历史program 的 type， 但是本文仍然认为这个信息是足够的。  2.  在 decode的时候把对function的预测和对于argument的预测做了独立，分别预测提升了计算效率，可以允许进行更大的 beam search 提高最后的结果。

具体方法： 在编码的过程中，首先通过 transformer 模型把句子编码，然后把句子的编码和对之前program的类型信息作一个attention, 产生对句子的编码。  在解码过程中，对函数序列的预测和对于参数的预测独立开来，不再是作一个 tree 结构预测或者是序列预测， 有点类似于分层的序列预测。  其中具体的参数值设置了四个来源： 之前函数调用的引用，静态词表中的值，句子中复制的字符串，实体提取器中得到的实体。  最终的实验结果中，在SMCALFLOW 和 TREEDST 两个数据集中分别提升了 7.3% 和 10.6%。

### Vocabulary Learning via Optimal Transport for Neural Machine Translation

ACL 2021 的最佳论文

https://aclanthology.org/2021.acl-long.571.pdf

token 词表的选择在NLP中非常重要， 传统的词表选择有基于character和基于word的，基于character的词表选择长度太长，学习难度大， 基于word的词表容易在 rare word熵出现 OOV 的问题， 所以之前的工作也提出了BPE的方法选择的高频的sub-words或者sub-word pieces来做更好的词表，这样使用更小的词表可以压缩数据，减小数据集的熵。 但是使用 BPE 方法的时候选用多大的词表是一个 trade-off问题，词表的增大可以使得语料库熵减小，但是会导致token sparsity， 之前使用 tral learning 需要遍历全部的情况，非常耗费时间，为此，作者提出了一种基于最优传输的词汇学习方法，简称VOLT。

具体来说，作者提出了 MUV（marginal utility of vocabularization ）的概念， 即词表的边际效应，整个算法的 优化目标就是 MUV， 首先作者进行了一些简单的实验，发现 MUV 曲线和最终的BLEU值相关程度非常高，说明优化MUV是有价值的。  直观地说，作者将词汇表构造想象为一个传输过程，将字符传输到数量高达S[t]的候选token中。由于字符数是固定的，并非所有候选token都能获得足够的字符。每个传输矩阵都可以通过收集带有字符的token来构建词汇表，最优传输的目标是找到一个传输矩阵来最小化传输成本，即在作者的设置中为负熵。 后面的数学处理相对比较硬核，就先不详细介绍了。

在实验部分，在多个数据集上都能取得很稳定的效果，在词表大小减小为1/3的情况下，BLEU值甚至能持平或者更好（高0.5%的点已经非常多了）。   总体来看， 本文基于optimal transport的思想，很新颖，并且实验的结果非常充分，solid。

### SMBOP Semi-autoregressive Bottom-up Semantic Parsing 

NAACL 2021 

https://arxiv.org/pdf/2010.12412.pdf

目前几乎大多数的 Text-to-SQL 方法采用的都是 top-down 的decoding 方法，也就是一种auto-regressive 的方法，从根节点开始，由上至下，由左至右逐渐decode成整个语法树。  在这个工作中，尝试了一种不同的decode方式，也就是一种 bottom up 的decode方式，具体含义就是在第t步去构造一个高度不超过t的 top-K 个子树。 这样做的优势在于，可以并行化的生成子树，使得算法的复杂度由线性变成了对数复杂度。 根据作者的实际数据，在SPIDER 数据集上的训练加速可以达到5倍，推断加速达到2.2倍。

在具体的方法上，作者重新对SQL的语法进行改写，形成了一套特殊的关系代数，和自然语言的形式相对更接近一点，更重要的是增加了keep操作，使得所有的SQL树都是平衡树（子树高度相同）。  后续的操作就是每一层的筛选，第一层是数据库的schema 和数据库中的值，对每种可能对匹配都去计算score,包括对一个子树的操作（unary）和对两个子树的操作（binary）， 形成所有候选后，根据score值选取 K 个进入下一层的计算，并且过程中还使用了 cross attention的方法去更新了子树的 向量表示。  loss值的计算是对标准子树的最大似然。

在实验结果上，本文在 SPIDER 数据集上的实验结果和RAT-SQL 类似， 只有 0.3%的点的损失，考虑到采用了Semi-autoregressive的情况下，已经是不错的结果了。 同时本文也分析了在 senmantic parsing的场景下，bottom-up方法和 top-down方法在理论上的差距不是很大。

### ConSERT A Contrastive Framework for Self-Supervised Sentence Representation Transfer 

ACL 2021

https://arxiv.org/pdf/2105.11741.pdf

BERT在对句子的编码中会出现崩塌的现象，即几乎所有的句子都会被编码到一个相对比较小的空间中，这导致了计算句子相关性的时候会把一些不太相关的句子也给出相对比较高的相似度。 具体的原因是BERT对于高频词的编码相对来说比较集中，其他的相对低频的词汇则很分散，这导致了对整个句子计算编码的时候会很大程度的受到高频词的影响。 

为了解决这个问题，作者提出了一个基于对比学习的方式，结合数据增强让BERT更好的学到句子相关性。 具体来说，本文使用了四种数据增强的方式，Adversarial Attack （打乱句子编码）， Token Shuffling （把句子中的Token打乱）， Cutoff （在句子中删除一些Token）, Dropout 
(把Embedding中的一些维度去掉)。  具体的学习loss和标准的对比学习loss是类似的，数据增强后的数据为正例，和原数据尽可能接近，其他的句子是负例，使距离尽可能远。 同时作者还增加了监督学习的设定，使用NLI任务进行fine-tune。

在实验结果上，在无监督设定和监督学习设定下，都在6 个STS 数据集上取得了最优效果，并且还在某些数据集上取得了非常大的提升。并且本文另一优势是fine-tune所需要的时间比较短， 单GPU几个小时就可以完成。


### SimCSE Simple Contrastive Learning of Sentence Embeddings 

https://arxiv.org/pdf/2104.08821.pdf

学习通用的对句子的编码在NLP任务中是非常重要的，在这篇文章中，作者在经典的语言模型比如 BERT 和 RoBERTa 中引入对比学习方法，很大程度的提升了对句子的编码表示。  具体的做法十分简单，把一个句子两次通过通过同一个 embedding 网络，但是两次embedding时的dropout值不同，这样产生的两个embedding就作为 positive pairs， 一个句子和其他的句子编码就是 negative pairs, 于是就可以进行标准的对比学习。  对于这种对比学习，可以理解成一种数据增强，这种数据增强方式大大增强了对于句子的表示。 作者也在文章中对比了其他的数据增强方式，比如删除，替换，mask词等，但是都是不如dropout都效果的。 同时作者也同时使用了 NLI 数据集中的数据作为监督学习的数据，把entailment 数据作为positive, Contradiction 的数据作为 negative。 更重的是，作者也从数学原理层面对于对比学习的有效性进行了一些解释，之前使用BERT的模型在句子编码上的效果不佳是因为词编码矩阵的奇异值只有几个非常大的值，其他的奇异值都接近0，通过数学推导也看出了对比学习有助于使得奇异值更加平坦。 

从实验效果看，在全部都7个语义相似度的数据集上都取得了最佳的效果，并且比之前的最佳都有着不小的进步，并且在7个迁移任务上也几乎都取得了最优的效果。  这篇文章和 ConSERT 的做法几乎是一样的，也几乎是同时期的工作，不过好像这篇文章暂时还没发表在会议上，但是更出名一点。

### Keep the Structure: A Latent Shift-Reduce Parser for Semantic Parsing 

IJCAI-2021

https://www.ijcai.org/proceedings/2021/0532.pdf

传统的端到端的semantic parsing模型都把自然语言看成是一个整体结构，然而自然语言本身也是可以进行结构划分的，并且划分后的结构可以和要预测的logic form进行一一对应。  所以在这篇文章中提出了一个使用 shift-reduce 解析器的方法，具体来说，文章中设定了一个 splitter可以把自然语言划分成不同的span, 之后对于每个span使用一个base parser解析出其结构，然后组合起来和ground truth进行对比。
方法的细节上，对于Base Parser，就是一个经典的seq2seq2结构，输入是span 的text 部分，经过一个 GRU 编码，又经过一个GPU 解码输出 sub logic form。 对于 Splitter， 作者把Splitter 的输出定义为针对栈和输入text的一系列action,包括shift操作，reduce操作，finish操作，通过这些操作每次找到一个span,就使用一个 span semantic vector 替换原有句子中的span部分，然后进行下一轮的操作。 最终所有的操作形成一个 action sequence, 作者称之为 trajectory（轨迹）。 

之后是训练方法，由于 Splitter 和 Base Parser 是两个相对独立的步骤，所以作者先进行了Trajectory Search 过程，尽可能搜索出大量的可能正确的 Trajectory， 然后使用 baseline parser对搜索出的 Trajectory 进行预测，对能成功匹配的部分直接作为pair, 不能匹配的部分直接作为一个比较大pair, 使用这些pairs对baseline parser进行训练，对于 Splitter， 把Trajectory视为 隐变量，使用maximum marginal likelihood (MML) estimation  进行训练。  整个系统有冷启动问题，所以一开始先使用全部的数据集对baseline parser进行预训练，防止训练完全偏离。

在实验结果上，在Geoquery dataset没有取得SOTA，但是比其他的所有不使用BERT的方法效果都好（本文也没有使用BERT）， 在更加复杂的 WQ dataset 数据集上取得了最佳的效果。   总体来看，本文通过引入 Splitter提升了 Semantic Parsing 的可解释性。


### MEDA Meta-Learning with Data Augmentation for Few-Shot Text Classification 

IJCAI-21 

https://www.ijcai.org/proceedings/2021/0541.pdf

元学习已经成了一个非常重要的来解决 few-shot learning 问题的手段，但是之前的方法基本上都在CV领域，很难直接迁移到 NLP 的任务上。 所以这篇文章提出了一个使用元学习配合数据增强的方式来解决NLP中的few-shot learning 的方法。 具体来说，本文的方法主要分成两个部分，第一个部分是 Ball Generator， 这个模块是在embedding 层面上增强了数据，因为原本的训练集数据只有 K-shot, 增强后可以大大增强学习效果。 具体的方法是把之前的空间看作是一个球，在中心和半径范围内通过生成算法随机生成数据。 第二个模块是Meta-learner ，这个模块是主要的学习部分，根据原本的K-shot数据和增强后的数据来学习，并且对query-set 中的数据进行预测，在这个模块，本文直接采用了Prototypical Networks  和Relation Networks 这样现有的网络解决方案。 最终loss由两个部分组成，第一个部分是 Ball Generator的loss，对于每个生成的embedding,都要和自己的中心尽可能接近，和其他的中心尽可能远。 第二个部分是 Meta-learner的，在query-set上的分类loss。 两个部分求和就是学习的目标。

在实验结果上，在 SNIPS 数据集和ARSC 数据集上都取得了SOTA的效果，并且对比实验也说明了数据增强方法和元学习的有效性。

### All NLP Tasks Are Generation Tasks: A General Pretraining Framework 

https://arxiv.org/pdf/2103.10360.pdf

目前的预训练模型有非常多种，包括autoregressive模型，autoencoding 模型，encoder-decoder 模型等等，但是并没有一类模型能在所有的NLPtask上（分类任务，条件生成任务，非条件生成任务）都能取得非常好的效果。 本文提出了GLM模型，希望能去解决这个问题，在多种类的任务上都能取得不错的效果，减少模型的选择问题。

具体来说，本文采用的预训练是Autoregressive Blank Infilling，是随机在文本中选择出 span, 然后用Autoregressive 的方式把这个span补全完整，这个任务结合了 autoencoding任务和 autoregressive任务，采用这个任务可以天然的在分类任务和生成任务都能取得还不错的效果。 不过之前也有SpanBERT做了类似的事情，本文的一个不同之处是本文找到了多个span, 然后把这些span做了打乱，然后放在原本序列后进行span补全任务。 这样生成的模型在做分类任务和生成任务都采用生成的模式来做， 在分类的任务的时候，会构造一个含有mask的句子，然后生成这个mask，根据生成的词汇来判断。 比如情分类似，会有一个 it is really [mask], 根据mask预测的词汇来进行判断（当然各种词汇的分类可能是另一个话题）。

在实验效果上，在NLU的GLUE数据集上基本能稳定比BERT好（同等参数量和数据量下）， 在 abstractive summarization 任务上取得了比UniLM 更好的效果，在Zero-shot language modeling 任务上取得了比GPT 模型更好的效果。


### Joint Verification and Reranking for Open Fact Checking Over Tables 

ACL 2021

https://aclanthology.org/2021.acl-long.529.pdf

结构信息对于fack checking 任务是一个非常重要的信息来源，然而之前的大多数工作工作都是仅仅采用了文本信息并且假设已经找到的正确的evidence。 在本文中，作者使用了结构化信息进行了 fack checking， 并且在是open domain 的设定下，即事先不知道对应的table，需要模型去寻找和学习。 

整个任务的定义是给一个问题和一个表格的集合，判断这个问题的真伪。 本文的方法首先找和问题最相关的K个table。 采用的方法其实就是非常经典的 TF_IDF 算法， 返回的 score最高的K个table。 在后续的分类中，需要把问题和table进行联合编码，首先再次把问题和table进行entity匹配，只找到最相关的3个column。 把选择table信息和问题拼接进入RoBERTa，取最后的 CLS-token 的编码值作为最后的编码结果， 为了能把各个表格的信息都能充分类用，后续还经过了一个cross attention层。  最终的分类层有两种策略，一种是 Joint reranking and verification， 即对每个编码经过分类层然后求和正则化作为最终logit, 然后求loss优化。 第二种是Ternary verification，即对每个table的编码单独进行验证，最终根据真伪的数量进行判断问题的真伪。

在实验结果上，作者首先验证了使用了TF-IDF算法的准确率，基本能有70-85的accuracy。 最终在作者的实验结果中，发现在open domain 设定下取得了结果已经可以和 close domain setting 下的去对比，仅仅比使用额外生成数据的模型效果差一点。  总结来看，本文应该是率先在open domain 的设定下做table-based fact check的，方向还相对比较初级，这个领域应该还是很多工作可以继续进行。

### Towards Table-to-Text Generation with Numerical Reasoning 

ACL 2021

https://aclanthology.org/2021.acl-long.115.pdf

近期的文本生成模型已经能够在生成对于结构化数据的描述文本上取得比较好的效果了。 但是目前仍然有一个非常大的挑战是产生一些需要数值推断的描述性问题。 针对这个问题，本文首先提出了一个新的数据集，数据集是由最近几年ACL会议的所有论文的表格和其描述组成的，由于论文中的描述一般都是由一些数值推断的，所以这个构造方式还比较合理。 同时作者还尝试了一些方法在这个新的数据集上的效果。

在数据集的构造的上，先用工具提取出了表格，然后筛选出有数字的描述，并且对所有的描述分成了三类，分别是数据描述，支持性描述和不相关的的描述，本文只选取了数据描述这个类别。
之后作者研究了如何对表格进行表示来更好的抽取推理信息，一个非常重要的步骤是对表格数据进行预计算， 把表格中的最大值，平均值，最小值等提前求出来，作为一个单独的 Pre-executed operation table, 然后把这个table和原始的table拼接起来线性化。

具体的方法上作者主要尝试了三种，第一种是templated-based method,从训练集上抽取模版，应用到测试集上， pointer generater 方法，使用 pointer generater 网络，不使用预训练模型。 第三种是使用预训练模型，同时设计了一种特殊的copy网络，主网络生成place-holder，比如<table_id>之类的，再由copy网络去选取具体的值。

在最终的实验结果上，  pointer generater 几乎不work, 使用预训练的GPT2模型取得了最好的效果，值得注意是copynet并没有带来提升。 总结来说，作者提出了这个数据集是有贡献意义的，但是并没有提出非常针对性的解决方案，说明这个数据集是有非常大的提升空间可以做的。

### Towards Robustness of Text-to-SQL Models against Synonym Substitution 

ACL 2021

https://aclanthology.org/2021.acl-long.195.pdf

虽然近期的很多 Text-to-SQL 工作取得了还不错的进展，模型的效果也在不断的提升，但是目前的目前非常依赖 schema-linking 这个机制，这就导致了系统十分不稳定并且容易受到攻击，当文本的表达被修改的时候，比如不使用 schema 中的词汇，那么 schema-linking 就会无效，在这宗情况下模型会受到多大的影响呢，本文就主要研究了这个问题。 本文针对Spider 数据集进行了二次标注，把自然语言表达中文本的大多数和数据库shcema 相同的本文都是用同义词进行替换，形成了一个新的Spider-Syn 数据集，并且针对这个新的数据集使用了两个相对比较简单的方法提升了模型的效果。

首先是数据标注的过程，使用了spider的训练集和验证集进行标注（因为测试集是不公开的），标注的过程并不是为了找到最优的攻击性case,而是尽量的模拟日常场景，因为在日常的用户的使用中就有可能会使用完全不同于数据库schema的词汇。针对不同的领域也给出了不同的同义词表方便标注使用。 

在方法上，主要使用了两个方法，第一个是Multi-Annotation Selection (MAS) ，使用标注的同义词汇同时进行训练。 第二个是Adversarial Training ，使用BERT-Attack 模型产生了 Adversarial Example， 使用这些example配合原有的数据集一起进行训练。

最终的实验效果上， 在Spider-Syn 数据集，目前的各个方法基本都下降了20个点以上， 作者使用的两个针对性增强方法分别也能提升10个点以上，可以说是非常的有效果了。 但离模型原本的效果也仍然有一定距离，说明这个方向仍然也有可以做的空间。

### LGESQL Line Graph Enhanced Text-to-SQL Model with Mixed Local and Non-Local Relations 

ACL 2021

https://aclanthology.org/2021.acl-long.198.pdf

把Text-to-SQL问题使用图网络建模已经有一些工作在做，但是这些工作基本都是使用以节点为中心的网络，并且使用不同的参数矩阵来对不同类型的边进行编码。 这样一方面就忽略了在边的拓扑结构中隐藏的语义信息，另一方面也不能对一个节点的 local 和 non-local的关系进行非常好的寻找。 在这篇工作提出了一个使用 line graph 来增强图建模的方法。 

方法的主要特点是提出了一个line graph, 和传统的 node graph相反， line graph 就是把经典图中的边看作节点，节点看作边。  这样形成的图其实就是之前的图的对偶图。 所以本文使用了两个几乎完全相同的网络来对line graph 和 node graph 进行建模。 最后再把对应的编码的拼接起来进行解码，解码的方式和rat-sql中的方法也完全相同，都是一个树形解码的方式。 同时本文还提出了一个gaph pruning 的辅助任务，用来识别正确的边，这也加强了网络对图的建模能力。

在实验效果上，作者在spider数据集上取得了 72%的效果，是目前最好的效果。 同时对比实验中也说明了line-graph引入带来的效果。

### Optimizing Deeper Transformers on Small Datasets 
ACL 2021

https://aclanthology.org/2021.acl-long.163.pdf

之前的工作大都普遍任务transformer需要大量的数据才能成功训练，在小的数据集上人们一般选择使用预训练好的模型进行微调。 但是这篇工作说明了Transformer通过更改一些初始化和优化的方式，可以在很小的数据集上也能训练的很好。 

之前已经有T-Fixup这篇文章做了相关的工作，但是T-fixup只针对了vanilla transformer，没有针对一些带有关系边的transformer,并且对于输入的初始化有严格的限制， 不能适配由预训练模型得到的初始化。 对此，本文做了一些改进， 首先本文通过一些数学理论分析了Transformer难以优化并且需要进行一些warmup  step的原因。 最后算法做的具体改进为1. 使用了Xavier初始化方法对于一些自由的参数。 2. 移除了warm-up 和所有后续transformer中的layer normalization(这被证明会影响优化的方差)。 3. 前向传播的时候控制每个向量中最大维度的大小 4. 通过上述的维度打来帮助相关 attention层进行参数的确定。 
最后作者在Text-to-SQL 数据集上进行了实验，使用优化后的RAT-SQL方法取得了70.9的效果，是当时的最佳效果。 并且在阅读理解任务上也取得了SOTA的效果。 


### From Paraphrasing to Semantic Parsing: Unsupervised Semantic Parsing via Synchronous Semantic Decoding

ACL2021

https://www.aclweb.org/anthology/2020.acl-main.608.pdf

本文提出了一个使用无监督的方式来做 semantic parsing的方法， 主要思想是，给一个输入的文本，该模型会同时生成范例文本和logic form,   并且在decode使用语法来限制了搜索的空间。  

本文的方法步骤大概是，首先选用一个预训练的 paraphrase generation model， 这个模型的主要作用是把文本变成规范化的文本， 但是目前预训练的模型其实做不到生成这种风格的文本。 所以本文首先写了一个从logic form 到规范化文本的一个标准解析器，产生一个数据集，可以使用这个数据集进行fine-tune。 然后使用fine-tune之后的模型生成文本的， 通过规则化解析也就生成了logic-form。
从实验结果上看，在OVERNIGHT 数据集上取得了SOTA的效果，在GEOGRANNO 和GEO 两个数据集上取得了远远比其他无监督方法好的效果。


### Span-based Semantic Parsing for Compositional Generalization

ACL 2021

https://aclanthology.org/2021.acl-long.74.pdf

使用 seq2seq 方法的 semantic parsing 被认为在组合泛化的效果上的比较差。 本文主要去解决这个问题。具体的， 本文使用了一个span-parser， 即使用使用句子中的每个span都去预测sub_tree(可能是实际值也可能是中间节点)。  这样就提高了模型学习文本和logic form之间关系的能力， 可解释性也就提升了， 理论上也就能提升组合泛化性。  文章中尝试了完全监督的模型（具体到span-level的标注）， 和隐式监督（只有针对整个句子的监督），这种监督需要EM算法来优化。  同时在inference的时候，使用了一个基于CKY的推理算法，能够保证生成语法正确的树。 

实验结果： 对 GEOQUERY, CLOSURE, SCAN 三个数据集进行了实验，在独立同分布的train-test 集划分设定下，和之前的seq2seq方法效果类似，但是在针对组合泛化性的 train-test 集划分设定下效果有了非常明显的提升。

### Compositional Generalization and Natural Language Variation: Can a Semantic Parsing Approach Handle Both? 

ACL 2021

https://aclanthology.org/2021.acl-long.75.pdf

本文对 semantic parsing 领域目前的主要挑战进行了分析，主要分成了两个方面，分别是自然语言的多样性和组合泛化性，目前的很多模型都很难同时做好这两个任务， 比如 seq2seq模型可以很好的利用神经网络的特性建模自然语言的多样性， 一些基于语法的模型可以相对比较好的处理组合泛化性但是对于一些特定的语言输入却无法产生输出， 本文就去探讨如何同时解决组合泛化性和自然语言多样性的问题。

本文提出了一个NQG-T5算法，其实就是简单的结合了NQG和T5两种算法，NQG是一种基于QCFG文法的parsing系统，这也是本文的主要贡献了，根据语法特征通过一些CRF模型和MML目标进行优化，但是这个模型对一些多样性的自然语言输入是无法处理的。 如果遇到无法处理的文本，就直接使用 T5 这个预训练模型进行seq2seq的输出。 

在实验结果上，在SCAN这个生成数据集上取得了最优效果，在GEOQUERY这个简单的数据集上取得了最优化效果，但是仍然在SPIDER数据集上效果不好。甚至NQG直接准确率是0。  总体来看，本文的工作还是相对比较简单， 没有把两个组件结合到一起，后续这个方向应该有非常大的空间。

### On the Sentence Embeddings from Pre-trained Language Models 

EMNLP 2020

https://aclanthology.org/2020.emnlp-main.733.pdf

尽管如BERT这样的预训练模型在NLP任务上取得了很大的成功，但是不经过 fine-tune 的BERT模型在判断句子相似性这个任务上做的并不算好。 这篇文章对这个问题进行了一些偏理论的分析，并且提出了改进方法成功提升了模型的性能。

作者首先对于BERT使用mask language model的预训练方式进行了一些理论上的分析，主要有了两个发现 1. 词出现的频率很大程度上的影响了编码空间。 2.  高频词分布的非常集中，低频词相对是比较分散的。   针对上面两个问题，作者提出了一个 flow-based方法，主要是通过一个可逆的变换把BERT变到一个标准高斯分布中，这样编码就从各向异性变成了各向同性。

在实验结果上，在SST的7个数据集上都取得了比较大的提升效果。 并且还详细分析了相关性和编辑距离的关系，说明了学到了有效的距离。

### All That’s ‘Human’ Is Not Gold: Evaluating Human Evaluation of Generated Text 

ACL 2021  Outstanding paper

https://aclanthology.org/2021.acl-long.565.pdf

在自然语言生成领域，人类评估一般被认为是标准， 但是随着模型生成的文本质量的提升，人类评估者是否能够非常好的对文本进行评估是需要进一步讨论的。 所以本文通过设计实验来评估人类是否能够区分 人类写作的文本和机器生成的文本， 在故事，新闻和食谱三个领域中，实验结果表明对于GPT2生成的文本识人类别准确率为57%， 对于GPT3生成的文本人类识别准确率为49%（甚至不如随机的50%）。 作者分析主要是随着模型性能的提升，生成文本的流利程度增加，错误往往发生在细节和逻辑层面，在生成的文本很长的情况下，非领域专家相对粗浅的阅读很难进行区分。 本文后续又通过三种方式对评估人员进行训练，分别是提供指导意见，给一些例子，给出对比，发现在给出例子的情况下准确率能到55%左右，虽然不是特别高，但是也已经好过了随机效果。   所以基于上述的实验结果，作者推荐后续的NLG领域在进行人工评估的时候最好给评估人员例子进行训练，并且对实验的设定进行更细节的报告来提升实验的可信程度。

### KILT a Benchmark for Knowledge Intensive Language Tasks

https://arxiv.org/pdf/2009.02252.pdf

目前的很多NLP任务都需要从外部的知识库中寻找一些知识，然后辅助任务的完成。 但是目前这类任务有非常多的数据集，每个任务都有不同的知识库，不同的假设，需要不同的数据加载器，评估方式和分析方式。  这样同一个模型在测试的时候就会带来不必要的计算开销，并且也很难测试知识在不同领域的迁移。

 所以本文提出了一个统一的知识库，KILT，使用了一个统一的知识库，就是wiki 百科的2019/08/01的版本，针对其他数据集中的不同版本，首先对问题的链接进行了重新匹配，对于文字修改的部分，取BLEU最大的部分进行匹配。  同时，本文针对Fack-checking, entity-linking,slot-filling,question answering 等任务的输入和输出都进行了适配，所有KILT框架可以适配于这些任务。 最终整个模型使用了一个简单的 seq2seq encoder的baseline,就能在上述各个任务上取得非常有竞争力的实验结果。
 
感觉这篇文章的主要贡献应该还是工程上的，统一了wiki百科的版本，给了一套统一的输入输出实验代码，测试起来更容易并且更有说服力了？

### WIKITABLET A Large-Scale Data-to-Text Dataset for Generating Wikipedia Article Sections 

https://arxiv-download.xixiaoyao.cn/pdf/2012.14919.pdf

2021 ACL findings

这篇文章主要在做 Data-to-Text 的数据集，之前的 Data-to-Text 数据集要么是多领域的单句生成，要么是单领域长篇生成。 这篇文章提出了一个非常大型的数据集，把wikipedia的文本和对应的表格数据和元数据进行了对应。 

总体来说，这个数据集有两大挑战， 一是在一些data-to-text 的case 上需要一些world knowledge, 有相关能力的模型可以在这里测试。  二是包括了多种多样的表格类型和数据领域。 

作者采用了一些比较简单的方法进行了测试，主要是使用了Transformer模型，并且辅助了一些优化方法，实验效果看起来不是很好，Transformer large模型甚至还不如Transformer base模型。  作者还进行了一些人工检查，结论是生成的文本流畅程度和质量不错，但是出现了一些一致性和事实性问题。  看文章中的说法这应该是非常大型的数据集，质量比较高，还有非常高的提升空间。


### Describing a Knowledge Base

ACL 2018

 https://arxiv.org/pdf/1809.01797.pdf

本文也是做 data-to-text 生成的问题。 本文主要有三个贡献， 一是提出了一个使用 slot-aware attention， table position self-attention  的pointer network 来做 data-to-text问题。 二是提出了一个新的评价指标KB reconstruction 。 三是提出了一个新的数据集。  下面一个一个的说。

方法方面，由于data-text 要求准确生成表格中内容，之前的seq2seq方法难做到，只使用 pointer network 又很难把slot-type和slot-value进行对齐，所以作者提出了slot-aware attention来解决这个问题。 同时，一些表格中的slot相互之间是有关系的，之前的模型在生成的时候可能不会考虑到，又增加了Table position attention 
来解决这个问题。

评价指标方面，提出了KB reconstruction指标，因为之前到BLEU指标很难全面的评价生成文本的质量。KB reconstruction的主要思想是根据生成的文本重新生成KB，和之前的KB逐项对比，生成生成一个准确率。  但是最重要的问题，根据文章的描述，这个生成KB好像是人工的！！

数据集方面，使用Wikipedia (2018/04/01) 和 Wikidata (2018/04/12) 在person 和 animal 两个领域进行了对齐，最后数据量是106,216 。 

实验结果上，BLEU最好也就能达到23，不算很高，倒是采用KB reconstruction能达到70%以上的F1 score。


### GenWiki A Dataset of 1.3 Million Content-Sharing Text and Graphs for Unsupervised Graph-to-Text Generation 

https://aclanthology.org/2020.coling-main.217.pdf

ICCL 2020

对于knowledge graph-text 领域的数据收集是十分困难的，所以当前的很多工作都在非常小的数据集上学习，并且该领域的无监督学习方法也开始活跃起来。 然而即使是采用无监督学习的方法，也有数据量不足的问题， 因为一个合格的无监督knowledge graph-text 数据需要满足四个条件： 1.  knowledge graph和text 的分布要尽可能相同 2. 文本要包含高质量的 entity 标注 3. 要比有监督数据集的规模大很多 4. 要有人工标注的测试集。 想要同时达到上面的条件还是非常困难的。 所以本文提出了GenWiki数据集，包含了1.3M的无监督数据对和1k的标注测试集。 

无监督数据的构造， 爬取了wikipedia的文本，并且对于网站上出现的所有包含超链接的实体查询相关的知识图谱， 之后进行一些过滤，去掉明显不相关的文本和知识图谱元素，然后设计了一些规则和算法对文本中的实体进行标注。 这样形成的文本和知识图谱对就构成了训练集。  对上述形成的无监督数据对，用户需要判断是否容易修改，如果容易修改，就对其进行修改然后形成正确的测试集。

作者尝试了多种无监督方法，最招的CycleGT 的BLEU值能达到40%以上， 效果已经还不错了，错误分析也是常识性错误比较多一点。


### WikiGraphs A Wikipedia Text Knowledge Graph Paired Dataset 

https://aclanthology.org/2021.textgraphs-1.7.pdf

这篇文章提出了一个Wipipedia 文章和知识图谱对齐的数据集， WikiGrpah, 可以方便条件文本生成，知识图谱生成，图表示学习等领域的研究。  WikiGrpah 数据集中的每条数据是wikipedia 文章和free-base中的子图对。  这个数据集的主要特点是文本比较长，知识图谱也比较大。 但是数据没有上一篇 GenWiki多，只有23522个训练数据，和48个验证集数据，43个测试集数据。 （验证集和测试集也太少了）

数据集的构造过程，先找到 WikiText-103中的文本，根据标题这样的关键信息去匹配Freebase中的实体，如果能匹配上，就进行下一个步骤，下一个步骤是去根据上一步匹配上的核心实体，在知识图谱上保留所有1-hop的子图。  最后一个步骤是过滤数据，对于相同类型边只选取一个典型。

在 data-text, graph-retrieval，text-retrieval 三个任务上都进行了实验，data-text 目前最好的效果能达到BLEU值30左右， text-retrieval  recall@5 能有35， graph-retrieval能达到100% （作者解释是这个任务比较简单）。

### Text-to-Table A New Way of Information Extraction 

https://arxiv.org/pdf/2109.02707.pdf

这篇文章提出了 text-to-table这个任务，可以认为是 table-text 任务的反向任务。 作者认为，这个任务和其他的信息抽取任务有两点不同，1是数据规模，可以从非常长的文本中抽取很大的表格。 2是这个抽取是完全数据驱动的，不需要显式的定义schema。 

因为schema不需要显式定义，table的结构约束并不多，所以本文采用了一个 Seq2Seq的框架来解决这个问题。 具体来说,baseline模型采用了BART这个预训练模型。  尽管table的结构限制并不多，但仅仅使用seq2seq模型仍并不能保证生成结构的准确性， 所以本文又额外增加了两个策略来缓解这个问题。

第一个策略是table constraint, 由于seq2seq模型不能保证生成的数据每行数量一样多，所以设计了这个算法首先记下第一行的长度，之后每行decode产生这个长度时就自动开始decode下一行。  第二个策略是table relation embedding,   由于table数据本身不同cell之间是存在关系的，比如一个cell 和其row header和column header都有相关性，所以在生成每个cell的时候都增加了和其相关cell的attention。  具体来说，采用思路就是Self-Attention with Relative Position Representations 这篇文章中的Relation-aware Self-Attention（但这篇文章没引用）， 即在 transformer中增加了关系编码，如果两个编码之间之间本身有关系（cell 和 header关系），在计算attention的时候会增加一个关系向量。

在实验结果方面，在Rotowire, E2E, Wikitabletext ，WikiBio 数据集上进行了实验，比较的方法基本是使用RE抽取关系，然后再构成表格。 评价指标就是和标准表对比，如果一个cell和值，column header, row header都一致，就算正确，然后计算  Pre, Rec, F1。 从结果上看，本文使用的改进Seq2Seq方法在其中三个数据集取得了最好的F1值。 但比Vanilla Seq2Seq方法的提升并不多。 

### BART Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension 

https://aclanthology.org/2020.acl-main.703.pdf

ACL 2020

从目前来看已经是一个非常出名的预训练模型了，之前一直没有对这篇文章进行总结。 

BART 是一个使用去噪自动编码器进行预训练的seq2seq模型。 在BART之前已经有很多的预训练模型采用了 MASK-LANGUAGE 方法来进行预训练，但是这些模型着重的END TASK都过于局限？  所以这篇文章提出了BART，和BERT不同的是，BART是一个seq2seq模型，同时使用了transformer的encoder和decoder，这也使得bart对一些生成式的下游效果更好。

具体来说，bart的预训练方式，首先对文本进行打乱，打乱的方式有以下几种，token masking(和Bert中相同)， Token Deletion （把一些词删除）， Text Infilling 
（根据泊松分布采样出span长度，mask这些span）, Sentence Permutation （对句子进行完全打乱）， Document Rotation（对文档进行旋转）， 把打乱后的文本放入encoder, 把正常顺序的文本作为decoder的输入，来构建出完整的文本。 

最终在 Sequence Classification Tasks ， Token Classification Tasks ， Sequence Generation Tasks ， Machine Translation 上都稳定取得了最好效果。

### Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer 

https://arxiv.org/pdf/1910.10683.pdf

Journal of Machine Learning Research 21 (2020) 

著名的T5模型，T5是Text-to-Text Transfer Transformer的缩写。 主要的思想是把所有的nlp任务都建模成了一个 text-to-text 任务，使用了一个encoder-decoder的transformer架构来学习几乎所有任务， 取得了不错的效果。 本文的特点是进行了非常大量的实验。

