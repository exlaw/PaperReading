## text-to-table 相关的论文

- [text-to-table 相关的论文](#text-to-table------)
  * [Text-to-Table A New Way of Information Extraction](#text-to-table-a-new-way-of-information-extraction)
  * [Bidirectional Transition-Based Dependency Parsing](#bidirectional-transition-based-dependency-parsing)
  * [Asynchronous Bidirectional Decoding for Neural Machine Translation](#asynchronous-bidirectional-decoding-for-neural-machine-translation)
  * [Agreement-Based Joint Training for Bidirectional Attention-Based Neural Machine Translation](#agreement-based-joint-training-for-bidirectional-attention-based-neural-machine-translation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


### Text-to-Table A New Way of Information Extraction 

https://arxiv.org/pdf/2109.02707.pdf

这篇文章提出了 text-to-table这个任务，可以认为是 table-text 任务的反向任务。 作者认为，这个任务和其他的信息抽取任务有两点不同，1是数据规模，可以从非常长的文本中抽取很大的表格。 2是这个抽取是完全数据驱动的，不需要显式的定义schema。 

因为schema不需要显式定义，table的结构约束并不多，所以本文采用了一个 Seq2Seq的框架来解决这个问题。 具体来说,baseline模型采用了BART这个预训练模型。  尽管table的结构限制并不多，但仅仅使用seq2seq模型仍并不能保证生成结构的准确性， 所以本文又额外增加了两个策略来缓解这个问题。

第一个策略是table constraint, 由于seq2seq模型不能保证生成的数据每行数量一样多，所以设计了这个算法首先记下第一行的长度，之后每行decode产生这个长度时就自动开始decode下一行。  第二个策略是table relation embedding,   由于table数据本身不同cell之间是存在关系的，比如一个cell 和其row header和column header都有相关性，所以在生成每个cell的时候都增加了和其相关cell的attention。  具体来说，采用思路就是Self-Attention with Relative Position Representations 这篇文章中的Relation-aware Self-Attention（但这篇文章没引用）， 即在 transformer中增加了关系编码，如果两个编码之间之间本身有关系（cell 和 header关系），在计算attention的时候会增加一个关系向量。

在实验结果方面，在Rotowire, E2E, Wikitabletext ，WikiBio 数据集上进行了实验，比较的方法基本是使用RE抽取关系，然后再构成表格。 评价指标就是和标准表对比，如果一个cell和值，column header, row header都一致，就算正确，然后计算  Pre, Rec, F1。 从结果上看，本文使用的改进Seq2Seq方法在其中三个数据集取得了最好的F1值。 但比Vanilla Seq2Seq方法的提升并不多。 


### Bidirectional Transition-Based Dependency Parsing 

https://ojs.aaai.org//index.php/AAAI/article/view/4733

AAAI-2019

Transition-Based Dependency Parsing:   把parsing tree变成一系列的action(transition),   也就是这个任务需要学习文本到action sequence的映射，其实和seq2seq任务很像了。 action一般有三种动作，SHIFT： 把输入队列中的元素移动到栈中。 LEFT（reduce）： 栈顶的两个元素是左子树关系。 RIGHT(reduce)： 栈顶的两个元素是右子树关系。

之前的很多 Transition-Based Dependency Parsing都是从左到右进行处理，这样容易造成误差累积，这篇文章同时学习了从左到右生成和从右到左生成两个parser, 然后设计了三种不同的decode算法来解决这个问题。

Vanilla Joint Scoring:  最简单的decode方法，先分别用两个parser完整的生成，然后分别使用两个paser对生成的结果进行打分，选择打分相加最高的。

Joint Decoding with Dual Decomposition:  其实仍然是一种基于 score的思想，优化目标是两个方向的parser 生成的 tree得分加起来最高, 限制条件是两个 tree 相同， 这是一个有限制条件的优化目标，作者使用拉格朗日松弛找到一个可以优化的上界（我能简单理解拉格朗日松弛，但是不理解他的公式怎么产生的）。  然后针对这个优化目标设计了一个decode的算法，就是Joint Decoding with Dual Decomposition， 主要思想是迭代法，每轮迭代两个paser会产生两个矩阵（和tree一样）， 如果两个矩阵不同就把不同的部分单独拿出来称为矩阵u，在后续迭代的时候输入矩阵u，使得两个paser得到的解尽可能相似。 因为由于之前说的这个算法能优化上界，所以也能尽可能的获得一个比较优的解。

Joint Decoding Guided by Dynamic Oracle:  这个decoding方法用到了Dynamic Oracle，这是一种在测试阶段提供一个gold树来防止误差累积的方法（Training Deterministic Parsers with Non-Deterministic Oracles）。 本文的这个decoding算法就用到了 Dynamic Oracle， 仍然是迭代的思想，两个paser在上一轮生成的树互为gold tree输入到下一轮，使用 Dynamic Oracle来指导两个paser生成相同的 tree。

最终实验结果上，在很多数据集都有提升，但提升也都不到一个点的样子。


### Asynchronous Bidirectional Decoding for Neural Machine Translation 

https://arxiv.org/pdf/1801.05122.pdf

AAAI 2018

尽管seq2seq方法采用的attention机制目前取得了还不错的效果，但是仍然存在的问题是不能利用 reverse context, 即right-to-left 方向的 context,  这样就会有 left-to-right方向的误差累积问题。

于是本文，在训练的时候采用的模型包括一个encder和两个decoder，两个decoder分别是 backward decoder（从right-left 方向decode）和 forward decoder（从left-right)方向进行decode, 具体的流程是，经过encoder之后，先经过一个 backward decoder, 生成反向的表示，然后经过 forward decoder, 此时计算attention会同时计算 encoder中的状态和backward decoder中的状态，这样就考虑了到了reverse context， 一定程度上的避免了只有left-to-right方向的误差累积问题。

在几个数据集上都能取得一个点以上的提升，效果还是不错的。

### Agreement-Based Joint Training for Bidirectional Attention-Based Neural Machine Translation

https://www.ijcai.org/Proceedings/16/Papers/392.pdf

IJCAI 2016

本文的主要目标是去优化在 seqseq 生成时 attention 矩阵的质量， 具体来说，之前的seq2seq模型在捕捉attention时可能仅仅会捕捉到特定方面的，甚至可能会有噪声出现（不正确的attention）。  于是本文设计了一种优化 attention 矩阵的质量 的方法， 主要方法是让 source-target 和 target-to-source 两个模型真对相同训练数据的attention矩阵算一个agreement 值，把这个agreement值也作为一个优化目标，最终使得两个模型的attention矩阵尽可能相似，最终优化出一个不错的attention矩阵。

本文设计了三种算法来去计算 aggrement 值， 分别是Square of addition (SOA)， Square of subtraction (SOS)， Multiplication (MUL) 。

实验结果上，提升的幅度还是不算小，说明了结果的有效性。


