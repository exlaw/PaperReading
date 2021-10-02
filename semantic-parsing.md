## semantic parsing 相关的论文

- [semantic parsing 相关的论文](#semantic-parsing------)
  * [Awakening Latent Grounding from Pretrained Language Models for Semantic Parsing](#awakening-latent-grounding-from-pretrained-language-models-for-semantic-parsing)
  * [Value-Agnostic Conversational Semantic Parsing](#value-agnostic-conversational-semantic-parsing)
  * [SMBOP Semi-autoregressive Bottom-up Semantic Parsing](#smbop-semi-autoregressive-bottom-up-semantic-parsing)
  * [Keep the Structure: A Latent Shift-Reduce Parser for Semantic Parsing](#keep-the-structure--a-latent-shift-reduce-parser-for-semantic-parsing)
  * [Towards Robustness of Text-to-SQL Models against Synonym Substitution](#towards-robustness-of-text-to-sql-models-against-synonym-substitution)
  * [LGESQL Line Graph Enhanced Text-to-SQL Model with Mixed Local and Non-Local Relations](#lgesql-line-graph-enhanced-text-to-sql-model-with-mixed-local-and-non-local-relations)
  * [From Paraphrasing to Semantic Parsing: Unsupervised Semantic Parsing via Synchronous Semantic Decoding](#from-paraphrasing-to-semantic-parsing--unsupervised-semantic-parsing-via-synchronous-semantic-decoding)
  * [Span-based Semantic Parsing for Compositional Generalization](#span-based-semantic-parsing-for-compositional-generalization)
  * [Compositional Generalization and Natural Language Variation: Can a Semantic Parsing Approach Handle Both?](#compositional-generalization-and-natural-language-variation--can-a-semantic-parsing-approach-handle-both-)
  * [Data Augmentation with Hierarchical SQL-to-Question Generation for Cross-domain Text-to-SQL Parsing](#data-augmentation-with-hierarchical-sql-to-question-generation-for-cross-domain-text-to-sql-parsing)
  * [Exploring Underexplored Limitations of Cross-Domain Text-to-SQL Generalization](#exploring-underexplored-limitations-of-cross-domain-text-to-sql-generalization)
  * [Natural SQL: Making SQL Easier to Infer from Natural Language Specifications](#natural-sql--making-sql-easier-to-infer-from-natural-language-specifications)
  * [GRAPPA GRAMMAR-AUGMENTED PRE-TRAINING FOR TABLE SEMANTIC PARSING](#grappa-grammar-augmented-pre-training-for-table-semantic-parsing)
  * [Learning Contextual Representations for Semantic Parsing with Generation-Augmented Pre-Training](#learning-contextual-representations-for-semantic-parsing-with-generation-augmented-pre-training)
  * [Zero-Shot Text-to-SQL Learning with Auxiliary Task](#zero-shot-text-to-sql-learning-with-auxiliary-task)
  * [Leveraging Table Content for Zero-shot Text-to-SQL with Meta-Learning](#leveraging-table-content-for-zero-shot-text-to-sql-with-meta-learning)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


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


### SMBOP Semi-autoregressive Bottom-up Semantic Parsing 

NAACL 2021 

https://arxiv.org/pdf/2010.12412.pdf

目前几乎大多数的 Text-to-SQL 方法采用的都是 top-down 的decoding 方法，也就是一种auto-regressive 的方法，从根节点开始，由上至下，由左至右逐渐decode成整个语法树。  在这个工作中，尝试了一种不同的decode方式，也就是一种 bottom up 的decode方式，具体含义就是在第t步去构造一个高度不超过t的 top-K 个子树。 这样做的优势在于，可以并行化的生成子树，使得算法的复杂度由线性变成了对数复杂度。 根据作者的实际数据，在SPIDER 数据集上的训练加速可以达到5倍，推断加速达到2.2倍。

在具体的方法上，作者重新对SQL的语法进行改写，形成了一套特殊的关系代数，和自然语言的形式相对更接近一点，更重要的是增加了keep操作，使得所有的SQL树都是平衡树（子树高度相同）。  后续的操作就是每一层的筛选，第一层是数据库的schema 和数据库中的值，对每种可能对匹配都去计算score,包括对一个子树的操作（unary）和对两个子树的操作（binary）， 形成所有候选后，根据score值选取 K 个进入下一层的计算，并且过程中还使用了 cross attention的方法去更新了子树的 向量表示。  loss值的计算是对标准子树的最大似然。

在实验结果上，本文在 SPIDER 数据集上的实验结果和RAT-SQL 类似， 只有 0.3%的点的损失，考虑到采用了Semi-autoregressive的情况下，已经是不错的结果了。 同时本文也分析了在 senmantic parsing的场景下，bottom-up方法和 top-down方法在理论上的差距不是很大。


### Keep the Structure: A Latent Shift-Reduce Parser for Semantic Parsing 

IJCAI-2021

https://www.ijcai.org/proceedings/2021/0532.pdf

传统的端到端的semantic parsing模型都把自然语言看成是一个整体结构，然而自然语言本身也是可以进行结构划分的，并且划分后的结构可以和要预测的logic form进行一一对应。  所以在这篇文章中提出了一个使用 shift-reduce 解析器的方法，具体来说，文章中设定了一个 splitter可以把自然语言划分成不同的span, 之后对于每个span使用一个base parser解析出其结构，然后组合起来和ground truth进行对比。
方法的细节上，对于Base Parser，就是一个经典的seq2seq2结构，输入是span 的text 部分，经过一个 GRU 编码，又经过一个GPU 解码输出 sub logic form。 对于 Splitter， 作者把Splitter 的输出定义为针对栈和输入text的一系列action,包括shift操作，reduce操作，finish操作，通过这些操作每次找到一个span,就使用一个 span semantic vector 替换原有句子中的span部分，然后进行下一轮的操作。 最终所有的操作形成一个 action sequence, 作者称之为 trajectory（轨迹）。 

之后是训练方法，由于 Splitter 和 Base Parser 是两个相对独立的步骤，所以作者先进行了Trajectory Search 过程，尽可能搜索出大量的可能正确的 Trajectory， 然后使用 baseline parser对搜索出的 Trajectory 进行预测，对能成功匹配的部分直接作为pair, 不能匹配的部分直接作为一个比较大pair, 使用这些pairs对baseline parser进行训练，对于 Splitter， 把Trajectory视为 隐变量，使用maximum marginal likelihood (MML) estimation  进行训练。  整个系统有冷启动问题，所以一开始先使用全部的数据集对baseline parser进行预训练，防止训练完全偏离。

在实验结果上，在Geoquery dataset没有取得SOTA，但是比其他的所有不使用BERT的方法效果都好（本文也没有使用BERT）， 在更加复杂的 WQ dataset 数据集上取得了最佳的效果。   总体来看，本文通过引入 Splitter提升了 Semantic Parsing 的可解释性。


### Towards Robustness of Text-to-SQL Models against Synonym Substitution 

ACL 2021

https://aclanthology.org/2021.acl-long.195.pdf

github 仓库信息 

https://github.com/ygan/Spider-Syn

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



### Data Augmentation with Hierarchical SQL-to-Question Generation for Cross-domain Text-to-SQL Parsing 

EMNLP 2021， 但是目前阅读的版本应该和最终版本是有差距的，目前应该读不到最终版本。

https://arxiv.org/abs/2103.02227

cross domain text-to-sql目前主要面临了两大问题，1是测试数据库在训练时无法见到，这样就需要模型有很强的泛化性。 2是数据量比较小，并且标注数据比较困难。 

本文采用数据增强的思路来解决这个问题，即直接从数据库中生成数据，并且采用了一种SQL-to-question 的方法获取 sql-text pair。 具体来说，根据SQL的语法采样的语法模版，应用到各个数据库中，就产生了大量的SQL语句，然后采用了一个 Hierarchical Generation 的方法，把SQL和文本的对应部分划分出来，分别训练seq2seq模型，最终再把生成的数据拼接起来形成数据对。 

实验方面，在WikiSQL， Spider ， DuSQL 三个数据上的 seen 和 unseen两个设定上进行了实验，发现了都是有提升的。  比较有意思的实验是三种训练策略，使用 pre-train方法竟然是提升最小的。


### Exploring Underexplored Limitations of Cross-Domain Text-to-SQL Generalization 

EMNLP 2021 短文

https://arxiv.org/pdf/2109.05157.pdf

本文属于一篇独辟蹊径的角度，虽然spider数据集确实是声明模型不需要领域知识的引入就能取得比较好的效果，但是本文却总结了5类需要领域知识的例子。 主要原因在于测试集和验证集的domain是不同的，一些隐含的领域知识可能需要对这个领域的学习才能得到。 于是作者在验证集中挑选出了一些需要domain kownledge 的数据作为新的spider-DK 验证集。

最终在实验结果上也表明，在新的验证集上，模型的表现并不好，这也就说明了目前的模型并不能够很好的建模领域知识。 同时一个实验数据也非常有意思，模型倾向于把预测order的顺序反向， 这样主要是由于训练数据label不均匀，所以后续解决这个问题可能也会成为一个方向。


### Natural SQL: Making SQL Easier to Infer from Natural Language Specifications 

EMNLP 2021 findings

https://arxiv.org/pdf/2109.05153.pdf

本文提出了一种更好的SQL 表示， 降低了自然语言和SQL语言之间的GAP，让模型可以更好的进行训练和推断。  

本文提出的表示主要有三个特点： 1.  消除了GROUP BY， HAVING， FROM， JOIN ON这些语句，仅仅保留了 SELECT，WHERE，ORDER BY 2. 消除了 SET， UNION， EXCEPT. 等语句， 并且消除了嵌套语句。3  减少了需要的 schema 数量，使得schema-linking 更加简单。

同时由于本文的value只存在于where语句中，可以限制按照顺序生成，所以这种表示方法容易生成比较高执行准确率的模型。 最终效果，在执行准确率和exact match 都有提升，尤其在执行准确率上取得了SOTA效果。

### GRAPPA GRAMMAR-AUGMENTED PRE-TRAINING FOR TABLE SEMANTIC PARSING

https://arxiv.org/pdf/2009.13845.pdf

ICLR 2021

相对看过比较早一篇文章，再来总结一下。

主要动机，想要进一步的去fine-tune当前的预训练模型， 因为经典的预训练模型在处理文本和表格数据时还是存在一定的gap的。

主要做法，先使用一个 SCFG 从数据库中采样出SQL语句（作者也承认这样采样出的SQL语句其实很粗糙，可能需要进一步的筛选）， 然后构建了SQL和文本之间的对齐模版，生成SQL后可以直接生成对应的文本。 在生成了大量的数据后，采用了两个任务对预训练模型进行进一步的fine-tune, 分别是 MLM （经典的训练任务）， SSP objective（预测一个列是否出现在文本中，并且是什么操作中出现的）。

在WIKISQL ， WIKITABLEQUESTIONS ，SPIDER数据集上都取得了很好的效果。

本文后续提出的几个可以详细思考的点，1. Pre-training objectives ， 同时MLM和SSP比两个单独使用效果要好很多。2. Generalization， 尽管是text-to-sql任务上训练的，但是却可以在其他的一些semantic parsing 任务上取得不错的效果。 3. Pre-training time and data： 他们的实验表明仅仅需要相对比较小的数据进行fine-tune就可以，需要的epoch也比较小，这样可以让BERT获得新的能力同时保留其encoding能力。  这里还有一点是，GRAPPA是用相对规则文本进行训练的，这样容易影响模型的encoding性能，但是如果使用预训练模型生成的数据可能会好一点？4.  Pre-training vs. training data augmentation  本文的实验表明采用预训练的方式是更好的选择。

### Learning Contextual Representations for Semantic Parsing with Generation-Augmented Pre-Training 

AAAI 2021

https://ojs.aaai.org/index.php/AAAI/article/view/17627

这也是一个很早就读过的文章了，再重新总结一下～

目前大多数的预训练模型都是适合在通用场景下，但是在应用到text-to-sql模型时会遇到几个问题： 分别是 1. fail to detect column mentions in the utterances 2. fail to infer column mentions from cell values 3. fail to compose complex SQL queries.    所以本文提出了一个框架，能够生成增强数据，然后再进行预训练。 

首先是预训练部分，本文一共有4个预训练任务，分别是 1. Column Prediction (CPred) ： 预测一个列是否在文本中出现 2.  Column Recovery (CRec):  从cell value 推断 column。（之前有一点低估了这个任务，现在看来还要重视一下）   3. SQL Generation (GenSQL):  直接生成SQL语句  4.  Masked Language Model(MLM)： 和经典预训练训练的方法一样。 

数据生成： SQL和表格都是爬虫获得的，SQL-to-Text 直接采用了 BART 模型，没有采用什么预训练的机制。  table-to-text:  使用了一些 control code 配合table生成了text。

实验结果表明， Column Recovery (CRec) 的效果竟然还是挺好的。

### Zero-Shot Text-to-SQL Learning with Auxiliary Task 

https://ojs.aaai.org/index.php/AAAI/article/view/6246

AAAI 2020

github地址 https://github.com/JD-AI-Research-Silicon-Valley/auxiliary-task-for-text-to-sql

本文研究 如何在 zero-shot 下进行 text-to-sql 工作，  具体来说，是在 wiki-SQL 数据集下，按照作者的说法， 在wikiSQL 数据集默认的划分中，测试集中70%的数据库在训练集中是见过的，这样不太符合设定。 （其实spider数据集也已经是 zero-shot setting 了，但本文没在这个数据集下实验）。

具体来说，首先使用了两个LSTM网络分别对问题和column进行编码， 然后使用了一个BiAttn 机制，对column和问题进行更好的互编码。  在decode的时候针对 select, agg 和 where分别设计了不同的解码模块。 同时设计了一个辅助任务来更好的对齐 column 和 问题， 辅助任务具体是先通过 sequence labeling 识别句子中和列对应的地方，然后使用一个 pointer网络上进行学习。

实验效果上，在WikiSQL 标准数据集下取得了3%的进步， zero-shot setting 的数据集下有 5%的提升。

### Leveraging Table Content for Zero-shot Text-to-SQL with Meta-Learning 

https://www.aaai.org/AAAI21Papers/AAAI-6324.ChenY.pdf

AAAI 2021

Github地址:   https://github.com/qjay612/meta_learning_NL2SQL

本文同样是去解决 zero-shot text-to-sql 的问题，没有使用其他的数据，仅仅使用wikiSQL 中的数据，引入了数据库内容并且使用 meta-learning 的方法在 zero-shot setting 下取得了很好的效果。

具体来说，在基础模型上，也是先给了一个skeleton，然后把整个任务分成了6个子任务 （Select-Column(SC),	 Select-Aggregation(SA), 	Where-Number(WN), Where-Column(WC), Where-Operator(WO)， Where-Value(WV)。   在引入数据库内容上，由于有的数据库内容会太多，不可能全部进行编码，所以本文首先了进行了筛选，筛选出了一些比相关的数据库内容。 在编码方式上，把问题和数据库column一同放到BERT中进行编码，然后把数据内容通过char embedding进行编码， 然后Ec 和 Eq 分别输入到6个不同的子任务模块中。  元学习方法就有点类似经典的方法了，不再多介绍了。

最终在多个wikiSQL 和 ESQL 数据集的 full, few-shot 和 zero-shot setting 下都取得了最优的效果。


