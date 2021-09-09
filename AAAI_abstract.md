# AAAI 投稿中关于 Text-to-SQL 任务做的点

暂时先整理了 Text-to-SQL 领域， semantic parsing 领域之后再去做。

- [AAAI 投稿中关于 Text-to-SQL 任务做的点](#aaai-------text-to-sql------)
    + [Reinforcement Learning to Jointly Encode Prompts and Database Schemas for Text-to-SQL Semantic Parsing](#reinforcement-learning-to-jointly-encode-prompts-and-database-schemas-for-text-to-sql-semantic-parsing)
    + [Semi-Supervised Self-Training for Text-to-SQL via Column-Specificity-Based Meta-Learning](#semi-supervised-self-training-for-text-to-sql-via-column-specificity-based-meta-learning)
    + [Harnessing Pre-Trained Language Model with Structure-Aware Ability for Text-to-SQL](#harnessing-pre-trained-language-model-with-structure-aware-ability-for-text-to-sql)
    + [Improving the Database Generalizability of the Text-to-SQL Task](#improving-the-database-generalizability-of-the-text-to-sql-task)
    + [AUTOSQL: Question Auto-Completion for Text-to-SQL](#autosql--question-auto-completion-for-text-to-sql)
    + [SeaD: End-to-end Text-to-SQL Generation with Schema-aware Denoising](#sead--end-to-end-text-to-sql-generation-with-schema-aware-denoising)
    + [Schema Dependency-Enhanced Curriculum Pre-Training for Table Semantic Parsing](#schema-dependency-enhanced-curriculum-pre-training-for-table-semantic-parsing)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


### Reinforcement Learning to Jointly Encode Prompts and Database Schemas for Text-to-SQL Semantic Parsing

讲用强化学习解决Text-to-SQL,没看出亮点。

Generating Structured Query Language (SQL) queries over a database from a text prompt constitutes a semantic parsing task that strongly relies on a system's ability to compare and jointly encode the prompt and database schema. In this work, we undertake a preliminary investigation into utilizing Reinforcement Learning (RL) to learn a procedure to read and jointly represent natural language and SQL database schemas, contrasting with prior RL work that doesn't explicitly learn an encoding procedure. We begin with a thorough literature review of the past several years of text-to-SQL research which identifies a key issue in current state-of-the-art models—namely, the inability for highly compositional & relational information (from the prompt & schema respectively) to be encoded into a fixed sized vector while also synthesizing this information in a meaningful way for successful decoding of an SQL query. Our solution to this is an iterative one, quite similar to a human writer's process for writing SQL: an agent performs a series of actions that scan from the prompt or database while also being able to decode tokens from the current representation. This approach results in the encoding at any given state being relevant primarily to the following tokens, and enables the agent to "query" for more information as it decodes the sentence further. Though our experiments are still in an early stage, we demonstrate that Deep Q Network (DQN) agents appear to learn simple procedures for jointly encoding text and schemas even with minimal training.

### Semi-Supervised Self-Training for Text-to-SQL via Column-Specificity-Based Meta-Learning


使用元学习来做单表text-to-sql self-training。 看起来有可能是有价值的。

Single-table text-to-SQL is a semantic parsing task that aims to translate natural language questions into Structure Query Language over one table. Most of the existing work only focuses on the performance of models trained by sufficient labeled data but ignores that there are usually few labeled data in actual scenarios. Meanwhile, the iteration of the business will result in the challenge of few-shot tables that do not follow the distribution of the training data. In this paper, we propose a Self-Training text-to-SQL model (ST-SQL) to solve these problems. We design a semi-supervised learning framework for the basic column-forced model to make effective use of unlabeled data. Furthermore, we present a column-specificity-based meta-learning algorithm to assist the framework in handling few-shot tables. In the experiments, Our method achieves state-of-the-art accuracy both on open-domain benchmark WikiSQL and domain-specific benchmark ESQL. The results of few-shot and zero-shot experiments also show the strong ability of our method in both scenarios which obtain a significant improvement compared with existing work.


### Harnessing Pre-Trained Language Model with Structure-Aware Ability for Text-to-SQL

这篇文章中提到的 additional denoising task that further improves the structure-aware ability through reconstructing the SQL query 十分有意思，可以猜测一下大概使用了什么方法。

The goal of Text-to-SQL is to generate an executable SQL given a natural language question and relational database as input. Current popular Text-to-SQL parsers heavily rely on complex model architecture design to encode the database structure, which makes them time-consuming then less applicable. To remedy this problem, we propose a general seq2seq framework via harnessing pre-trained language model with structure-aware ability, which is more light-weight with fewer parameters and shorter training/inference time, and adaptive to various types of Text-to-SQL tasks (e.g, single-turn/multi-turn, single-table/multi-table, multi-lingual). Specifically, we construct the input consisting of (1) basic elements: database (column, table) and question; (2) structure description: linking of database; (3) linking tags: alignment among elements. Furthermore, we develop an additional denoising task that further improves the structure-aware ability through reconstructing the SQL query. With a few highly intuitive designed descriptions, we obtain substantial improvements and achieve state-of-the-art or comparable results to previous best-performing systems across lots of popular Text-to-SQL benchmarks.

### Improving the Database Generalizability of the Text-to-SQL Task

这篇文章提到的点非常不错，解决schema-linking的泛化性问题，如果做的快说不定可以出一个同期工作。

The research on text-to-SQL semantic parsing usually applies end-to-end machine translation-based techniques for generating SQL queries from natural language questions. An important mechanism added to the machine translation technique is the schema linking which serves as an entity linking module between natural language inquiry and database schema objects, e.g., table names, column names, and existing values. However, schema linking is becoming the main bottleneck for the generalizability of the text-to-SQL model. Since it can detect the exact match phrase of the schema object primarily. Furthermore, it cannot detect out-of-database values, especially numerical values, frequently appear in the input questions. This paper presents a novel approach for in-crease generalizability by building a schema linking model using the BERT contextual embedding, NumER numerical entity detection, and multilayer perceptron. These techniques help the model learn to perform the schema linking task by understanding the context and numerical category information without requiring a heuristic approach. Our proposed method is surpassing the traditional method’s accuracy by 15%

### AUTOSQL: Question Auto-Completion for Text-to-SQL

挺有意思的一个话题，帮助用户更好的根据数据库中的内容生成查询文本。

Many database users do not precisely know the structure or the content of databases, and thus their queries do not exactly reflect their information needs. To help users formulate their questions when they query databases, we propose AU-TOSQL, a question auto-completion framework in Text-to-SQL for natural language interfaces to databases. We construct a benchmark dataset of question prefixes and corresponding SQLs for possible complete questions based on five existing Text-to-SQL datasets including Advising, ATIS, GeoQuery, Scholar, and Spider, and propose new metrics to evaluate how much user effort can be saved by our models. We experiment with both generation models and retrieval models on our constructed dataset to provide candidates from prefixes. Experiment results demonstrate that our dataset is challenging for current models and we incorporate curriculum learning to improve our model’s performance.

### SeaD: End-to-end Text-to-SQL Generation with Schema-aware Denoising

这篇文章也是提出了一个denoising task, 感觉对于这方面设计任务可以好好去研究一下。

In text-to-SQL task, seq-to-seq models often lead to sub-optimal performance due to limitations in their architecture. In this paper, we present a simple yet effective approach that adapts transformer-based seq-to-seq model to robust text-to-SQL generation. Instead of inducing constraint to decoder or reformat the task as slot-filling, we propose to train seq-to-seq model with Schema aware Denoising (SeaD), which consists of two denoising objectives that train model to either recover input or predict output from two novel erosion and shuffle noises. These denoising objectives acts as the auxiliary tasks for better modeling the structural data in S2S generation. In addition, we improve and propose a clause-sensitive execution guided (EG) decoding strategy to overcome the limitation of EG decoding for generative model. The experiments show that the proposed method improves the performance of seq-to-seq model in both schema linking and grammar correctness and establishes new state-of-the-art on WikiSQL benchmark. The results indicate that the capacity of vanilla seq-to-seq architecture for text-to-SQL may have been under-estimated.


### Schema Dependency-Enhanced Curriculum Pre-Training for Table Semantic Parsing

也是提出了几个预训练的任务，这也提示了要从预训练的

Recently pre-training models have significantly improved the performance of table semantic parsing. However, existing pre-training approaches have not carefully explored interaction relationships between a question and the corresponding database schema, which is a key ingredient for uncovering their semantic and structural correspondence. In this paper, we design a schema dependency pre-training objective to impose the desired inductive bias into the learned representations for table pre-training. We further propose a schema-aware curriculum learning approach to alleviate the impact of noise and learn effectively from the pre-training data in an easy-to-hard manner. We evaluate our pre-trained framework by fine-tuning it on two benchmarks, WikiSQL and Spider. The results demonstrate the effectiveness of our pre-training objective and curriculum in comparison to a variety of baselines.




