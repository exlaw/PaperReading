### Reinforcement Learning to Jointly Encode Prompts and Database Schemas for Text-to-SQL Semantic Parsing

讲用强化学习解决Text-to-SQL,没看出亮点。

Generating Structured Query Language (SQL) queries over a database from a text prompt constitutes a semantic parsing task that strongly relies on a system's ability to compare and jointly encode the prompt and database schema. In this work, we undertake a preliminary investigation into utilizing Reinforcement Learning (RL) to learn a procedure to read and jointly represent natural language and SQL database schemas, contrasting with prior RL work that doesn't explicitly learn an encoding procedure. We begin with a thorough literature review of the past several years of text-to-SQL research which identifies a key issue in current state-of-the-art models—namely, the inability for highly compositional & relational information (from the prompt & schema respectively) to be encoded into a fixed sized vector while also synthesizing this information in a meaningful way for successful decoding of an SQL query. Our solution to this is an iterative one, quite similar to a human writer's process for writing SQL: an agent performs a series of actions that scan from the prompt or database while also being able to decode tokens from the current representation. This approach results in the encoding at any given state being relevant primarily to the following tokens, and enables the agent to "query" for more information as it decodes the sentence further. Though our experiments are still in an early stage, we demonstrate that Deep Q Network (DQN) agents appear to learn simple procedures for jointly encoding text and schemas even with minimal training.

### Semi-Supervised Self-Training for Text-to-SQL via Column-Specificity-Based Meta-Learning


使用元学习来做单表text-to-sql self-training。 看起来有可能是有价值的。

Single-table text-to-SQL is a semantic parsing task that aims to translate natural language questions into Structure Query Language over one table. Most of the existing work only focuses on the performance of models trained by sufficient labeled data but ignores that there are usually few labeled data in actual scenarios. Meanwhile, the iteration of the business will result in the challenge of few-shot tables that do not follow the distribution of the training data. In this paper, we propose a Self-Training text-to-SQL model (ST-SQL) to solve these problems. We design a semi-supervised learning framework for the basic column-forced model to make effective use of unlabeled data. Furthermore, we present a column-specificity-based meta-learning algorithm to assist the framework in handling few-shot tables. In the experiments, Our method achieves state-of-the-art accuracy both on open-domain benchmark WikiSQL and domain-specific benchmark ESQL. The results of few-shot and zero-shot experiments also show the strong ability of our method in both scenarios which obtain a significant improvement compared with existing work.





