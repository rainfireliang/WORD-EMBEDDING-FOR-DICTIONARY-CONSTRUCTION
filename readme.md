#These are the replication materials for 
Liang, H., Ng, Y. M. M., & Tsang, N. L. T. (2023). Word embedding enrichment for dictionary construction: An example of incivility in Cantonese. Computational Communication Research.

##Code files:

"1.Code_Table1.py": the Python script to replicate Table 1. It includes the code for conventional machine learning models in predicting incivility.
"1.Code_Table1_Appendix.html": the Python script to replicate Tables A/B/C 1 in the appendix.

"2.Code_Table2_Doc2Vec.py": the Python script to replicate part of Table 2. It includes the code for conventional machine learning models with Word2Vec. 
"2.Code_Table2_Doc2Vec_Appendix.html": the Python script to replicate Tables A/B/C 2 in the appendix.

"3.Code_Table2_DeepNN.py": the Python script to replicate the rest of Table 2. It includes the deep learning models with the pre-trained Word2Vec as the weights in an embedding layer.
"3.Code_Table2_DeepNN_Appendix.html": the Python script to replicate Tables A/B/C 2 in the appendix.

"4.Code_Figure2B&C.py": the Python script to create the dataset for producing Figures 2B and 2C.
"4.RCode_Fig2ABC.R": the R script to plot Figure 2.

"5.RCode_PerceptionAnalysis.R": the R script to analyze the data from the survey experiment.

##Datasets:

"train.csv": the training dataset for Tables 1&2.
"train_crowd.csv": the training dataset for Tables in the appendix (using perceptions as the ground truth)
"Figure3.xlsx": the dataset to produce Figure 3.

"word2vec_hk_2022.model": the Word2Vec model trained with the forum comments. (a large dataset to be downloaded here: https://www.dropbox.com/s/3do1mejt84gs1rb/word2vec_hk_2022.model?dl=0)
"namecalling_byRound.rds": name-calling words in each round of the iteration.
"vulgarity_byRound.rds": vulgar words in each round of the iteration.
"perception_survey.rds": the dataset from the survey experiment.

"stopCantonese.txt": a list of stop words in Cantonese.
"word_dic_2022.txt": some customized words for Chinese tokenization.

"uncivil_words.txt": the incivility dictionary created in this article.
