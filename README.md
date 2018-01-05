# clickstream-datamining
AI Project 5 Report

Question 1: Clickstream Data Mining

Algorithm: ID3 Decision Tree Learning

Training Files: train.csv, train_label.csv

Step1: Calculating the entropy of the target. 
	   The target here is the train_label.csv

       Entropy of the target is:
       E(X) = SUM_x -p(x)log_2(x)
 
Step2: The dataset has 274 features. Therefore, the entropy of each feature w.r.t the target is calculated. 

Entropy of a feature w.r.t. the target is: 
        E(T,X) = SUM_c P(c)E(c)
 
Step3: Calculate the information gain of a feature.
        Gain(T,X) = Entropy(T) - Entropy(T,X)

Step4: Select the feature with the highest information gain. That feature is the best feature. 

Step5: Choose the attribute with the best feature and divide the dataset by its children and repeat the process for each child.
        In the question, we’re getting 89 as the first best feature.
        89 has 5 children. Thus, we’ll divide the dataset into 5 tables, one for each child. 

Step6: Check the entropy of each child. The children with entropy 0, will be a leaf node. Other children will need 
        further splitting.     

Step7: Run the ID3 algorithm recursively on the data tables of each subsequent children. 

The algorithm is a depth first search algorithm and expands nodes to a certain depth. That depth is calculated by 
the Stopping Criterion.

The Stopping Criterion given in the question is Chi-squared criterion. 
Lets assume that the feature we want to split on is irrelevant. We expect its +ve and –ve examples to be distributed 
according to chi-squared distribution. 
Suppose that splitting on feature T, will produce sets {T_i} where i = 1 to m.

Let p, n denote the number of positive and negative examples that we have in our dataset. 
    p_i' = p*(|T_i|/N)
    
    n_i' = n*(|T_i|/N)

are be the expected number of positives and negatives in each partition, if the attribute is irrelevant to the class. 
Then the statistic of interest is:
    S = SUM_i (((p_i'-p_i)^2)/p_i') + ((n_i'-n_i)^2)/n_i')

p-value is that probability of observing a value X at least as extreme as S coming from the distribution. To find 
that, we compute P(X >= S). The test is passed if the p-value is smaller than some threshold.

The thresholds given in the question are 0.05, 0.01, 1.

Threshold 1 is used when we are not using chi-square distribution.

Results: 

Threshold 1: 
Tree Prediction Accuracy: 0.73672
Output File Prediction Accuracy: 0.72112 = 72.11%

Threshold 0.01:
Tree Prediction Accuracy: 0.7482
Output File Prediction Accuracy: 0.72112 = 74.82%
