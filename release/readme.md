
# BioDiscML

Large-scale automatic feature selection for biomarker discovery in high-dimensional 
OMICs data

## Short description
Automates the execution of many machine learning algorithms across various
optimization and evaluation procedures to identify the best model and signature

## Description
The identification of biomarker signatures in omics molecular profiling is an 
important challenge to predict outcomes in precision medicine context, such as 
patient disease susceptibility, diagnosis, prognosis and treatment response. To 
identify these signatures we present BioDiscML (Biomarker Discovery by Machine 
Learning), a tool that automates the analysis of complex biological datasets 
using machine learning methods. From a collection of samples and their associated 
characteristics, i.e. the biomarkers (e.g. gene expression, protein levels, 
clinico-pathological data), the goal of BioDiscML is to produce a minimal subset 
of biomarkers and a model that will predict efficiently a specified outcome. To 
this purpose, BioDiscML uses a large variety of machine learning algorithms to 
select the best combination of biomarkers for predicting  either categorical or 
continuous outcome from highly unbalanced datasets. Finally, BioDiscML also 
retrieves correlated biomarkers not included in the final model to better 
understand the signature. The software has been implemented to automate all 
machine learning steps, including data pre-processing, feature selection, model 
selection, and performance evaluation. 
https://github.com/mickaelleclercq/BioDiscML/

Full manuscript: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6532608/

## Requirements
JAVA 8 (https://www.java.com/en/download/)

## Program usage
BioDiscML can be started either by creating a config file or using command line

### By config file
Before executing BioDiscML, a config file must be created. Use the template to 
create your own. Everything is detailed in the config.conf file. Examples are 
available in the Test_datasets at: 
https://github.com/mickaelleclercq/BioDiscML/tree/master/release/Test_datasets

#### Train a new model
```Bash
java -jar biodiscml.jar -config config.conf -train
```
config_myData.conf (text file) quick start content example (See release/Test_datasets folder)
This configuration takes as input a file (myData.csv) and name it myProjectName.
A sampling (default 2/3 for training and 1/3 for testing) is performed before 
classification procedure to predict the myOutcome class. One best model will 
be selected based on Repated Holdout performance MCC on the train set. 
config.conf file example: 
```
project=myProjectName
trainFile=myData.csv
sampling=true
doClassification=true
classificationClassName=myOutcome
numberOfBestModels=1
numberOfBestModelsSortingMetric=TRAIN_TEST_RH_MCC
```

#### Choose best model(s)
```Bash
java -jar biodiscml.jar -config config.conf -bestmodel 
```
When training completed, stopped or still in execution, best model 
selection can be executed. This command reads the results file. Best 
models are selected based on a strategy provided in config file. You 
can also choose your own models manually, by opening the results file 
in an excel-like program and order models by your favorite metrics or 
filters. Each model has an identifier (modelID) you can provide to 
the command. Example:
```Bash
java -jar biodiscml.jar -config config.conf -bestmodel modelID_1 modelID_2
```

#### Predict new data
```Bash
java -jar biodiscml.jar -config config.conf -predict 
```
Once the best model obtained, you can predict new data or test a blind test set 
put aside by yourself before training. The file should be of same format and structure 
as the training input files. This file must contain at least all features of the 
selected best model signature. Features present in the newData file but absent from 
the signature of the model will simply be ignored during the prediction. If a 
class to predict column is present, BioDiscML will return errors statistics.
config.conf file example: 
```
project=myProjectName
newDataFile=myNewData.csv
doClassification=true
classificationClassName=class
modelFile=myBestModel.model
```

### By command line
The same parameters from config file can be used to be inputed in a command line.
Example: 
```Bash
time java -jar biodiscml.jar -train -project=myProject -excluded=excludedColumn 
-doClassification=true -classificationClassName=class -trainFile=data.csv 
-debug=true -bootstrapFolds=10 -loocv=false -cpus=10 -computeBestModel=false 
-classificationFastWay=true -ccmd=bayes.AveragedNDependenceEstimators.A1DE -F 1 -M 1.0 -W
```
Note that the option -ccmd must stay at the end of the command line when classifier
parameters follows it. 


## Output files
Note: {project_name} is set in the config.conf file

- {project_name}_a.*  

A csv file and a copy in arff format (weka input format) are created here. 
They contain the merged data of input files with some adaptations.

- {project_name}_b.*

A csv file and a copy in arff format (weka input format) are also created here. 
They are produced after feature ranking and are already a subset of 
{project_name}_a.*. Feature ranking is performed by Information gain 
for categorial class. Features having infogain <0.0001 are discarded.
For numerical class, RELIEFF is used. Only best 1000 features are kept, 
or having a score greater than 0.0001.

- {project_name}_c.*results.csv

Results file. Summary of all trained model with their evaluation metrics 
and selected attributes. Use the bestmodel command to extract models.
Column index of selected attributes column correspond to the 
{project_name}_b.*csv file. 
For each model, we perform various evaluations summarized in this table:

| Header | Description |
| ------------ | ------------ |
| ID | Model unique identifier. Can be passed as argument for best model selection |
| Classifier | Machine learning classifier name |
| Options | Classifier hyperparameters options |
| OptimizedValue | Optimized criterion used for feature selection procedure |
| SearchMode | Type of feature selection procedure: <br>- Forward Stepwise Selection (F)<br>- Backward stepwise selection (B)<br>- Forward stepwise selection and Backward stepwise elimination (FB)<br>- Backward stepwise selection and Forward stepwise elimination (BF)<br>- "top k" features.|
| nbrOfFeatures | Number of features in the signature |
| TRAIN_10CV_ACC | 10 fold cross validation Accuracy on train set |
| TRAIN_10CV_AUC | 10 fold cross validation Area Under The Curve on train set|
| TRAIN_10CV_AUPRC |10 fold cross validation Area Under Precision Recall Curve on train set |
| TRAIN_10CV_SEN | 10 fold cross validation Sensitivity on train set|
| TRAIN_10CV_SPE |10 fold cross validation Specificity on train set |
| TRAIN_10CV_MCC |10 fold cross validation Matthews Correlation Coefficient on train set |
| TRAIN_10CV_MAE |10 fold cross validation Mean Absolute Error on train set |
| TRAIN_10CV_BER | 10 fold cross validation Balanced Error Rate on train set|
| TRAIN_10CV_FPR |10 fold cross validation False Positive Rate on train set |
| TRAIN_10CV_FNR |10 fold cross validation False Negative Rate on train set |
| TRAIN_10CV_PPV | 10 fold cross validation Positive Predictive value on train set|
| TRAIN_10CV_FDR | 10 fold cross validation False Discovery Rate on train set|
| TRAIN_10CV_Fscore | 10 fold cross validation F-score on train set|
| TRAIN_10CV_kappa | 10 fold cross validation Kappa on train set|
| TRAIN_matrix | 10 fold cross validation Matrix on train set |
| TRAIN_LOOCV_ACC | Leave-One-Out Cross Validation Accuracy on Train set |
| TRAIN_LOOCV_AUC | Leave-One-Out Cross Validation Area Under The Curve on Train set|
| TRAIN_LOOCV_AUPRC | Leave-One-Out Cross Validation Area Under Precision Recall Curve on Train set|
| TRAIN_LOOCV_SEN | Leave-One-Out Cross Validation Sensitivity on Train set|
| TRAIN_LOOCV_SPE | Leave-One-Out Cross Validation Specificity on Train set |
| TRAIN_LOOCV_MCC | Leave-One-Out Cross Validation Matthews Correlation Coefficient on Train set|
| TRAIN_LOOCV_MAE | Leave-One-Out Cross Validation Mean Absolute Error on Train set|
| TRAIN_LOOCV_BER | Leave-One-Out Cross Validation Balanced Error Rate on Train set|
| TRAIN_RH_ACC | Repeated holdout Accuracy on Train set |
| TRAIN_RH_AUC | Repeated holdout Area Under The Curve on Train set |
| TRAIN_RH_AUPRC | Repeated holdout Area Under Precision Recall Curve on Train set |
| TRAIN_RH_SEN | Repeated holdout Sensitivity on Train set |
| TRAIN_RH_SPE | Repeated holdout Specificity on Train set |
| TRAIN_RH_MCC | Repeated holdout Matthews Correlation Coefficient on Train set |
| TRAIN_RH_MAE | Repeated holdout Mean Absolute Error on Train set |
| TRAIN_RH_BER | Repeated holdout Balanced Error Rate on Train set |
| TRAIN_BS_ACC | Bootstrap Accuracy on Train set |
| TRAIN_BS_AUC | Bootstrap Area Under The Curve on Train set |
| TRAIN_BS_AUPRC | Bootstrap Area Under Precision Recall Curve on Train set |
| TRAIN_BS_SEN | Bootstrap Sensitivity on Train set |
| TRAIN_BS_SPE | Bootstrap Specificity on Train set |
| TRAIN_BS_MCC | Bootstrap Matthews Correlation Coefficient on Train set |
| TRAIN_BS_MAE | Bootstrap Mean Absolute Error on Train set |
| TRAIN_BS_BER | Bootstrap Balanced Error Rate on Train set |
| TRAIN_BS.632+ | Bootstrap .632+ rule |
| TEST_ACC | Evaluation Accuracy on test set|
| TEST_AUC | Evaluation Area Under The Curve on test set|
| TEST_AUPRC |Evaluation Area Under Precision Recall Curve on test set |
| TEST_SEN |Evaluation Sensitivity on test set |
| TEST_SPE | Evaluation Specificity on test set|
| TEST_MCC | Evaluation Matthews Correlation Coefficient on test set|
| TEST_MAE | Evaluation Mean Absolute Error on test set|
| TEST_BER | Evaluation Balanced Error Rate on test set|
| TRAIN_TEST_RH_ACC |Repeated holdout Accuracy on merged Train and Test sets |
| TRAIN_TEST_RH_AUC | Repeated holdout  Area Under The Curve on merged Train and Test sets|
| TRAIN_TEST_RH_AUPRC | Repeated holdout Area Under Precision Recall Curve on merged Train and Test sets|
| TRAIN_TEST_RH_SEN | Repeated holdout Sensitivity on merged Train and Test sets|
| TRAIN_TEST_RH_SPE | Repeated holdout Specificity on merged Train and Test sets|
| TRAIN_TEST_RH_MCC | Repeated holdout Matthews Correlation Coefficient on merged Train and Test sets|
| TRAIN_TEST_RH_MAE | Repeated holdout Mean Absolute Error on merged Train and Test sets|
| TRAIN_TEST_RH_BER | Repeated holdout Balanced Error Rate on merged Train and Test sets|
| TRAIN_TEST_BS_ACC |Bootstrap Accuracy on merged Train and Test sets |
| TRAIN_TEST_BS_AUC | Bootstrap  Area Under The Curve on merged Train and Test sets|
| TRAIN_TEST_BS_AUPRC | Bootstrap Area Under Precision Recall Curve on merged Train and Test sets|
| TRAIN_TEST_BS_SEN | Bootstrap Sensitivity on merged Train and Test sets|
| TRAIN_TEST_BS_SPE | Bootstrap Specificity on merged Train and Test sets|
| TRAIN_TEST_BS_MCC | Bootstrap Matthews Correlation Coefficient on merged Train and Test sets|
| TRAIN_TEST_BS_MAE | Bootstrap Mean Absolute Error on merged Train and Test sets|
| TRAIN_TEST_BS_BER | Bootstrap Balanced Error Rate on merged Train and Test sets|
| TRAIN_TEST_BS_BER_BS.632+ | Bootstrap .632+ rule on merged Train and Test sets|
| AVG_BER | Average of all calculated Balanced Error Rates |
| STD_BER | Standard deviation of the calculated Balanced Error Rates|
| AVG_MAE | Average of all calculated Mean Absolute Errors |
| STD_MAE |Standard deviation of the calculated Mean Absolute Errors |
| AVG_MCC | Average of all calculated Matthews Correlation Coefficients |
| STD_MCC |Standard deviation of the calculated Matthews Correlation Coefficients |
| AttributeList | Selected features. Use the option -bestmodel to generate a report and get the features' full names|

Note that all columns refering to a test set will be empty if no test set have been generated or provided

- {project_name}_d.{model_name}_{model_hyperparameters}_{feature_search_mode}.*details.txt

Detailled information about the model and its performance, with the full signature and 
correlated features.


- {project_name}_.{model_name}_{model_hyperparameters}_{feature_search_mode}.*features.csv

Features retained by the model in csv.
If a test set have been generated or provided, a file will be generated for:
-- the train set (*.train_features.csv)
-- both train and test sets (*all_features.csv)


- {project_name}_.{model_name}_{model_hyperparameters}_{feature_search_mode}.*corrFeatures.csv

Features retained by the model with their correlated features in csv
If a test set have been generated or provided, a file will be generated for:
-- the train set (*.train_corrFeatures.csv)
-- both train and test sets (*all_corrfeatures.csv)

- {project_name}_.{model_name}_{model_hyperparameters}_{feature_search_mode}.*roc.png

Boostrap roc curves (EXPERIMENTAL)
Must be enabled in configuration file. 
If a test set have been generated or provided, a roc curve picture will be generated
for both train and test sets.


- {project_name}_.{model_name}_{model_hyperparameters}_{feature_search_mode}.*model

Serialized model compatible with weka

