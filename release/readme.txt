###################
#### BioDiscML ####
###################

# Description
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

#### Program usage #### 
# Config file
Before executing BioDiscML, a config file must be created. Use the template to 
create your own. Everything is detailled in the config.conf file. Examples are 
available in the Test_datasets at: 
https://github.com/mickaelleclercq/BioDiscML/tree/master/release/Test_datasets

# train a new model
java -jar bruteforceML.jar -config config.conf -train
    example: java -jar bruteforceML.jar -config config_example.conf -train
    You can stop it at any moment and choose best model(s)

# Choose best model(s)
java -jar bruteforceML.jar -config config.conf -bestmodel
    When training completed, stopped or in execution, best model selection can 
    be executed. This command reads the results file.
    Best models are selected based on configuration provided in config file.
    You can also choose your own models manually, by opening the results file
    in an excel-like program and order models by your favorite metric.
    Each model has an identifier (modelID) you can provide to the command. Example:
java -jar bruteforceML.jar -config config.conf -bestmodel modelID_1 modelID_2

# Output files
Note: {project_name} is set in the config.conf file

{project_name}_a.*  
    A csv file and a copy in arff format (weka input format) are created here. 
    They contain the merged data of input files with some adaptations.

{project_name}_b.*
    A csv file and a copy in arff format (weka input format) are also created here. 
    They are produced after feature ranking and are already a subset of 
    {project_name}_a.*
    Feature ranking is performed by Information gain for categorial class. Features 
    having infogain <0.0001 are discarded.
    For numerical class, RELIEFF is used. Only best 1000 features are kept, or having
    a score greater than 0.0001.

{project_name}_c.*results.csv
    Results file. Summary of all trained model with their evaluation metrics and
    selected attributes. Use the bestmodel command to extract models.
    Column index of selected attributes column correspond to the 
    {project_name}_b.*csv file    

{project_name}_d.{model_name}_{model_hyperparameters}_{feature_search_mode}.*details.txt
    Detailled information about the model, includes correlations with other features
    feature_search_mode are either 
        - Forward Stepwise Selection (F)
        - Backward stepwise selection (B)
        - Forward stepwise selection and Backward stepwise elimination (FB)
        - Backward stepwise selection and Forward stepwise elimination (BF)
        - “top k” features.

{project_name}_.{model_name}_{model_hyperparameters}_{feature_search_mode}.*features.csv
    features retained by the model in csv
    If a test set have been generated or provided, a file will be generated for:
        - the train set (*.train_features.csv)
        - both train and test sets (*all_features.csv)

{project_name}_.{model_name}_{model_hyperparameters}_{feature_search_mode}.*corrFeatures.csv
    features retained by the model with their correlated features in csv
    If a test set have been generated or provided, a file will be generated for:
            - the train set (*.train_corrFeatures.csv)
            - both train and test sets (*all_corrfeatures.csv)

{project_name}_.{model_name}_{model_hyperparameters}_{feature_search_mode}.*roc.png
    boostrap roc curves (EXPERIMENTAL)
    If a test set have been generated or provided, a roc curves picture will be generated
    for both train and test sets.

{project_name}_.{model_name}_{model_hyperparameters}_{feature_search_mode}.*model
    serialized model compatible with weka


