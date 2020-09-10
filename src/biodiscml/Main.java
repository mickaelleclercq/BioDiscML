/*
 * Run all routine to execute the training
 */
package biodiscml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

/**
 *
 * @author Mickael
 */
public class Main {

    public static boolean debug = false;
    public static boolean isClassification = true;

    public static String wd = "";
    public static String project = "myProject";
    public static String CVfolder = "tmpCV"; //cross validation folder. Disabled

    //program functions
    public static String configFile = ""; //config file
    public static boolean needConfigFile = true;
    public static boolean training = false;
    public static boolean predictNewData = false;
    public static boolean trainingBestModel = false;
    public static HashMap<String, String> hmTrainingBestModelList = new HashMap<>();//modelID, identifier prefix

    public static String modelFile = "";
    public static HashMap<String, String> hmExcludedFeatures = new HashMap<>();//features to exclude from the final dataset

    //config
    public static String mergingID = "Instance";

    //source files
    public static HashMap<String, String> hmTrainFiles = new HashMap<>();//filename, identifier prefix
    public static HashMap<String, String> hmNewDataFiles = new HashMap<>();//filename, identifier prefix

    //options
    public static Boolean doClassification = false;
    public static String classificationClassName = "class";
    public static String regressionClassName = "class";
    public static String separator = "";
    public static Boolean doRegression = false;
    public static Boolean classificationFastWay = false;
    public static Integer numberOfBestModels = 1;
    public static ArrayList<String> classificationFastWayCommands = new ArrayList<>(); //classifier, optimizer
    public static ArrayList<String> classificationBruteForceCommands = new ArrayList<>(); //classifier, optimizer
    public static String classificationOptimizers = "auc, mcc, fdr, ber, acc";
    public static String searchmodes = "f,fb,b,bf,top1,top5,top10,top15,top20,top30,top40,top50,top75,top100";
    public static Boolean metaCostSensitiveClassifier = false;
    public static Boolean regressionFastWay = false;
    public static String regressionOptimizers = "CC, RMSE";
    public static Boolean metaAdditiveRegression = false;
    public static ArrayList<String> regressionFastWayCommands = new ArrayList<>(); //classifier, optimizer
    public static ArrayList<String> regressionBruteForceCommands = new ArrayList<>(); //classifier, optimizer
    public static double pAUC_lower = 0;
    public static double pAUC_upper = 0.3;
    public static double spearmanCorrelation_lower = -0.99;
    public static double spearmanCorrelation_upper = 0.99;
    public static double pearsonCorrelation_lower = -0.99;
    public static double pearsonCorrelation_upper = 0.99;
    public static String bestModelsSortingMetric = "AVG_MCC";
    public static double bestModelsSortingMetricThreshold = 0.1;
    public static Integer maxNumberOfFeaturesInModel = 250;
    public static int maxNumberOfSelectedFeatures = 1000;
    public static boolean doSampling = true;
    public static int samplingFold = 3; //separate the set in x parts, keep 1 for test, others for training
    public static int bootstrapAndRepeatedHoldoutFolds = 100; // Also used for repeated holdout
    public static String cpus = "max";
    public static boolean combineModels = false;
    public static boolean retrieveCorrelatedGenes = true;

    public static String combinationRule = "AVG";
    public static double maxRankingScoreDifference = 0.005; //for correlated gene retreiving
    public static boolean loocv = true;
    public static boolean computeBestModel = true;
    //benchmark
    public static String bench_AUC = "";

    //TODO
    static boolean retreiveCorrelatedGenesByRankingScore = true; //avoid for non-binary classes and regression
    static boolean ROCcurves = false; //experimental
    static boolean UpSetR = false; //experimental

    public static void main(String[] args) throws IOException {
        System.out.println("#### BioDiscML ####\n");
        //read configuration file
        System.out.println("#### Parsing options...");
        setOptionsFromCommandLine(args); //from command line

        if (!configFile.isEmpty() && needConfigFile) {
            setConfiguration();
        }

        //set models
        if (!classificationFastWay && !regressionFastWay) {
            try {
                String line = "";
                File classifiers = new File("classifiers.conf");
                BufferedReader br;
                if (!classifiers.exists()) {
                    try {
                        br = new BufferedReader(new InputStreamReader(Main.class.getResourceAsStream("/classifiers.conf")));
                    } catch (Exception e) {
                        br = new BufferedReader(new FileReader("/classifiers.conf"));
                    }
                } else {
                    br = new BufferedReader(new FileReader(classifiers));
                }
                line = "";
                while (br.ready()) {
                    if (!line.startsWith("#") && !line.trim().isEmpty()) {
                        String option = line.split("=")[0].trim();
                        String value = line.split("=")[1].trim();
                        switch (option) {
                            case "ccmd":
                                classificationBruteForceCommands.add(value.trim());
                                break;
                            case "rcmd":
                                regressionBruteForceCommands.add(value.trim());
                                break;
                        }
                    }
                    line = br.readLine();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("Model search mode: Fast way mode");
        }

        //set number of max cpus to use
        if (!cpus.equals("max")) {
            System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", cpus);
        }

        // Go to training
        if (training) {
            System.out.println("#### Start training...");
            //CLASSIFICATION
            if (doClassification) {
                isClassification = true;
                //put data together in the same file for ML
                System.out.println("## Preprocessing of the input file(s)");
                String CLASSIFICATION_FILE = wd + project + "a.classification.data_to_train.csv"; //output of AdaptDatasetToWeka()
                AdaptDatasetToTraining c = new AdaptDatasetToTraining(CLASSIFICATION_FILE);

                //execute feature selection and training
                System.out.println("## Feature selection and training");
                String FEATURE_SELECTION_FILE = wd + project + "b.featureSelection.infoGain.csv"; // output of Training(), feature selection result
                String TRAINING_RESULTS_FILE = wd + project + "c.classification.results.csv"; // output of Training(), models performances
                Training m = new Training(CLASSIFICATION_FILE, TRAINING_RESULTS_FILE, FEATURE_SELECTION_FILE, "class");

                //choose best model
                if (computeBestModel) {
                    System.out.println("## Best model selection");
                    BestModelSelectionAndReport b = new BestModelSelectionAndReport(CLASSIFICATION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
                            "classification");
                }
            }

            //REGRESSION
            if (doRegression) {
                isClassification = false;
                //put data together in the same file for ML
                System.out.println("## Preprocessing of the input file(s)");
                String REGRESSION_FILE = wd + project + "a.regression.data_to_train.csv";
                AdaptDatasetToTraining c = new AdaptDatasetToTraining(REGRESSION_FILE);

                //execute training
                System.out.println("## Feature selection and training");
                String FEATURE_SELECTION_FILE = wd + project + "b.featureSelection.RELIEFF.csv";//filled by feature selection algo type
                String TRAINING_RESULTS_FILE = wd + project + "c.regression.results.csv";
                Training m = new Training(REGRESSION_FILE, TRAINING_RESULTS_FILE, FEATURE_SELECTION_FILE, "reg");

                //choose best model
                if (computeBestModel) {
                    System.out.println("## Best model selection");
                    BestModelSelectionAndReport b = new BestModelSelectionAndReport(REGRESSION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
                            "regression");
                }
            }

            if (!doClassification && !doRegression) {
                System.err.println("[error] No prediction type has been set (classification or regression)."
                        + " Set doClassification or doRegression at true");
                System.exit(0);
            }
        }

        if (predictNewData) {
            System.out.println("#### Start predicting new data...");
            if (modelFile.isEmpty()) {
                System.err.println("[error] No model file have been provided (Set a modelFile in config file)");
                System.exit(0);
            }
            if (hmNewDataFiles.isEmpty()) {
                System.err.println("[error] No new data file have been provided (Set a newDataFile in config file)");
                System.exit(0);
            }

            //put data together in the same file for ML
            String NEWDATA_FILE = wd + project + ".data_to_predict.csv"; //output of AdaptDatasetToWeka()
            AdaptDatasetToTesting c = null;
            if (doClassification) {
                c = new AdaptDatasetToTesting(classificationClassName, hmNewDataFiles,
                        NEWDATA_FILE, separator, wd + modelFile);
            } else {
                c = new AdaptDatasetToTesting(regressionClassName, hmNewDataFiles,
                        NEWDATA_FILE, separator, wd + modelFile);
            }

            //execute feature selection and training
            String PREDICTIONS_RESULTS_FILE = wd + project + modelFile + "_" + ".prediction.results.txt"; // output of Testing
            TestingAndEvaluate t = new TestingAndEvaluate();
            t.TestingAndEvaluate(wd + modelFile, NEWDATA_FILE, PREDICTIONS_RESULTS_FILE, c.isMissingClass());
        }

        if (trainingBestModel) {
            System.out.println("#### Start best model selection...");
            if (doClassification) {
                String CLASSIFICATION_FILE = wd + project + "a.classification.data_to_train.csv";
                String TRAINING_RESULTS_FILE = wd + project + "c.classification.results.csv"; // output of Training(), models performances
                String FEATURE_SELECTION_FILE = wd + project + "b.featureSelection.infoGain.csv"; // output of Training(), feature selection result
                BestModelSelectionAndReport b = new BestModelSelectionAndReport(CLASSIFICATION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
                        "classification");

            } else {
                String REGRESSION_FILE = wd + project + "a.regression.data_to_train.csv";
                String TRAINING_RESULTS_FILE = wd + project + "c.regression.results.csv"; // output of Training(), models performances
                String FEATURE_SELECTION_FILE = wd + project + "b.featureSelection.RELIEFF.csv"; // output of Training(), feature selection result
                BestModelSelectionAndReport b = new BestModelSelectionAndReport(REGRESSION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
                        "regression");
            }
        }

        /*
         Exit
         */
        //System.out.println("##Finished with success !");
        //System.exit(0);
    }

    public static void Main() {
        setConfiguration();
    }

    /**
     * set options read from command line
     *
     * @param args
     */
    public static void setOptionsFromCommandLine(String[] args) {
        //parse option
        String cmd = " ";
        for (String s : args) {
            cmd += s + " ";
        }
        String[] options = cmd.split(" -");

        //in case of ccmd present (only authorized AT THE END for now)
        //this implementation is temporary, need for something more generalized
        for (int i = 0; i < options.length; i++) {
            if (options[i].startsWith("ccmd") || options[i].startsWith("rcmd")) {
                for (int j = i + 1; j < options.length; j++) {
                    if (!options[j].startsWith("ccmd")
                            && !options[j].startsWith("rcmd") //TODO implement: && NOT A RESERVED KEYWORD
                            ) {
                        options[i] += " -" + options[j];
                        options[j] = "";
                    } else {
                        break;
                    }
                }
            }

        }

        //set options
        boolean prefixesDefined = false;
        for (String s : options) {
            if (s.equals("help")) {
                System.out.println("Check readme.md file or https://github.com/mickaelleclercq/BioDiscML");
                System.exit(0);
            }
            if (s.contains("=")) {
                setOption(s.split("=")[0], s.split("=")[1]);
            }
            // get config
            if (s.startsWith("config") && configFile.isEmpty()) {
                configFile = s.split(" ")[1].trim();
            }
            // training
            if (s.trim().equals("train")) {
                training = true;
            }

            //bestmodel
            if (s.startsWith("bestmodel")) {
                trainingBestModel = true;
                if (s.trim().contains(" ")) {
                    String modelID[] = s.split(" ");
                    for (int i = 1; i < modelID.length; i++) {
                        hmTrainingBestModelList.put(modelID[i], i + "");
                    }
                }
            }

            //Predict new data
            if (s.startsWith("predict")) {
                predictNewData = true;
            }

        }

        //if no config file is provided for training, check if we have enough
        //information to start biodiscml
        if (configFile.isEmpty() && training) {
            if (!hmTrainFiles.isEmpty()) {
                needConfigFile = false;
            }
        }

        if (!prefixesDefined) {
            for (String file : hmTrainFiles.keySet()) {
                hmTrainFiles.put(file, "");
            }
        }

        if (predictNewData) {
            System.out.println("#### Mode: Prediction");
            System.out.println("Merging ID: " + mergingID);
            System.out.println("Configuration file: " + configFile);
            if (isClassification) {
                System.out.println("Prediction type: Classification");
            } else {
                System.out.println("Prediction type: Regression");
            }
        } else if (training) {
            System.out.println("#### Mode: Training");
        } else if (trainingBestModel) {
            System.out.println("#### Mode: Best model");
        } else {
            System.err.println("[error] No mode selected (train, bestmodel or predict). "
                    + "Add -train or -bestmodel or -predict to your command line");
        }

        //export config file when executing command line options
        if (configFile.isEmpty()) {
            try {
                if (debug) {
                    System.out.println("Export config file in " + wd + project + ".config");
                }
                PrintWriter pw = new PrintWriter(new FileWriter(wd + project + ".config"));
                for (String option : options) {
                    if (option.contains("=")) {
                        pw.println(option);
                    }
                }
                pw.close();
            } catch (Exception e) {
                e.printStackTrace();
            }

        }

    }

    /**
     * read config file
     */
    public static void setConfiguration() {
        if (!new File(configFile).exists()) {
            if (configFile.isEmpty()) {
                configFile = "empty";
            }
            System.err.println("[error] Configuration file not found (provided source: " + configFile + "). Set config file with -config option");
            System.exit(0);
        }
        System.out.println("#### Reading configuration file " + configFile);
        String line = null;
        try {
            BufferedReader br = new BufferedReader(new FileReader(configFile));
            while (br.ready()) {
                line = br.readLine();
                if (!line.startsWith("#") && !line.trim().isEmpty()) {
                    String option = line.split("=")[0].trim();
                    String value = line.split("=")[1].trim();
                    //System.out.println(option + ":" + value);
                    setOption(option, value);
                }
            }
            CVfolder = wd + CVfolder;

        } catch (Exception e) {
            System.err.println("Parsing error in config file at line " + line);
            e.printStackTrace();
            System.exit(0);
        }

    }

    private static void setOption(String option, String value) {
        switch (option) {
            case "config":
                configFile = value.trim();
                break;
            case "debug":
                debug = Boolean.valueOf(value.trim());
                break;
            case "wd":
                wd = value.trim();
                if (!wd.endsWith(File.separator)) {
                    wd = wd + File.separator;
                }
                break;
            case "project":
                project = value.trim() + "_";
                System.out.println("Project name: " + project);
                break;
            case "cvfolder":
                CVfolder = value.trim();
//                            if (!trainingBestModel) {
//                                System.out.println("Cross validation folder:" + CVfolder);
//                            }
                break;
            case "trainFile":
                try {
                    hmTrainFiles.put(wd + value.split(",")[0].trim(), value.split(",")[1].trim()); //filename,prefix
                } catch (Exception e) {
                    hmTrainFiles.put(wd + value.replace(",", "").trim(), ""); //filename
                }
                break;
            case "newDataFile":
                try {
                    hmNewDataFiles.put(wd + value.split(",")[0].trim(), value.split(",")[1].trim()); //filename,prefix
                } catch (Exception e) {
                    hmNewDataFiles.put(wd + value.replace(",", "").trim(), ""); //filename
                }
                break;
            case "validationFile":
                try {
                    hmNewDataFiles.put(wd + value.split(",")[0].trim(), value.split(",")[1].trim()); //filename,prefix
                } catch (Exception e) {
                    hmNewDataFiles.put(wd + value.replace(",", "").trim(), ""); //filename
                }
                break;
            case "excluded":
                String excluded[] = value.split(",");
                for (String ex : excluded) {
                    hmExcludedFeatures.put(ex.trim(), "");
                }
                break;
            case "mergingID":
                mergingID = value.trim();
                if (!trainingBestModel) {
                    System.out.println("Merging ID: " + mergingID);
                }
                break;
            case "separator":
                separator = value.trim();
                break;
            case "classification":
                isClassification = true;
                break;
            case "regression":
                isClassification = false;
                break;

            case "doClassification":
                doClassification = Boolean.valueOf(value.trim());
                break;

            case "classificationClassName":
                classificationClassName = value.trim();
                if (doClassification) {
                    System.out.println("ClassificationClassName: " + classificationClassName);
                }
                break;
            case "classificationFastWay":
                classificationFastWay = Boolean.valueOf(value.trim());
                break;
            case "numberOfBestModels":
                numberOfBestModels = Integer.valueOf(value.trim());
                break;
            case "numberOfBestModelsSortingMetric":
                bestModelsSortingMetric = value.trim().toUpperCase();
                break;
            case "numberOfBestModelsSortingMetricThreshold":
                bestModelsSortingMetricThreshold = Double.valueOf(value.trim());
                break;
            case "ccmd":
                switch (value.split(",").length) {
                    case 1:
                        classificationFastWayCommands.add(value.trim()
                                + ":allopt:allsearch");
                        break;
                    case 2:
                        classificationFastWayCommands.add(value.split(",")[0].trim()
                                + ":" + value.split(",")[1].trim().toLowerCase() + ":allsearch");
                        break;
                    case 3:
                        classificationFastWayCommands.add(value.split(",")[0].trim()
                                + ":" + value.split(",")[1].trim().toLowerCase()
                                + ":" + value.split(",")[2].trim().toLowerCase());
                        break;
                    default:
                        break;
                }

                break;
            case "rcmd":
                switch (value.split(",").length) {
                    case 1:
                        regressionFastWayCommands.add(value.trim()
                                + ":allopt:allsearch");
                        break;
                    case 2:
                        regressionFastWayCommands.add(value.split(",")[0].trim()
                                + ":" + value.split(",")[1].trim().toLowerCase() + ":allsearch");
                        break;
                    case 3:
                        regressionFastWayCommands.add(value.split(",")[0].trim()
                                + ":" + value.split(",")[1].trim().toLowerCase()
                                + ":" + value.split(",")[2].trim().toLowerCase());
                        break;
                    default:
                        break;
                }
            case "coptimizers":
                classificationOptimizers = value.trim().toLowerCase();
                break;
            case "searchmodes":
                searchmodes = value.trim().toLowerCase();
                break;
            case "doRegression":
                doRegression = Boolean.valueOf(value.trim());
                break;
            case "regressionClassName":
                regressionClassName = value.trim();
                break;
            case "regressionFastWay":
                regressionFastWay = Boolean.valueOf(value.trim());
                break;

            case "roptimizers":
                regressionOptimizers = value.trim().toLowerCase();
                break;
            case "maxNumberOfSelectedFeatures":
                maxNumberOfSelectedFeatures = Integer.valueOf(value.trim());
                break;
            case "maxNumberOfFeaturesInModel":
                maxNumberOfFeaturesInModel = Integer.valueOf(value.trim());
                break;
            case "bootstrapFolds":
                bootstrapAndRepeatedHoldoutFolds = Integer.valueOf(value.trim());
                break;
            case "spearmanCorrelation_lower":
                spearmanCorrelation_lower = Double.valueOf(value.trim());
                break;
            case "spearmanCorrelation_upper":
                spearmanCorrelation_upper = Double.valueOf(value.trim());
                break;
            case "pearsonCorrelation_lower":
                pearsonCorrelation_lower = Double.valueOf(value.trim());
                break;
            case "pearsonCorrelation_upper":
                pearsonCorrelation_upper = Double.valueOf(value.trim());
                break;
            case "maxRankingScoreDifference":
                maxRankingScoreDifference = Double.valueOf(value.trim());
                break;
            case "retreiveCorrelatedGenesByRankingScore":
                retreiveCorrelatedGenesByRankingScore = Boolean.valueOf(value.trim());
                break;
            case "combineModels":
                combineModels = Boolean.valueOf(value.trim());
                break;
            case "retrieveCorrelatedGenes":
                retrieveCorrelatedGenes = Boolean.valueOf(value.trim());
                break;
            case "combinationRule":
                combinationRule = value.trim().toUpperCase();
                break;
            case "sampling":
                doSampling = Boolean.valueOf(value.trim());
                break;
            case "roc_curves":
                ROCcurves = Boolean.valueOf(value.trim());
                break;
            case "loocv":
                loocv = Boolean.valueOf(value.trim());
                break;
            case "samplingFold":
                samplingFold = Integer.valueOf(value.trim());
                break;
            case "cpus":
                cpus = value.trim();
                break;
            case "computeBestModel":
                computeBestModel = Boolean.valueOf(value.trim());
                break;
            case "modelFile":
                modelFile = value.trim();
                break;

        }
    }

}
