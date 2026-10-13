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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.nio.file.StandardCopyOption;

/**
 *
 * @author Mickael
 */
public class Main {

    public static boolean debug = false;
    public static boolean debug2 = false;
    public static boolean printFailedModels = false; //print errors of failed models
    public static boolean isClassification = true;

    public static String wd = "";
    public static String project = "myProject";

    //program functions
    public static String configFile = ""; //config file
    public static boolean needConfigFile = true;
    public static boolean training = false;
    public static boolean predictNewData = false;
    public static boolean trainingBestModel = false;
    public static LinkedHashMap<String, String> hmTrainingBestModelList = new LinkedHashMap<>();//modelID, identifier prefix

    public static String modelFile = "";
    public static HashMap<String, String> hmExcludedFeatures = new HashMap<>();//features to exclude from the final dataset

    //config
    public static String mergingID = "Instance";

    //source files, in the order of the configuration (the class is taken
    //from the first file that contains it)
    public static HashMap<String, String> hmTrainFiles = new LinkedHashMap<>();//filename, identifier prefix
    public static HashMap<String, String> hmNewDataFiles = new LinkedHashMap<>();//filename, identifier prefix (-predict)
    public static HashMap<String, String> hmValidationFiles = new LinkedHashMap<>();//filename, identifier prefix (-train)

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
    public static String searchmodes = "f,fb,b,bf,top1,top5,top10,top15,top20,top30,top40,top50,top75,top100,top200,all";
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
    public static Integer maxNumberOfFeaturesInModel = 200;
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
    public static boolean repeatedHoldout = true;
    public static boolean bootstrap = true;
    public static boolean computeBestModel = true;

    public static boolean resumeTraining = false;
    public static boolean restoreRun = false;
    public static String previousRunPath = "";
    public static String previousRunProjectName = "";

    public static Boolean noFeatureSelection = false;
    public static boolean generateModelWithCorrelatedGenes = false;
    public static String missingValueToReplace = "?"; //for testing, if a feature is missing, set a value to replace missing data

    //benchmark
    public static String bench_AUC = "";

    //TODO
    static boolean retreiveCorrelatedGenesByRankingScore = true; //avoid for non-binary classes and regression
    static boolean ROCcurves = false; //experimental
    static boolean UpSetR = false; //experimental
    static boolean performShortTest = true;
    // seed of all random number generators (sampling, cross validations,
    // holdouts, bootstraps), so that two identical runs give identical results
    public static int seed = 1;
    // positive class of a binary classification (sensitivity, specificity,
    // PPV, FDR, F-score, AUPRC). Empty: see Weka_module.getPositiveClassIndex
    public static String positiveClass = "";
    // exit code of the run: set to 1 by the errors that do not stop the run
    // at once, so that schedulers and pipelines see the failure
    public static int exitCode = 0;

    public static void main(String[] args) throws IOException {
        System.out.println("#### BioDiscML ####\n");
        // -help works with any java version
        for (String arg : args) {
            String a = arg.trim().replaceFirst("^-+", "");
            if (a.equals("help")) {
                System.out.println("Read readme.md file or https://github.com/mickaelleclercq/BioDiscML");
                System.exit(0);
            }
        }
        // check java version
        String version = System.getProperty("java.version");
        if (!version.contains("1.8")) {
            String arg = "";
            for (String argl : args) {
                arg += argl + " ";
            }
            System.out.println("ERROR: Java version is " + version + "."
                    + "\nYou need java 1.8. If it is installed, try this command:\n"
            );
            if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                System.out.println("\"c:\\Program Files\\Java\\jdk1.8.0_111\\bin\\java.exe\" -jar biodiscml.jar " + arg);
            } else {
                System.out.println("/mnt/software/jvm/jdk1.8.0_371/bin/java -jar biodiscml.jar " + arg);
            }
            System.exit(1);
        }
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
                File classifiers = new File(wd + "/" + "classifiers.conf");
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
                // (the last line used to be skipped by a br.ready() loop)
                while ((line = br.readLine()) != null) {
                    if (!line.trim().startsWith("#") && line.contains("=")) {
                        String option = line.substring(0, line.indexOf("=")).trim();
                        String value = line.substring(line.indexOf("=") + 1).trim();
                        switch (option) {
                            case "ccmd":
                                classificationBruteForceCommands.add(value.trim());
                                break;
                            case "rcmd":
                                regressionBruteForceCommands.add(value.trim());
                                break;
                        }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
                System.err.println("[error] Unable to read classifiers.conf");
                System.exit(1);
            }
        } else {
            System.out.println("Model search mode: Fast way mode");
        }

        //set number of max cpus to use
        if (!cpus.equals("max")) {
            // with a parallelism of 0, the common pool has no worker thread and,
            // with Java 8, a parallel stream can wait forever for a task that
            // only a worker would run (observed in -bestmodel, when retrieving
            // correlated features): use 1, i.e. run sequentially
            try {
                if (Integer.parseInt(cpus) < 1) {
                    System.err.println("[warning] cpus=" + cpus + " is not a valid number of cpus: 1 is used");
                    cpus = "1";
                }
            } catch (NumberFormatException e) {
                System.err.println("[warning] cpus=" + cpus + " is not a number (or max): max is used");
                cpus = "max";
            }
        }
        if (!cpus.equals("max")) {
            System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", cpus);
        }

        // Retrieve previously existing a (train and test data) and b (infogain) files
        if (restoreRun) {
            resumeTraining = true;
            // get current directory
            if (wd.isEmpty()) {
                wd = new java.io.File(".").getCanonicalPath() + java.io.File.separator;
            }
            System.out.println("Restoring data from previous project " + previousRunProjectName
                    + "\nCopying files from " + previousRunPath);
            //copy previous run
            try {
                Path source = Paths.get(previousRunPath + previousRunProjectName + "_a.classification.all_data.arff");
                Path destination = Paths.get(wd + "/" + project + "a.classification.all_data.arff");
                Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);

                source = Paths.get(previousRunPath + previousRunProjectName + "_a.classification.all_data.csv");
                destination = Paths.get(wd + "/" + project + "a.classification.all_data.csv");
                Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);

                source = Paths.get(previousRunPath + previousRunProjectName + "_a.classification.data_to_test.arff");
                destination = Paths.get(wd + "/" + project + "a.classification.data_to_test.arff");
                Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);

                source = Paths.get(previousRunPath + previousRunProjectName + "_a.classification.data_to_test.csv");
                destination = Paths.get(wd + "/" + project + "a.classification.data_to_test.csv");
                Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);

                source = Paths.get(previousRunPath + previousRunProjectName + "_a.classification.data_to_train.arff");
                destination = Paths.get(wd + "/" + project + "a.classification.data_to_train.arff");
                Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);

                source = Paths.get(previousRunPath + previousRunProjectName + "_a.classification.data_to_train.csv");
                destination = Paths.get(wd + "/" + project + "a.classification.data_to_train.csv");
                Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);

                source = Paths.get(previousRunPath + previousRunProjectName + "_b.featureSelection.infoGain.arff");
                destination = Paths.get(wd + "/" + project + "b.featureSelection.infoGain.arff");
                Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);

                source = Paths.get(previousRunPath + previousRunProjectName + "_b.featureSelection.infoGain.csv");
                destination = Paths.get(wd + "/" + project + "b.featureSelection.infoGain.csv");
                Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);

                File f = new File(previousRunPath + previousRunProjectName + "_c.classification.results.csv");
                f.createNewFile();
            } catch (Exception e) {
                e.printStackTrace();
                System.err.println("[error] Unable to restore the previous run");
                System.exit(1);
            }

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
                if (debug) {
                    System.out.println("CLASSIFICATION_FILE: " + CLASSIFICATION_FILE);
                }
                if (new File(CLASSIFICATION_FILE).exists() && resumeTraining) {
                    System.out.println("Preprocessing of the input file(s) already done... skipping");
                } else {
                    AdaptDatasetToTraining c = new AdaptDatasetToTraining(CLASSIFICATION_FILE);
                }
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
                if (new File(REGRESSION_FILE).exists() && resumeTraining) {
                    System.out.println("Preprocessing of the input file(s) already done... skipped by resumeTraining");
                } else {
                    AdaptDatasetToTraining c = new AdaptDatasetToTraining(REGRESSION_FILE);
                }

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
                System.exit(1);
            }
        }

        if (predictNewData) {
            System.out.println("#### Start predicting new data...");
            //prediction type from the configuration
            if (doRegression && !doClassification) {
                isClassification = false;
            } else if (doClassification) {
                isClassification = true;
            }
            // printed here: they are only known once the config file is read
            System.out.println("Merging ID: " + mergingID);
            System.out.println("Prediction type: " + (isClassification ? "Classification" : "Regression"));
            if (modelFile.isEmpty()) {
                System.err.println("[error] No model file have been provided (Set a modelFile in config file)");
                System.exit(1);
            }
            if (hmNewDataFiles.isEmpty()) {
                System.err.println("[error] No new data file have been provided (Set a newDataFile in config file)");
                System.exit(1);
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
        if (exitCode != 0) {
            System.err.println("[error] BioDiscML finished with errors (see above)");
            System.exit(exitCode);
        }
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
            if (s.trim().equals("help")) {
                System.out.println("Read readme.md file or https://github.com/mickaelleclercq/BioDiscML");
                System.exit(0);
            }
            if (s.contains("=")) {
                setOption(s.substring(0, s.indexOf("=")).trim(), s.substring(s.indexOf("=") + 1).trim());
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
            System.out.println("Configuration file: " + configFile);
        } else if (training) {
            System.out.println("#### Mode: Training");
        } else if (trainingBestModel) {
            System.out.println("#### Mode: Best model");
        } else {
            System.err.println("[error] No mode selected (train, bestmodel or predict). "
                    + "Add -train or -bestmodel or -predict to your command line");
            System.exit(1);
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
            System.exit(1);
        }
        System.out.println("#### Reading configuration file " + configFile);
        String line = null;
        try {
            BufferedReader br = new BufferedReader(new FileReader(configFile));
            while ((line = br.readLine()) != null) {
                if (!line.trim().startsWith("#") && !line.trim().isEmpty()) {
                    if (!line.contains("=")) {
                        System.err.println("[warning] Line ignored in config file (no option=value): " + line);
                        continue;
                    }
                    // split at the first '=' only: values can contain '='
                    String option = line.substring(0, line.indexOf("=")).trim();
                    // remove end of line comments ("value # comment")
                    String value = line.substring(line.indexOf("=") + 1).replaceAll("\\s+#.*$", "").trim();
                    if (value.isEmpty()) {
                        System.err.println("[warning] No value for option " + option + " in config file: default value kept");
                        continue;
                    }
                    //System.out.println(option + ":" + value);
                    setOption(option, value);
                }
            }
            br.close();

        } catch (Exception e) {
            System.err.println("[error] Parsing error in config file at line " + line);
            e.printStackTrace();
            System.exit(1);
        }

    }

    /**
     * Parse a fast way command: "classifier options[, optimizer[, search mode]]".
     * The optimizer and the search mode are recognized from the end, so that
     * the classifier options can contain commas.
     *
     * @param value
     * @return classifier options:optimizer:searchmode (allopt and allsearch if
     * not provided)
     */
    static String parseFastWayCommand(String value) {
        String parts[] = value.split(",", -1);
        int n = parts.length;
        String optimizer = "allopt";
        String searchmode = "allsearch";
        int end = n; // parts[0..end-1] are the classifier and its options
        if (n >= 3 && isSearchMode(parts[n - 1]) && isOptimizer(parts[n - 2])) {
            optimizer = parts[n - 2].trim().toLowerCase();
            searchmode = parts[n - 1].trim().toLowerCase();
            end = n - 2;
        } else if (n >= 2 && isOptimizer(parts[n - 1])) {
            optimizer = parts[n - 1].trim().toLowerCase();
            end = n - 1;
        }
        String command = String.join(",", java.util.Arrays.copyOfRange(parts, 0, end)).trim();
        return command + ":" + optimizer + ":" + searchmode;
    }

    private static boolean isOptimizer(String s) {
        s = s.trim().toLowerCase();
        return s.equals("allopt") || Training.KNOWN_OPTIMIZERS.contains(s);
    }

    private static boolean isSearchMode(String s) {
        s = s.trim().toLowerCase();
        return s.equals("allsearch") || s.equals("f") || s.equals("fb") || s.equals("b")
                || s.equals("bf") || s.equals("all") || s.matches("top[0-9]+");
    }

    /**
     * @param option
     * @param value
     * @return the boolean value: true/yes/1/on or false/no/0/off (any case)
     */
    static boolean parseBoolean(String option, String value) {
        switch (value.trim().toLowerCase()) {
            case "true":
            case "yes":
            case "1":
            case "on":
                return true;
            case "false":
            case "no":
            case "0":
            case "off":
                return false;
            default:
                System.err.println("[warning] Invalid value " + value + " for option " + option
                        + " (expected true or false): false is used");
                return false;
        }
    }

    private static void setOption(String option, String value) {
        switch (option) {
            case "config":
                configFile = value.trim();
                break;
            case "debug":
                debug = parseBoolean(option, value);
                break;
            case "debug2":
                debug2 = parseBoolean(option, value);
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
                    hmValidationFiles.put(wd + value.split(",")[0].trim(), value.split(",")[1].trim()); //filename,prefix
                } catch (Exception e) {
                    hmValidationFiles.put(wd + value.replace(",", "").trim(), ""); //filename
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
                doClassification = parseBoolean(option, value);
                break;

            case "classificationClassName":
                classificationClassName = value.trim();
                if (doClassification) {
                    System.out.println("ClassificationClassName: " + classificationClassName);
                }
                break;
            case "classificationFastWay":
                classificationFastWay = parseBoolean(option, value);
                break;
            case "numberOfBestModels":
                numberOfBestModels = Integer.valueOf(value.trim());
                break;
            case "numberOfBestModelsSortingMetric":
                // case is resolved against the results file header
                bestModelsSortingMetric = value.trim();
                break;
            case "numberOfBestModelsSortingMetricThreshold":
                bestModelsSortingMetricThreshold = Double.valueOf(value.trim());
                break;
            case "ccmd":
                classificationFastWayCommands.add(parseFastWayCommand(value));
                break;
            case "rcmd":
                regressionFastWayCommands.add(parseFastWayCommand(value));
                break;
            case "coptimizers":
                classificationOptimizers = value.trim().toLowerCase();
                break;
            case "searchmodes":
                searchmodes = value.trim().toLowerCase();
                break;
            case "doRegression":
                doRegression = parseBoolean(option, value);
                break;
            case "regressionClassName":
                regressionClassName = value.trim();
                break;
            case "regressionFastWay":
                regressionFastWay = parseBoolean(option, value);
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
                retreiveCorrelatedGenesByRankingScore = parseBoolean(option, value);
                break;
            case "combineModels":
                combineModels = parseBoolean(option, value);
                break;
            case "retrieveCorrelatedGenes":
                retrieveCorrelatedGenes = parseBoolean(option, value);
                break;
            case "generateModelWithCorrelatedGenes":
                generateModelWithCorrelatedGenes = parseBoolean(option, value);
                break;
            case "combinationRule":
                combinationRule = value.trim().toUpperCase();
                break;
            case "sampling":
                doSampling = parseBoolean(option, value);
                break;
            case "roc_curves":
                ROCcurves = parseBoolean(option, value);
                break;
            case "loocv":
                loocv = parseBoolean(option, value);
                break;
            case "samplingFold":
                samplingFold = Integer.valueOf(value.trim());
                break;
            case "cpus":
                cpus = value.trim();
                break;
            case "computeBestModel":
                computeBestModel = parseBoolean(option, value);
                break;
            case "modelFile":
                modelFile = value.trim();
                break;
            case "printFailedModels":
                printFailedModels = parseBoolean(option, value);
                break;
            case "resumeTraining":
                resumeTraining = parseBoolean(option, value);
                break;
            case "upsetr":
                UpSetR = parseBoolean(option, value);
                break;
            case "repeatedHoldoutTrain":
                repeatedHoldout = parseBoolean(option, value);
                break;
            case "bootstrap":
                bootstrap = parseBoolean(option, value);
                break;
            case "restoreRun":
                restoreRun = parseBoolean(option, value);
                break;
            case "noFeatureSelection":
                noFeatureSelection = parseBoolean(option, value);
                break;
            case "previousRunPath":
                previousRunPath = value.trim();
                break;
            case "previousRunProjectName":
                previousRunProjectName = value.trim();
                break;
            case "performShortTest":
                performShortTest = parseBoolean(option, value);
                break;
            case "positiveClass":
                positiveClass = value.trim();
                break;
            case "seed":
                seed = Integer.valueOf(value.trim());
                break;
            case "missingValueToReplace":
                missingValueToReplace = value.trim();
                break;
            default:
                // e.g. a misspelled option, which would otherwise be silently ignored
                System.err.println("[warning] Unknown option " + option + ": ignored");
                break;
        }
    }

}
