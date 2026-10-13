/*
 * execute machine learning algorithms in parrallel
 * Works with regression and classification
 */
package biodiscml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import org.apache.commons.math3.stat.descriptive.moment.*;
import utils.Weka_module;
import utils.utils;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;

/**
 *
 * @author Mickael
 */
public class Training {

    public static Weka_module weka = new Weka_module();
    public static AttributeSelection ranking;
    public static ArrayList<String[]> alClassifiers = new ArrayList<>();
    public static boolean isClassification = true;
    public static String resultsSummaryHeader = "";
    // incremented by the parallel training threads
    public static AtomicInteger cptPassed = new AtomicInteger();
    public static AtomicInteger cptFailed = new AtomicInteger();
    public boolean parrallel = true;
    public static String trainFileName = "";
    public static PrintWriter pw;

    public Training() {
    }

    /**
     * @param dataToTrainModel
     * @param resultsFile
     * @param featureSelectionFile
     * @param type. Is "class" for classification, else "reg" for regression
     */
    public Training(String dataToTrainModel, String resultsFile,
            String featureSelectionFile, String type) {
        // deleting previous run
        if (!Main.restoreRun && !Main.resumeTraining) {
            if (new File(featureSelectionFile).exists()) {
                System.out.println("\t" + featureSelectionFile + " exist... deleting...");
                new File(featureSelectionFile).delete();
                new File(featureSelectionFile.replace(".csv", ".arff")).delete();
            }
            if (new File(resultsFile).exists()) {
                System.out.println("\t" + resultsFile + " exist... deleting...");
                new File(resultsFile).delete();
                new File(resultsFile.replace(".csv", ".arff")).delete();
            }
        }

        if (Main.cpus.equals("1")) {
            parrallel = false;
        }
        System.out.println("Training on " + dataToTrainModel + ". All tested models results will be in " + resultsFile);
        alClassifiers = new ArrayList<>();
        isClassification = type.equals("class");
        trainFileName = dataToTrainModel;
//        File f = new File(Main.CVfolder);//cross validation folder outputs
//        if (!f.exists()) {
//            f.mkdir();
//        }

        //convert csv to arff
        if (dataToTrainModel.endsWith(".csv") && !new File(dataToTrainModel.replace(".csv", ".arff")).exists()) {
            weka.setCSVFile(new File(dataToTrainModel));
            weka.csvToArff(isClassification);
        } else {
            weka.setARFFfile(dataToTrainModel.replace(".csv", ".arff"));
        }

        //set local variable of weka object from ARFFfile
        weka.setDataFromArff();
        weka.myData = weka.convertStringsToNominal(weka.myData);

//        // check if class has numeric values, hence regression, instead of nominal class (classification)
        //classification = weka.isClassification();
        //CLASSIFICATION
        if (isClassification) {
            if (Main.debug) {
                System.out.println("Only use classification algorithms");
            }
            printPositiveClass(weka.myData);
            //HEADER FOR SUMMARY OUTPUT
            resultsSummaryHeader = "ID"
                    + "\tclassifier"
                    + "\tOptions"
                    + "\tOptimizedValue"
                    + "\tSearchMode"
                    + "\tnbrOfFeatures"
                    //10CV
                    + "\tTRAIN_10CV_ACC"
                    + "\tTRAIN_10CV_AUC"
                    + "\tTRAIN_10CV_AUPRC"
                    + "\tTRAIN_10CV_SEN"
                    + "\tTRAIN_10CV_SPE"
                    + "\tTRAIN_10CV_MCC"
                    + "\tTRAIN_10CV_MAE"
                    + "\tTRAIN_10CV_BER"
                    + "\tTRAIN_10CV_FPR"
                    + "\tTRAIN_10CV_FNR"
                    + "\tTRAIN_10CV_PPV"
                    + "\tTRAIN_10CV_FDR"
                    + "\tTRAIN_10CV_Fscore"
                    + "\tTRAIN_10CV_kappa"
                    + "\tTRAIN_matrix"
                    //LOOCV
                    + "\tTRAIN_LOOCV_ACC"
                    + "\tTRAIN_LOOCV_AUC"
                    + "\tTRAIN_LOOCV_AUPRC"
                    + "\tTRAIN_LOOCV_SEN"
                    + "\tTRAIN_LOOCV_SPE"
                    + "\tTRAIN_LOOCV_MCC"
                    + "\tTRAIN_LOOCV_MAE"
                    + "\tTRAIN_LOOCV_BER"
                    //Repeated Holdout TRAIN
                    + "\tTRAIN_RH_ACC"
                    + "\tTRAIN_RH_AUC"
                    + "\tTRAIN_RH_AUPRC"
                    + "\tTRAIN_RH_SEN"
                    + "\tTRAIN_RH_SPE"
                    + "\tTRAIN_RH_MCC"
                    + "\tTRAIN_RH_MAE"
                    + "\tTRAIN_RH_BER"
                    //Bootstrap TRAIN
                    + "\tTRAIN_BS_ACC"
                    + "\tTRAIN_BS_AUC"
                    + "\tTRAIN_BS_AUPRC"
                    + "\tTRAIN_BS_SEN"
                    + "\tTRAIN_BS_SPE"
                    + "\tTRAIN_BS_MCC"
                    + "\tTRAIN_BS_MAE"
                    + "\tTRAIN_BS_BER"
                    //Bootstrap .632+ TRAIN
                    + "\tTRAIN_BS.632+"
                    //test
                    + "\tTEST_ACC"
                    + "\tTEST_AUC"
                    + "\tTEST_AUPRC"
                    + "\tTEST_SEN"
                    + "\tTEST_SPE"
                    + "\tTEST_MCC"
                    + "\tTEST_MAE"
                    + "\tTEST_BER"
                    //Repeated Holdout TRAIN_TEST
                    + "\tTRAIN_TEST_RH_ACC"
                    + "\tTRAIN_TEST_RH_AUC"
                    + "\tTRAIN_TEST_RH_AUPRC"
                    + "\tTRAIN_TEST_RH_SEN"
                    + "\tTRAIN_TEST_RH_SPE"
                    + "\tTRAIN_TEST_RH_MCC"
                    + "\tTRAIN_TEST_RH_MAE"
                    + "\tTRAIN_TEST_RH_BER"
                    //Bootstrap TRAIN_TEST
                    + "\tTRAIN_TEST_BS_ACC"
                    + "\tTRAIN_TEST_BS_AUC"
                    + "\tTRAIN_TEST_BS_AUPRC"
                    + "\tTRAIN_TEST_BS_SEN"
                    + "\tTRAIN_TEST_BS_SPE"
                    + "\tTRAIN_TEST_BS_MCC"
                    + "\tTRAIN_TEST_BS_MAE"
                    + "\tTRAIN_TEST_BS_BER"
                    //Bootstrap .632+ TRAIN
                    + "\tTRAIN_TEST_BS.632+"
                    //stats
                    + "\tAVG_BER"
                    + "\tSTD_BER"
                    + "\tAVG_MAE"
                    + "\tSTD_MAE"
                    + "\tAVG_MCC"
                    + "\tSTD_MCC"
                    + "\tAttributeList";

            //fast way classification. Predetermined commands.
            if (Main.classificationFastWay) {
                //hmclassifiers.put(new String[]{"misc.VFI", "-B 0.6", "AUC"}, "");
                //hmclassifiers.put(new String[]{"meta.CostSensitiveClassifier", "-cost-matrix \"[0.0 ratio; 100.0 0.0]\" -S 1 -W weka.classifiers.misc.VFI -- -B 0.4", "MCC"}, "");

                for (String cmd : Main.classificationFastWayCommands) {
                    String optimizer = cmd.split(":")[1];
                    String searchmode = cmd.split(":")[2];
                    String classifier = cmd.split(":")[0].split(" ")[0];
                    String options = cmd.split(":")[0].replace(classifier, "");
                    //case where optimizer AND search modes are empty
                    if (optimizer.equals("allopt") && searchmode.equals("allsearch")) {
                        for (String allOptimizers : Main.classificationOptimizers.split(",")) {
                            for (String allSearchModes : Main.searchmodes.split(",")) {
                                alClassifiers.add(new String[]{classifier, options, allOptimizers.trim(), allSearchModes.trim()});
                            }
                        }
                        //Case where only searchmode is empty
                    } else if (!optimizer.equals("allopt") && searchmode.equals("allsearch")) {
                        for (String allSearchModes : Main.searchmodes.split(",")) {
                            alClassifiers.add(new String[]{classifier, options, optimizer.trim(), allSearchModes.trim()});
                        }
                        //case where only optimizer is empty
                    } else if (optimizer.equals("allopt") && !searchmode.equals("allsearch")) {
                        for (String allOptimizers : Main.classificationOptimizers.split(",")) {
                            alClassifiers.add(new String[]{classifier, options, allOptimizers.trim(), searchmode.trim()});
                        }
                        //case where optimizer and searchmode are provided
                    } else {
                        alClassifiers.add(new String[]{classifier, options, optimizer, searchmode});
                    }

                }
            } else {
                // if no feature selection is asked
                if (Main.noFeatureSelection) {
                    Main.maxNumberOfSelectedFeatures = weka.myData.numAttributes();
                    Main.maxNumberOfFeaturesInModel = weka.myData.numAttributes();
                    for (String cmd : Main.classificationBruteForceCommands) {
                        String classifier = cmd.split(" ")[0];
                        String options = cmd.replace(classifier, "").trim();
                        addClassificationToQueue(classifier, options);
                    }

                } else {
                    //brute force classification, try everything in the provided classifiers and optimizers
                    for (String cmd : Main.classificationBruteForceCommands) {
                        String classifier = cmd.split(" ")[0];
                        String options = cmd.replace(classifier, "").trim();
                        addClassificationToQueue(classifier, options);
                    }
                }
            }

            if (!Main.noFeatureSelection) {
                System.out.print("Feature selection and ranking...");
                if (new File(featureSelectionFile).exists() && Main.resumeTraining) {
                    System.out.print("\nFeature selection and ranking already done... skipping");
                } else {
                    //ATTRIBUTE SELECTION for classification
                    weka.attributeSelectionByInfoGainRankingAndSaveToCSV(featureSelectionFile);
                    //get the rank of attributes
                    ranking = weka.featureRankingForClassification();
                    System.out.println("[done]");
                    //reset arff and keep compatible header
                    weka.setCSVFile(new File(featureSelectionFile));
                    weka.csvToArff(isClassification);
                    weka.makeCompatibleARFFheaders(dataToTrainModel.replace("data_to_train.csv", "data_to_train.arff"),
                            featureSelectionFile.replace("infoGain.csv", "infoGain.arff"));
                }
                weka.setARFFfile(featureSelectionFile.replace("infoGain.csv", "infoGain.arff"));
                weka.setDataFromArff();
            }

        } else {
            //REGRESSION
            if (Main.debug) {
                System.out.println("Only use regression algorithms");
            }
            resultsSummaryHeader = "ID"
                    + "\tclassifier"
                    + "\tOptions"
                    + "\tOptimizedValue"
                    + "\tSearchMode"
                    + "\tnbrOfFeatures"
                    //10cv
                    + "\tTRAIN_10CV_CC"
                    + "\tTRAIN_10CV_MAE"
                    + "\tTRAIN_10CV_RMSE"
                    + "\tTRAIN_10CV_RAE"
                    + "\tTRAIN_10CV_RRSE"
                    //LOOCV
                    + "\tTRAIN_LOOCV_CC"
                    + "\tTRAIN_LOOCV_MAE"
                    + "\tTRAIN_LOOCV_RMSE"
                    + "\tTRAIN_LOOCV_RAE"
                    + "\tTRAIN_LOOCV_RRSE"
                    //Repeated Holdout
                    + "\tTRAIN_RH_CC"
                    + "\tTRAIN_RH_MAE"
                    + "\tTRAIN_RH_RMSE"
                    + "\tTRAIN_RH_RAE"
                    + "\tTRAIN_RH_RRSE"
                    //Bootstrap
                    + "\tTRAIN_BS_CC"
                    + "\tTRAIN_BS_MAE"
                    + "\tTRAIN_BS_RMSE"
                    + "\tTRAIN_BS_RAE"
                    + "\tTRAIN_BS_RRSE"
                    //TEST SET
                    + "\tTEST_CC"
                    + "\tTEST_MAE"
                    + "\tTEST_RMSE"
                    + "\tTEST_RAE"
                    + "\tTEST_RRSE"
                    //Repeated Holdout TRAIN_TEST
                    + "\tTRAIN_TEST_RH_CC"
                    + "\tTRAIN_TEST_RH_MAE"
                    + "\tTRAIN_TEST_RH_RMSE"
                    + "\tTRAIN_TEST_RH_RAE"
                    + "\tTRAIN_TEST_RH_RRSE"
                    //Bootstrap TRAIN_TEST
                    + "\tTRAIN_TEST_BS_CC"
                    + "\tTRAIN_TEST_BS_MAE"
                    + "\tTRAIN_TEST_BS_RMSE"
                    + "\tTRAIN_TEST_BS_RAE"
                    + "\tTRAIN_TEST_BS_RRSE"
                    //stats
                    + "\tAVG_CC"
                    + "\tSTD_CC"
                    + "\tAVG_MAE"
                    + "\tSTD_MAE"
                    + "\tAVG_RMSE"
                    + "\tSTD_RMSE"
                    + "\tAttributeList";

            //fast way regression. Predetermined commands.
            if (Main.regressionFastWay) {
                //hmclassifiers.put(new String[]{"functions.GaussianProcesses", "-L 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 1.0\"", "CC"}, "");
                //hmclassifiers.put(new String[]{"meta.AdditiveRegression",
                //       "-S 1.0 -I 10 -W weka.classifiers.functions.GaussianProcesses -- -L 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0 -L\"", "CC"}, "");

                for (String cmd : Main.regressionFastWayCommands) {
                    String optimizer = cmd.split(":")[1];
                    String searchmode = cmd.split(":")[2];
                    String classifier = cmd.split(":")[0].split(" ")[0];
                    String options = cmd.split(":")[0].replace(classifier, "");

                    //case where optimizer AND search modes are empty
                    if (optimizer.equals("allopt") && searchmode.equals("allsearch")) {
                        for (String allOptimizers : Main.regressionOptimizers.split(",")) {
                            for (String allSearchModes : Main.searchmodes.split(",")) {
                                alClassifiers.add(new String[]{classifier, options, allOptimizers.trim(), allSearchModes.trim()});
                            }
                        }
                        //Case where only searchmode is empty
                    } else if (!optimizer.equals("allopt") && searchmode.equals("allsearch")) {
                        for (String allSearchModes : Main.searchmodes.split(",")) {
                            alClassifiers.add(new String[]{classifier, options, optimizer.trim(), allSearchModes.trim()});
                        }
                        //case where only optimizer is empty
                    } else if (optimizer.equals("allopt") && !searchmode.equals("allsearch")) {
                        for (String allOptimizers : Main.regressionOptimizers.split(",")) {
                            alClassifiers.add(new String[]{classifier, options, allOptimizers.trim(), searchmode.trim()});
                        }
                        //case where optimizer and searchmode are provided
                    } else {
                        alClassifiers.add(new String[]{classifier, options, optimizer, searchmode});
                    }
                }
            } else {
                //brute force classification, try everything in the provided classifiers and optimizers
                for (String cmd : Main.regressionBruteForceCommands) {
                    String classifier = cmd.split(" ")[0];
                    String options = cmd.replace(classifier, "");
                    addRegressionToQueue(classifier, options);
                }
            }

            //ATTRIBUTE SELECTION for regression
            if (!Main.noFeatureSelection) {
                System.out.print("Selecting attributes and ranking by RelieFF...");
                if (new File(featureSelectionFile).exists() && Main.resumeTraining) {
                    System.out.print("Selecting attributes and ranking by RelieFF already done... skipped by resumeTraining");
                } else {
                    weka.attributeSelectionByRelieFFAndSaveToCSV(featureSelectionFile);

                    System.out.println("[done]");

                    //reset arff and keep compatible header
                    weka.setCSVFile(new File(featureSelectionFile));
                    weka.csvToArff(isClassification);
                    weka.makeCompatibleARFFheaders(dataToTrainModel.replace("data_to_train.csv", "data_to_train.arff"),
                            featureSelectionFile.replace("RELIEFF.csv", "RELIEFF.arff"));
                    weka.setARFFfile(featureSelectionFile.replace("RELIEFF.csv", "RELIEFF.arff"));
                    weka.setDataFromArff();
                }
            }

        }
        //skip the models with an unknown optimizer: the stepwise searches
        //would fail, and the topX and all searches would be labelled with it
        HashMap<String, String> hmUnknownOptimizers = new HashMap<>();
        for (int i = 0; i < alClassifiers.size(); i++) {
            String optimizer = alClassifiers.get(i)[2];
            if (!KNOWN_OPTIMIZERS.contains(optimizer.toLowerCase().trim())) {
                hmUnknownOptimizers.put(optimizer, "");
                alClassifiers.remove(i);
                i--;
            }
        }
        for (String unknownOptimizer : hmUnknownOptimizers.keySet()) {
            System.err.println("[warning] Unknown optimizer " + unknownOptimizer
                    + ": the models using it are skipped. Available optimizers: " + KNOWN_OPTIMIZERS);
        }

        //resume training, remove from alClassifier all classifiers already trained
        if (Main.resumeTraining && !Main.restoreRun) {
            HashMap<String, String> hm = new HashMap<>();
            //get data
            try {
                BufferedReader br = new BufferedReader(new FileReader(resultsFile));
                br.readLine(); //skip header
                while (br.ready()) {
                    String line[] = br.readLine().split("\t");
                    if (!line[0].startsWith("ERROR")) {
                        String s[] = new String[4];
                        s[0] = line[1];
                        s[1] = line[2];
                        s[2] = line[3].toLowerCase();
                        s[3] = line[4].toLowerCase();
                        hm.put(line[1] + "\t" + line[2] + "\t" + line[3].toLowerCase() + "\t" + line[4].toLowerCase(), "");
                    }
                }
            } catch (Exception e) {
                if (Main.debug) {
                    e.printStackTrace();
                }
            }
            //remove from alClassifiers
            int alClassifiersBeforeRemoval = alClassifiers.size();
            for (int i = 0; i < alClassifiers.size(); i++) {
                String s = alClassifiers.get(i)[0] + "\t" + alClassifiers.get(i)[1] + "\t" + alClassifiers.get(i)[2] + "\t" + alClassifiers.get(i)[3];
                if (hm.containsKey(s)) {
                    alClassifiers.remove(i);
                    i--; // the next item is now at index i
                }
            }
            int alClassifiersAfterRemoval = alClassifiers.size();
            int totalRemoved = alClassifiersBeforeRemoval - alClassifiersAfterRemoval;
            if (Main.debug) {
                System.out.println("Total removed from alClassifier after resumeTraining = " + totalRemoved);
            }

            System.out.println("ResumeTraining: Remains " + alClassifiersAfterRemoval
                    + " classifiers to train on the " + alClassifiersBeforeRemoval);

        }

        try {
            //PREPARE OUTPUT
            if (Main.resumeTraining && !Main.restoreRun) {
                pw = new PrintWriter(new FileWriter(resultsFile, true));
                //pw.println("Resumed here");
                pw.flush();
            } else {
                pw = new PrintWriter(new FileWriter(resultsFile));
                pw.println(resultsSummaryHeader);
            }

            //EXECUTE IN PARRALLEL
            System.out.println("Total classifiers to test: " + alClassifiers.size());

            // models that worked, to detect a run where all of them failed
            AtomicInteger trainedModels = new AtomicInteger();
            if (parrallel) {
                // results are written as soon as each model is done (so that an
                // interrupted run can be resumed), then the file is rewritten in
                // the order of the queue, so that it does not depend on thread timing
                String[] outputs = new String[alClassifiers.size()];
                IntStream.range(0, alClassifiers.size()).parallel().forEach((i) -> {
                    String[] classif = alClassifiers.get(i);
                    String s;
                    if (isNotThreadSafe(classif[0], classif[1])) {
                        // trained one at a time (other models still run in parallel)
                        synchronized (NOT_THREAD_SAFE_LOCK) {
                            s = StepWiseFeatureSelectionTraining(classif[0], classif[1], classif[2], classif[3]);
                        }
                    } else {
                        s = StepWiseFeatureSelectionTraining(classif[0], classif[1], classif[2], classif[3]);
                    }
                    if (s == null) {
                        s = "ERROR\t" + classif[0] + " " + classif[1] + " | " + classif[3] + " | no result";
                    }
                    if (!isFailedModel(s)) {
                        trainedModels.incrementAndGet();
                    }
                    if (!isFailedModel(s) || Main.printFailedModels) {
                        outputs[i] = s;
                        synchronized (pw) {
                            pw.println(s);
                            pw.flush();
                        }
                    }
                });
                pw.close();
                sortResultsFile(resultsFile, outputs);
            } else {
                alClassifiers.stream().map((classif) -> {
                    String s = StepWiseFeatureSelectionTraining(classif[0], classif[1], classif[2], classif[3]);
                    if (s == null) {
                        s = "ERROR\t" + classif[0] + " " + classif[1] + " | " + classif[3] + " | no result";
                    }
                    if (!isFailedModel(s)) {
                        trainedModels.incrementAndGet();
                        pw.println(s);
                    } else if (Main.printFailedModels) {
                        pw.println(s);
                    }
                    return classif;
                }).forEach((_item) -> {
                    pw.flush();
                });
            }
            pw.close();

            //END
            System.out.println("Total model tested: " + cptPassed + "/" + alClassifiers.size()
                    + ", including " + cptFailed + " incompatible models");
            if (!alClassifiers.isEmpty() && trainedModels.get() == 0) {
                System.err.println("[error] None of the " + alClassifiers.size() + " models could be trained");
                Main.exitCode = 1;
            }

        } catch (Exception e) {
            if (Main.debug) {
                e.printStackTrace();
            }
            System.err.println("[error] Training stopped: " + e);
            Main.exitCode = 1;
            System.out.println("Total model tested: " + cptPassed + "/" + alClassifiers.size()
                    + ", including " + cptFailed + " incompatible models");
            pw.close();

        } catch (Error err) {
            if (Main.debug) {
                err.printStackTrace();
            }
            System.err.println("[error] Training stopped: " + err);
            Main.exitCode = 1;
            System.out.println("Total model tested: " + cptPassed + "/" + alClassifiers.size()
                    + ", including " + cptFailed + " incompatible models");
            pw.close();
        }
    }

    /**
     * @param output output line of StepWiseFeatureSelectionTraining
     * @return true if the model failed. Failed models start with ERROR (the
     * output of a model that worked starts with its ID, and may contain
     * "error" in its options, e.g. weka.classifiers.functions.loss.SquaredError)
     */
    static boolean isFailedModel(String output) {
        return output.startsWith("ERROR");
    }

    private static final Object NOT_THREAD_SAFE_LOCK = new Object();

    /**
     * Weka classifiers that share static state between instances: two of them
     * trained at the same time fail (CHIRP: NullPointerException in
     * weka.classifiers.misc.chirp), so that the models kept depended on the
     * thread timing.
     *
     * @param classifier
     * @param options
     * @return true if the classifier must not be trained in parallel with
     * another model using it
     */
    static boolean isNotThreadSafe(String classifier, String options) {
        return (classifier + " " + options).contains("CHIRP");
    }

    /**
     * print which class is used as positive class for SEN, SPE, PPV, etc.
     *
     * @param data
     */
    public static void printPositiveClass(Instances data) {
        try {
            Attribute classAttribute = data.attribute(data.numAttributes() - 1);
            if (classAttribute.isNominal() && classAttribute.numValues() == 2) {
                String positive = classAttribute.value(Weka_module.getPositiveClassIndex(classAttribute));
                if (!Main.positiveClass.trim().isEmpty() && Weka_module.indexOfPositiveClassOption(classAttribute) < 0) {
                    System.err.println("[warning] positiveClass " + Main.positiveClass
                            + " is not a value of the class " + classAttribute + ": " + positive + " is used instead");
                }
                if (Weka_module.isPositiveClassArbitrary(classAttribute)) {
                    System.err.println("[warning] positiveClass is not set and the class has no value true or 1: "
                            + positive + " (the first class value) is used as positive class for SEN, SPE, FPR, FNR, "
                            + "PPV, FDR, Fscore and AUPRC. This choice is arbitrary: set positiveClass in the config file");
                }
                System.out.println("Positive class (for SEN, SPE, FPR, FNR, PPV, FDR, Fscore, AUPRC): "
                        + positive + ". Use the positiveClass option to choose it.");
            } else if (classAttribute.isNominal()) {
                System.out.println("More than 2 classes: SEN, SPE, FPR, FNR, PPV, FDR, Fscore and AUPRC "
                        + "are averages weighted by class size");
            }
        } catch (Exception e) {
            if (Main.debug) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Rewrite the results file with the models of this run in the order of
     * the queue. Lines that were already in the file before this run (resumed
     * training) are kept first, in their original order.
     *
     * @param resultsFile
     * @param outputs output line of each model of the queue (null if not
     * written)
     */
    private static void sortResultsFile(String resultsFile, String[] outputs) {
        try {
            HashMap<String, String> hmThisRun = new HashMap<>();
            for (String output : outputs) {
                if (output != null) {
                    hmThisRun.put(output, "");
                }
            }
            ArrayList<String> lines = new ArrayList<>();
            BufferedReader br = new BufferedReader(new FileReader(resultsFile));
            String line;
            while ((line = br.readLine()) != null) {
                if (!hmThisRun.containsKey(line)) {
                    lines.add(line);
                }
            }
            br.close();
            for (String output : outputs) {
                if (output != null) {
                    lines.add(output);
                }
            }
            PrintWriter pwSorted = new PrintWriter(new FileWriter(resultsFile));
            for (String l : lines) {
                pwSorted.println(l);
            }
            pwSorted.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * StepWise training. We add the features one by one in the model if they
     * improve the value to maximize We only test the first 500 attributes (or
     * less if the number of attributes is less than 500)
     *
     * @param classifier
     * @param classifier_options
     * @param valueToMaximizeOrMinimize
     * @return
     */
    private static String StepWiseFeatureSelectionTraining(String classifier, String classifier_options,
            String valueToMaximizeOrMinimize, String searchMethod) {
        String out = classifier + "\t" + classifier_options + "\t" + valueToMaximizeOrMinimize.toUpperCase() + "\t" + searchMethod;
        Instant start = Instant.now();
        System.out.println("[model] (" + cptPassed.getAndIncrement() + "/" + alClassifiers.size() + ")" + out);
        String lastOutput = "";
        // models are trained in parallel and DecimalFormat is not thread-safe
        DecimalFormat df = utils.newDecimalFormat();

        boolean minimize = isMinimizedOptimizer(valueToMaximizeOrMinimize);

        try {
            Object o = null;
            double previousMeasureToMaximize = -1000.0;
            double previousMeasureToMinimize = 1000.0;
            ArrayList<Integer> alAttributes = new ArrayList<>();
            int classIndex = weka.myData.numAttributes();
            for (int i = 1; i <= classIndex; i++) { // weka starts at 1, not 0
                alAttributes.add(i);
            }
            AttributeOject ao = new AttributeOject(alAttributes);
            int cpt = 0;
            Weka_module.ClassificationResultsObject cr = null;
            Weka_module.RegressionResultsObject rr = null;

            // SHORT TEST to ensure compatibility of the model
            if (Main.performShortTest) {
                o = weka.shortTestTrainClassifier(classifier, classifier_options,
                        ao.getAttributesIdClassInString(), isClassification);
                if (o.getClass().getName().equals("java.lang.String")) {
                    if (Main.debug) {
                        System.out.println("SHORT TEST FAILED");
                    }
                    return "ERROR\t" + o.toString();
                }
            }
            if (Main.debug) {
                System.out.println("\tGoing to training");
            }

            //all features search, no feature selection
            if (searchMethod.startsWith("all")) {
                int top = ao.alAttributes.size();
                //Create attribute list
                for (int i = 0; i < top; i++) { //from ID to class (excluded)
                    //add new attribute to the set of retainedAttributes
                    ao.addNewAttributeToRetainedAttributes(i);
                }
                //TRAIN 10CV
                o = weka.trainClassifier(classifier, classifier_options,
                        ao.getRetainedAttributesIdClassInString(), isClassification, 10);

                if (!(o instanceof String) || o == null) {
                    if (isClassification) {
                        cr = (Weka_module.ClassificationResultsObject) o;
                    } else {
                        rr = (Weka_module.RegressionResultsObject) o;
                    }
                }
            }

            //TOP X features search
            if (searchMethod.startsWith("top")) {
                int top = Integer.valueOf(searchMethod.replace("top", ""));
                if (top <= ao.alAttributes.size()) {
                    //Create attribute list
                    for (int i = 0; i < top; i++) { //from ID to class (excluded)
                        //add new attribute to the set of retainedAttributes
                        ao.addNewAttributeToRetainedAttributes(i);
                    }
                    //TRAIN 10CV
                    o = weka.trainClassifier(classifier, classifier_options,
                            ao.getRetainedAttributesIdClassInString(), isClassification, 10);

                    if (!(o instanceof String) || o == null) {
                        if (isClassification) {
                            cr = (Weka_module.ClassificationResultsObject) o;
                        } else {
                            rr = (Weka_module.RegressionResultsObject) o;
                        }
                    }
                } else {
                    o = null;
                }
            }

            //FORWARD AND FORWARD-BACKWARD search AND BACKWARD AND BACKWARD-FORWARD search
            if (searchMethod.startsWith("f") || searchMethod.startsWith("b")) {
                boolean doForwardBackward_OR_BackwardForward = searchMethod.equals("fb") || searchMethod.equals("bf");
                boolean StartBackwardInsteadForward = searchMethod.startsWith("b");
                if (StartBackwardInsteadForward) {
                    Collections.reverse(ao.alAttributes);
                }

                for (int i = 0; i < ao.alAttributes.size(); i++) { //from ID to class (excluded)
                    // the model is full: no other feature can be added
                    if (ao.retainedAttributesOnly.size() >= Main.maxNumberOfFeaturesInModel) {
                        break;
                    }
                    cpt++;
                    //add new attribute to the set of retainedAttributes
                    ao.addNewAttributeToRetainedAttributes(i);
                    Weka_module.ClassificationResultsObject oldcr = cr;
                    Weka_module.RegressionResultsObject oldrr = rr;
                    //do feature selection by forward(-backward)
                    o = weka.trainClassifier(classifier, classifier_options,
                            ao.getRetainedAttributesIdClassInString(), isClassification, 10);

                    if (o == null) {
                        return "ERROR\t" + classifier + " " + classifier_options + " | " + searchMethod + " | training failed";
                    } else if (o instanceof String) {
                        return "ERROR\t" + o;
                    } else if (isClassification) {
                        cr = (Weka_module.ClassificationResultsObject) o;
                    } else {
                        rr = (Weka_module.RegressionResultsObject) o;
                    }

                    //choose what we want to maximize or minimize (such as error rates)
                    double currentMeasure = getValueToMaximize(valueToMaximizeOrMinimize, cr, rr);

                    //Report results
                    boolean modelIsImproved = false;
                    //i<2 is to avoid return no attribute at all
                    // test if model is improved
                    if (minimize) {
                        modelIsImproved = (i < 2 && currentMeasure <= previousMeasureToMinimize) || currentMeasure < previousMeasureToMinimize;
                    } else {
                        modelIsImproved = (i < 2 && currentMeasure >= previousMeasureToMaximize) || currentMeasure > previousMeasureToMaximize;
                    }

                    if (modelIsImproved) {
                        // do backward OR forward, check if we have an improvement if we remove previously chosen features
                        // (all of them except the one we just added)
                        if (doForwardBackward_OR_BackwardForward && ao.retainedAttributesOnly.size() > 1) {
                            int j = 0;
                            while (j < ao.retainedAttributesOnly.size() - 1) {
                                ArrayList<Integer> attributesToTestInBackward = new ArrayList<>(ao.retainedAttributesOnly);
                                attributesToTestInBackward.remove(j);
                                String featuresToTest = ao.ID + "," + utils.arrayToString(attributesToTestInBackward, ",") + "," + ao.Class;
                                //train
                                Object ob = weka.trainClassifier(classifier, classifier_options,
                                        featuresToTest, isClassification, 10);
                                if (ob == null || ob instanceof String) {
                                    // cannot evaluate the model without this feature: keep it
                                    j++;
                                    continue;
                                }
                                Weka_module.ClassificationResultsObject crBackward = null;
                                Weka_module.RegressionResultsObject rrBackward = null;
                                if (isClassification) {
                                    crBackward = (Weka_module.ClassificationResultsObject) ob;
                                } else {
                                    rrBackward = (Weka_module.RegressionResultsObject) ob;
                                }
                                //get measure
                                double measureWithRemovedFeature = getValueToMaximize(valueToMaximizeOrMinimize, crBackward, rrBackward);
                                //check if we have improvement (or the same performance with one feature less)
                                boolean removalIsBetter;
                                if (minimize) {
                                    removalIsBetter = measureWithRemovedFeature <= currentMeasure;
                                } else {
                                    removalIsBetter = measureWithRemovedFeature >= currentMeasure;
                                }

                                if (removalIsBetter) {
                                    //definitively discard the feature. The next one is now at index j
                                    ao.retainedAttributesOnly = attributesToTestInBackward;
                                    cr = crBackward;
                                    rr = rrBackward;
                                    o = ob;
                                    currentMeasure = measureWithRemovedFeature;
                                } else {
                                    //keep the feature, test the next one
                                    j++;
                                }
                            }
                        }
                        if (!minimize) {
                            previousMeasureToMaximize = currentMeasure;
                        } else {
                            previousMeasureToMinimize = currentMeasure;
                        }

                        //modify results summary output
                        //only for DEBUG purposes
                        if (isClassification) {
                            lastOutput = out
                                    + "\t" + cr.numberOfFeatures + "\t" + cr.toString() + "\t" + ao.getRetainedAttributesIdClassInString();

                        } else {
                            lastOutput = out
                                    + "\t" + rr.numberOfFeatures + "\t" + rr.toString() + "\t" + ao.getRetainedAttributesIdClassInString();
                        }

                    } else {
                        //back to previous attribute if no improvement with the new attribute and go to the next
                        ao.retainedAttributesOnly.remove(ao.retainedAttributesOnly.size() - 1);
                        cr = oldcr;
                        rr = oldrr;
                    }
                }
                // no feature could be retained (e.g. undefined measure)
                if ((isClassification && cr == null) || (!isClassification && rr == null)) {
                    o = null;
                }
            }

            //now we have a good model, evaluate the performance using various approaches
            if (!(o instanceof String) && o != null) {
                //LOOCV
                Weka_module.ClassificationResultsObject crLoocv = null;
                Weka_module.RegressionResultsObject rrLoocv = null;
                String loocvOut = "";
                if (Main.loocv) {
                    if (Main.debug) {
                        System.out.println("\tLOOCV Train set " + Main.bootstrapAndRepeatedHoldoutFolds + " times");
                    }
                    Object oLoocv;
                    if (isClassification) {
                        oLoocv = weka.trainClassifier(classifier, classifier_options,
                                ao.getRetainedAttributesIdClassInString(), isClassification,
                                cr.dataset.numInstances());
                    } else {
                        oLoocv = weka.trainClassifier(classifier, classifier_options,
                                ao.getRetainedAttributesIdClassInString(), isClassification,
                                rr.dataset.numInstances());
                    }

                    if (oLoocv instanceof String || oLoocv == null) {
                        if (Main.debug) {
                            System.err.println("[error] LOOCV failed");
                        }
                        // empty columns, to keep the following columns aligned
                        if (isClassification) {
                            loocvOut = "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t";
                        } else {
                            loocvOut = "\t" + "\t" + "\t" + "\t";
                        }
                    } else if (isClassification) {
                        crLoocv = (Weka_module.ClassificationResultsObject) oLoocv;
                        loocvOut = crLoocv.toStringShort();
                    } else {
                        rrLoocv = (Weka_module.RegressionResultsObject) oLoocv;
                        loocvOut = rrLoocv.toString();
                    }

                } else {
                    if (isClassification) {
                        loocvOut = "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t";
                    } else {
                        loocvOut = "\t" + "\t" + "\t" + "\t";
                    }
                }
                //Repeated Holdout
                Weka_module.evaluationPerformancesResultsObject eproRHTrain = new Weka_module.evaluationPerformancesResultsObject();
                if (Main.repeatedHoldout) {
                    if (Main.debug) {
                        System.out.println("\tRepeated Holdout on Train set " + Main.bootstrapAndRepeatedHoldoutFolds + " times");
                    }
                    if (isClassification) {
                        for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                            Weka_module.ClassificationResultsObject cro
                                    = (Weka_module.ClassificationResultsObject) weka.trainClassifierHoldOutValidation(classifier, classifier_options,
                                            ao.getRetainedAttributesIdClassInString(), isClassification, i);
                            if (cro == null) {
                                continue; // failed repetition
                            }
                            eproRHTrain.alAUCs.add(Double.valueOf(cro.AUC));
                            eproRHTrain.alpAUCs.add(Double.valueOf(cro.pAUC));
                            eproRHTrain.alAUPRCs.add(Double.valueOf(cro.AUPRC));
                            eproRHTrain.alACCs.add(Double.valueOf(cro.ACC));
                            eproRHTrain.alSEs.add(Double.valueOf(cro.TPR));
                            eproRHTrain.alSPs.add(Double.valueOf(cro.TNR));
                            eproRHTrain.alMCCs.add(Double.valueOf(cro.MCC));
                            eproRHTrain.alMAEs.add(Double.valueOf(cro.MAE));
                            eproRHTrain.alBERs.add(Double.valueOf(cro.BER));
                        }
                    } else {
                        for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                            Weka_module.RegressionResultsObject rro
                                    = (Weka_module.RegressionResultsObject) weka.trainClassifierHoldOutValidation(classifier, classifier_options,
                                            ao.getRetainedAttributesIdClassInString(), isClassification, i);
                            if (rro == null) {
                                continue; // failed repetition
                            }
                            eproRHTrain.alCCs.add(Double.valueOf(rro.CC));
                            eproRHTrain.alMAEs.add(Double.valueOf(rro.MAE));
                            eproRHTrain.alRMSEs.add(Double.valueOf(rro.RMSE));
                            eproRHTrain.alRAEs.add(Double.valueOf(rro.RAE));
                            eproRHTrain.alRRSEs.add(Double.valueOf(rro.RRSE));
                        }
                    }
                    eproRHTrain.computeMeans();
                }

                //BOOTSTRAP AND BOOTSTRAP .632+ rule TRAIN
                if (Main.debug) {
                    System.out.println("\tBootstrapping on Train set " + Main.bootstrapAndRepeatedHoldoutFolds + " times");
                }
                double bootstrapTrain632plus = 1000;
                Weka_module.evaluationPerformancesResultsObject eproBSTrain
                        = new Weka_module.evaluationPerformancesResultsObject();
                if (Main.bootstrap) {
                    if (isClassification) {
                        bootstrapTrain632plus = weka.trainClassifierBootstrap632plus(classifier, classifier_options,
                                ao.getRetainedAttributesIdClassInString());
                        for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                            Weka_module.ClassificationResultsObject cro
                                    = (Weka_module.ClassificationResultsObject) weka.trainClassifierBootstrap(classifier, classifier_options,
                                            ao.getRetainedAttributesIdClassInString(), isClassification, i);
                            if (cro == null) {
                                continue; // failed repetition
                            }
                            eproBSTrain.alAUCs.add(Double.valueOf(cro.AUC));
                            eproBSTrain.alpAUCs.add(Double.valueOf(cro.pAUC));
                            eproBSTrain.alAUPRCs.add(Double.valueOf(cro.AUPRC));
                            eproBSTrain.alACCs.add(Double.valueOf(cro.ACC));
                            eproBSTrain.alSEs.add(Double.valueOf(cro.TPR));
                            eproBSTrain.alSPs.add(Double.valueOf(cro.TNR));
                            eproBSTrain.alMCCs.add(Double.valueOf(cro.MCC));
                            eproBSTrain.alMAEs.add(Double.valueOf(cro.MAE));
                            eproBSTrain.alBERs.add(Double.valueOf(cro.BER));
                        }
                    } else {
                        for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                            Weka_module.RegressionResultsObject rro
                                    = (Weka_module.RegressionResultsObject) weka.trainClassifierBootstrap(classifier, classifier_options,
                                            ao.getRetainedAttributesIdClassInString(), isClassification, i);
                            if (rro == null) {
                                continue; // failed repetition
                            }

                            eproBSTrain.alCCs.add(Double.valueOf(rro.CC));
                            eproBSTrain.alMAEs.add(Double.valueOf(rro.MAE));
                            eproBSTrain.alRMSEs.add(Double.valueOf(rro.RMSE));
                            eproBSTrain.alRAEs.add(Double.valueOf(rro.RAE));
                            eproBSTrain.alRRSEs.add(Double.valueOf(rro.RRSE));
                        }
                    }
                    eproBSTrain.computeMeans();
                }

                // TEST SET
                if (Main.debug) {
                    System.out.println("\tEvalutation on Test set ");
                }
                Weka_module.testResultsObject tro = new Weka_module.testResultsObject();
                String testResults = null;
                if (isClassification) {
                    testResults = "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t";
                } else {
                    testResults = "\t" + "\t" + "\t" + "\t";
                }
                try {
                    if (Main.doSampling) {
                        //testing
                        Weka_module weka2 = new Weka_module();
                        if (isClassification) {
                            Weka_module.ClassificationResultsObject cr2 = null;
                            try {
                                cr2 = (Weka_module.ClassificationResultsObject) weka2.testClassifierFromModel(cr.model,
                                        trainFileName.replace("data_to_train.csv", "data_to_test.csv"),//test file
                                        Main.isClassification, out);
                            } catch (Exception e) {
                                if (Main.debug) {
                                    e.printStackTrace();
                                }
                            }
                            tro.ACC = cr2.ACC;
                            tro.AUC = cr2.AUC;
                            tro.AUPRC = cr2.AUPRC;
                            tro.SE = cr2.TPR;
                            tro.SP = cr2.TNR;
                            tro.MCC = cr2.MCC;
                            tro.MAE = cr2.MAE;
                            tro.BER = cr2.BER;
                            testResults = tro.toStringClassification();

                        } else {
                            Weka_module.RegressionResultsObject rr2
                                    = (Weka_module.RegressionResultsObject) weka2.testClassifierFromModel(rr.model,
                                            trainFileName.replace("data_to_train.csv", "data_to_test.csv"),//test file
                                            Main.isClassification, out);
                            tro.CC = rr2.CC;
                            tro.RRSE = rr2.RRSE;
                            tro.RMSE = rr2.RMSE;
                            tro.MAE = rr2.MAE;
                            tro.RAE = rr2.RAE;
                            testResults = tro.toStringRegression();
                        }
                    }
                } catch (Exception e) {
                    if (Main.debug) {
                        e.printStackTrace();
                    }
                }

                //REPEATED HOLDOUT TRAIN_TEST
                if (Main.debug) {
                    System.out.println("\tRepeated Holdout on Train AND Test set "
                            + Main.bootstrapAndRepeatedHoldoutFolds + " times");
                }
                Weka_module.evaluationPerformancesResultsObject eproRHTrainTest
                        = new Weka_module.evaluationPerformancesResultsObject();
                try {
                    if (Main.doSampling && Main.repeatedHoldout) {
                        Weka_module weka2 = new Weka_module();
                        weka2.setARFFfile(trainFileName.replace("data_to_train.csv", "all_data.arff"));
                        weka2.setDataFromArff();
                        if (isClassification) {
                            weka2.myData = weka2.extractFeaturesFromDatasetBasedOnModel(cr.model, weka2.myData);
                            for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                                Weka_module.ClassificationResultsObject cro
                                        = (Weka_module.ClassificationResultsObject) weka2.trainClassifierHoldOutValidation(
                                                classifier, classifier_options,
                                                null, isClassification, i);
                                if (cro == null) {
                                    continue; // failed repetition
                                }

                                eproRHTrainTest.alAUCs.add(Double.valueOf(cro.AUC));
                                eproRHTrainTest.alpAUCs.add(Double.valueOf(cro.pAUC));
                                eproRHTrainTest.alAUPRCs.add(Double.valueOf(cro.AUPRC));
                                eproRHTrainTest.alACCs.add(Double.valueOf(cro.ACC));
                                eproRHTrainTest.alSEs.add(Double.valueOf(cro.TPR));
                                eproRHTrainTest.alSPs.add(Double.valueOf(cro.TNR));
                                eproRHTrainTest.alMCCs.add(Double.valueOf(cro.MCC));
                                eproRHTrainTest.alMAEs.add(Double.valueOf(cro.MAE));
                                eproRHTrainTest.alBERs.add(Double.valueOf(cro.BER));
                            }
                        } else {
                            weka2.myData = weka2.extractFeaturesFromDatasetBasedOnModel(rr.model, weka2.myData);
                            for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                                Weka_module.RegressionResultsObject rro
                                        = (Weka_module.RegressionResultsObject) weka2.trainClassifierHoldOutValidation(
                                                classifier, classifier_options,
                                                null, isClassification, i);
                                if (rro == null) {
                                    continue; // failed repetition
                                }
                                eproRHTrainTest.alCCs.add(Double.valueOf(rro.CC));
                                eproRHTrainTest.alMAEs.add(Double.valueOf(rro.MAE));
                                eproRHTrainTest.alRMSEs.add(Double.valueOf(rro.RMSE));
                                eproRHTrainTest.alRAEs.add(Double.valueOf(rro.RAE));
                                eproRHTrainTest.alRRSEs.add(Double.valueOf(rro.RRSE));
                            }
                        }
                        eproRHTrainTest.computeMeans();
                    }
                } catch (Exception e) {
                    if (Main.debug) {
                        e.printStackTrace();
                    }
                }

                //BOOTSTRAP TRAIN_TEST AND BOOTSTRAP .632+ rule TRAIN_TEST
                if (Main.debug) {
                    System.out.println("\tBootstrapping on Train AND Test set " + Main.bootstrapAndRepeatedHoldoutFolds + " times");
                }
                double bootstrapTrainTest632plus = 1000;
                Weka_module.evaluationPerformancesResultsObject eproBSTrainTest = new Weka_module.evaluationPerformancesResultsObject();
                try {
                    if (Main.doSampling && Main.bootstrap) {
                        Weka_module weka2 = new Weka_module();
                        weka2.setARFFfile(trainFileName.replace("data_to_train.csv", "all_data.arff"));
                        weka2.setDataFromArff();
                        if (isClassification) {
                            weka2.myData = weka2.extractFeaturesFromDatasetBasedOnModel(cr.model, weka2.myData);
                            bootstrapTrainTest632plus = weka2.trainClassifierBootstrap632plus(classifier, classifier_options,
                                    null);
                            for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                                Weka_module.ClassificationResultsObject cro
                                        = (Weka_module.ClassificationResultsObject) weka2.trainClassifierBootstrap(
                                                classifier, classifier_options,
                                                null, isClassification, i);
                                if (cro == null) {
                                    continue; // failed repetition
                                }

                                eproBSTrainTest.alAUCs.add(Double.valueOf(cro.AUC));
                                eproBSTrainTest.alpAUCs.add(Double.valueOf(cro.pAUC));
                                eproBSTrainTest.alAUPRCs.add(Double.valueOf(cro.AUPRC));
                                eproBSTrainTest.alACCs.add(Double.valueOf(cro.ACC));
                                eproBSTrainTest.alSEs.add(Double.valueOf(cro.TPR));
                                eproBSTrainTest.alSPs.add(Double.valueOf(cro.TNR));
                                eproBSTrainTest.alMCCs.add(Double.valueOf(cro.MCC));
                                eproBSTrainTest.alMAEs.add(Double.valueOf(cro.MAE));
                                eproBSTrainTest.alBERs.add(Double.valueOf(cro.BER));
                            }
                        } else {
                            weka2.myData = weka2.extractFeaturesFromDatasetBasedOnModel(rr.model, weka2.myData);
                            for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                                Weka_module.RegressionResultsObject rro
                                        = (Weka_module.RegressionResultsObject) weka2.trainClassifierBootstrap(
                                                classifier, classifier_options,
                                                null, isClassification, i);
                                if (rro == null) {
                                    continue; // failed repetition
                                }
                                eproBSTrainTest.alCCs.add(Double.valueOf(rro.CC));
                                eproBSTrainTest.alMAEs.add(Double.valueOf(rro.MAE));
                                eproBSTrainTest.alRMSEs.add(Double.valueOf(rro.RMSE));
                                eproBSTrainTest.alRAEs.add(Double.valueOf(rro.RAE));
                                eproBSTrainTest.alRRSEs.add(Double.valueOf(rro.RRSE));
                            }
                        }
                        eproBSTrainTest.computeMeans();
                    }
                } catch (Exception e) {
                    if (Main.debug) {
                        e.printStackTrace();
                    }
                }

                //STATISTICS AND OUTPUT
                if (isClassification) {
                    //average BER
                    List<Double> alBERs = new ArrayList<>();
                    try {
                        alBERs.add(Double.valueOf(cr.BER));
                    } catch (Exception e) {
                    }
                    try {
                        alBERs.add(Double.valueOf(eproRHTrain.meanBERs));
                    } catch (Exception e) {
                    }
                    try {
                        alBERs.add(Double.valueOf(eproBSTrain.meanBERs));
                    } catch (Exception e) {
                    }
                    try {
                        alBERs.add(Double.valueOf(crLoocv.BER));
                    } catch (Exception e) {
                    }
                    try {
                        alBERs.add(Double.valueOf(tro.BER));
                    } catch (Exception e) {
                    }
                    try {
                        alBERs.add(Double.valueOf(eproRHTrainTest.meanBERs));
                    } catch (Exception e) {
                    }
                    try {
                        alBERs.add(Double.valueOf(eproBSTrainTest.meanBERs));
                    } catch (Exception e) {
                    }

                    double BERs[] = utils.arrayToDoubleWithoutNaN(alBERs);

                    StandardDeviation std = new StandardDeviation();
                    std.setData(BERs);
                    double StdBER = std.evaluate();

                    Mean m = new Mean();
                    m.setData(BERs);
                    double averageBER = m.evaluate();

                    //average MAE
                    List<Double> alMAEs = new ArrayList<>();
                    try {
                        alMAEs.add(Double.valueOf(cr.MAE));
                    } catch (Exception e) {
                    }
                    try {
                        alMAEs.add(Double.valueOf(eproRHTrain.meanMAEs));
                    } catch (Exception e) {
                    }
                    try {
                        alMAEs.add(Double.valueOf(eproBSTrain.meanMAEs));
                    } catch (Exception e) {
                    }
                    try {
                        alMAEs.add(Double.valueOf(crLoocv.MAE));
                    } catch (Exception e) {
                    }
                    try {
                        alMAEs.add(Double.valueOf(tro.MAE));
                    } catch (Exception e) {
                    }
                    try {
                        alMAEs.add(Double.valueOf(eproRHTrainTest.meanMAEs));
                    } catch (Exception e) {
                    }
                    try {
                        alMAEs.add(Double.valueOf(eproBSTrainTest.meanMAEs));
                    } catch (Exception e) {
                    }

                    double MAEs[] = utils.arrayToDoubleWithoutNaN(alMAEs);

                    std = new StandardDeviation();
                    std.setData(MAEs);
                    double StdMAE = std.evaluate();

                    m = new Mean();
                    m.setData(MAEs);
                    double averageMAE = m.evaluate();

                    //average MCC
                    List<Double> alMCCs = new ArrayList<>();
                    try {
                        alMCCs.add(Double.valueOf(cr.MCC));
                    } catch (Exception e) {
                    }
                    try {
                        alMCCs.add(Double.valueOf(eproRHTrain.meanMCCs));
                    } catch (Exception e) {
                    }
                    try {
                        alMCCs.add(Double.valueOf(eproBSTrain.meanMCCs));
                    } catch (Exception e) {
                    }
                    try {
                        alMCCs.add(Double.valueOf(crLoocv.MCC));
                    } catch (Exception e) {
                    }
                    try {
                        alMCCs.add(Double.valueOf(tro.MCC));
                    } catch (Exception e) {
                    }
                    try {
                        alMCCs.add(Double.valueOf(eproRHTrainTest.meanMCCs));
                    } catch (Exception e) {
                    }
                    try {
                        alMCCs.add(Double.valueOf(eproBSTrainTest.meanMCCs));
                    } catch (Exception e) {
                    }

                    double MCCs[] = utils.arrayToDoubleWithoutNaN(alMCCs);

                    std = new StandardDeviation();
                    std.setData(MCCs);
                    double StdMCC = std.evaluate();

                    m = new Mean();
                    m.setData(MCCs);
                    double averageMCC = m.evaluate();

                    String stats = df.format(averageBER) + "\t"
                            + df.format(StdBER) + "\t"
                            + df.format(averageMAE) + "\t"
                            + df.format(StdMAE) + "\t"
                            + df.format(averageMCC) + "\t"
                            + df.format(StdMCC);

                    //output
                    String bt632 = df.format(bootstrapTrain632plus);
                    if (bootstrapTrain632plus == 1000) {
                        bt632 = "";
                    }
                    String btt632 = df.format(bootstrapTrainTest632plus);
                    if (bootstrapTrainTest632plus == 1000) {
                        btt632 = "";
                    }

                    lastOutput = out
                            + "\t" + cr.numberOfFeatures
                            + "\t" + cr.toString()
                            + "\t" + loocvOut
                            + "\t" + eproRHTrain.toStringClassification()
                            + "\t" + eproBSTrain.toStringClassification()
                            + "\t" + bt632
                            + "\t" + testResults
                            + "\t" + eproRHTrainTest.toStringClassification()
                            + "\t" + eproBSTrainTest.toStringClassification()
                            + "\t" + btt632
                            + "\t" + stats + "\t" + ao.getRetainedAttributesIdClassInString();
                } else {
                    //statistics

                    //average CC
                    List<Double> alCCs = new ArrayList<>();
                    try {
                        alCCs.add(Double.valueOf(rr.CC));
                    } catch (Exception e) {
                    }
                    try {
                        alCCs.add(Double.valueOf(eproRHTrain.meanCCs));
                    } catch (Exception e) {
                    }
                    try {
                        alCCs.add(Double.valueOf(eproBSTrain.meanCCs));
                    } catch (Exception e) {
                    }
                    try {
                        alCCs.add(Double.valueOf(rrLoocv.CC));
                    } catch (Exception e) {
                    }
                    try {
                        alCCs.add(Double.valueOf(tro.CC));
                    } catch (Exception e) {
                    }
                    try {
                        alCCs.add(Double.valueOf(eproRHTrainTest.meanCCs));
                    } catch (Exception e) {
                    }
                    try {
                        alCCs.add(Double.valueOf(eproBSTrainTest.meanCCs));
                    } catch (Exception e) {
                    }

                    double CCs[] = utils.arrayToDoubleWithoutNaN(alCCs);

                    StandardDeviation std = new StandardDeviation();
                    std.setData(CCs);
                    double StdCC = std.evaluate();

                    Mean m = new Mean();
                    m.setData(CCs);
                    double averageCC = m.evaluate();

                    //average MAE
                    List<Double> alMAEs = new ArrayList<>();
                    try {
                        alMAEs.add(Double.valueOf(rr.MAE));
                    } catch (Exception e) {
                    }
                    try {
                        alMAEs.add(Double.valueOf(eproRHTrain.meanMAEs));
                    } catch (Exception e) {
                    }
                    try {
                        alMAEs.add(Double.valueOf(eproBSTrain.meanMAEs));
                    } catch (Exception e) {
                    }
                    try {
                        alMAEs.add(Double.valueOf(rrLoocv.MAE));
                    } catch (Exception e) {
                    }
                    try {
                        alMAEs.add(Double.valueOf(tro.MAE));
                    } catch (Exception e) {
                    }
                    try {
                        alMAEs.add(Double.valueOf(eproRHTrainTest.meanMAEs));
                    } catch (Exception e) {
                    }
                    try {
                        alMAEs.add(Double.valueOf(eproBSTrainTest.meanMAEs));
                    } catch (Exception e) {
                    }

                    double MAEs[] = utils.arrayToDoubleWithoutNaN(alMAEs);

                    std = new StandardDeviation();
                    std.setData(MAEs);
                    double StdMAE = std.evaluate();

                    m = new Mean();
                    m.setData(MAEs);
                    double averageMAE = m.evaluate();

                    //average RMSE
                    List<Double> alRMSEs = new ArrayList<>();
                    try {
                        alRMSEs.add(Double.valueOf(rr.RMSE));
                    } catch (Exception e) {
                    }
                    try {
                        alRMSEs.add(Double.valueOf(eproRHTrain.meanRMSEs));
                    } catch (Exception e) {
                    }
                    try {
                        alRMSEs.add(Double.valueOf(eproBSTrain.meanRMSEs));
                    } catch (Exception e) {
                    }
                    try {
                        alRMSEs.add(Double.valueOf(rrLoocv.RMSE));
                    } catch (Exception e) {
                    }
                    try {
                        alRMSEs.add(Double.valueOf(tro.RMSE));
                    } catch (Exception e) {
                    }
                    try {
                        alRMSEs.add(Double.valueOf(eproRHTrainTest.meanRMSEs));
                    } catch (Exception e) {
                    }
                    try {
                        alRMSEs.add(Double.valueOf(eproBSTrainTest.meanRMSEs));
                    } catch (Exception e) {
                    }

                    double RMSEs[] = utils.arrayToDoubleWithoutNaN(alRMSEs);

                    std = new StandardDeviation();
                    std.setData(RMSEs);
                    double StdRMSE = std.evaluate();

                    m = new Mean();
                    m.setData(RMSEs);
                    double averageRMSE = m.evaluate();

                    String stats = df.format(averageCC) + "\t"
                            + df.format(StdCC) + "\t"
                            + df.format(averageMAE) + "\t"
                            + df.format(StdMAE) + "\t"
                            + df.format(averageRMSE) + "\t"
                            + df.format(StdRMSE);
                    lastOutput = out
                            + "\t" + rr.numberOfFeatures + "\t" + rr.toString()
                            + "\t" + loocvOut
                            + "\t" + eproRHTrain.toStringRegression()
                            + "\t" + eproBSTrain.toStringRegression()
                            + "\t" + testResults
                            + "\t" + eproRHTrainTest.toStringRegression()
                            + "\t" + eproBSTrainTest.toStringRegression()
                            + "\t" + stats
                            + "\t" + ao.getRetainedAttributesIdClassInString();
                }

                //CREATE ID
                // seeded by the model configuration, so that IDs are reproducible
                Random r = new Random(Main.seed + out.hashCode());
                int randomNumber = r.nextInt(10000 - 10) + 10;
                out = (lastOutput.split("\t")[0] + "_" + lastOutput.split("\t")[2]
                        + "_" + lastOutput.split("\t")[3] + "_" + lastOutput.split("\t")[4] + "_" + lastOutput.split("\t")[5] + "_" + randomNumber);

                // temporary models output
//            String filename = Main.CVfolder + File.separatorChar + out;
//            PrintWriter pw = new PrintWriter(new FileWriter(filename + ".txt"));
//            pw.println("#" + classifier + " " + classifier_options + "\n#Value to maximize or minimize:" + valueToMaximizeOrMinimize);
//            if (isClassification) {
//                pw.print("#ID\tactual\tpredicted\terror\tprobability\n");
//            } else {
//                pw.print("#ID\tactual\tpredicted\terror\n");
//            }
//            pw.print(predictions + "\n\n");
//            pw.print("#Selected Attributes\t(Total attributes:" + numberOfAttributes + ")\n");
//            pw.print(selectedAttributes);
//            pw.close();
//            //save model
//            SerializationHelper.write(filename + ".model", model);
            } else {
                out = "ERROR";
                if (Main.debug) {
                    System.err.println(o);
                }
                if (o == null) {
                    out += "\t " + classifier + " " + classifier_options + " | " + searchMethod + " | Error probably because of number of features inferior to topX";
                }
            }

        } catch (Exception e) {
            out = "ERROR\t" + classifier + " " + classifier_options + " | " + searchMethod + " | " + e.getMessage();
            cptFailed.incrementAndGet();
            if (Main.debug) {
                e.printStackTrace();
            }
        }
        Instant finish = Instant.now();
        long s = Duration.between(start, finish).toMillis() / 1000;
        if (Main.debug) {
            System.out.println("model created in [" + s + "s]");
        }
        return out + "\t" + lastOutput;
    }

    /**
     * get value of interest
     *
     * @param valueWanted
     * @param cr
     * @param rr
     * @return
     */
    private static double getValueToMaximize(String valueWanted, Weka_module.ClassificationResultsObject cr,
            Weka_module.RegressionResultsObject rr) {

        switch (valueWanted.toLowerCase().trim()) {
            //classification
            case "auc":
                return Double.parseDouble(cr.AUC);
            case "pauc":
                return Double.parseDouble(cr.pAUC);
            case "acc":
                return Double.parseDouble(cr.ACC);
            case "sen":
            case "tpr":
                return Double.parseDouble(cr.TPR);
            case "spe":
            case "tnr":
                return Double.parseDouble(cr.TNR);
            case "mcc":
                return Double.parseDouble(cr.MCC);
            case "kappa":
                return Double.parseDouble(cr.kappa);
            case "auprc":
            case "aupcr": //misspelling accepted in previous versions
                return Double.parseDouble(cr.AUPRC);
            case "fscore":
                return Double.parseDouble(cr.Fscore);
            case "precision":
                return Double.parseDouble(cr.precision);
            case "recall":
                return Double.parseDouble(cr.recall);
            case "fdr":
                return Double.parseDouble(cr.FDR);//to minimize
            case "ber":
                return Double.parseDouble(cr.BER);//to minimize
            case "tp+fn":
                double sumOfTP_TN = Double.parseDouble(cr.TPR) + Double.parseDouble(cr.TNR);
                return sumOfTP_TN;
            //regression
            case "cc":
                return Double.parseDouble(rr.CC);
            case "mae":
                return Double.parseDouble(rr.MAE);//to minimize
            case "rmse":
                return Double.parseDouble(rr.RMSE);//to minimize
            case "rae":
                return Double.parseDouble(rr.RAE);//to minimize
            case "rrse":
                return Double.parseDouble(rr.RRSE);//to minimize
        }
        throw new IllegalArgumentException("unknown optimizer " + valueWanted);
    }

    /**
     * optimizers accepted by getValueToMaximize
     */
    static final List<String> KNOWN_OPTIMIZERS = java.util.Arrays.asList(
            "auc", "pauc", "acc", "sen", "tpr", "spe", "tnr", "mcc", "kappa", "auprc", "aupcr",
            "fscore", "precision", "recall", "fdr", "ber", "tp+fn",
            "cc", "mae", "rmse", "rae", "rrse");

    /**
     * @param optimizer
     * @return true if the optimizer is an error rate, to minimize
     */
    private static boolean isMinimizedOptimizer(String optimizer) {
        switch (optimizer.toLowerCase().trim()) {
            case "fdr":
            case "mae":
            case "rmse":
            case "ber":
            case "rae":
            case "rrse":
                return true;
            default:
                return false;
        }
    }

    /**
     * add to hm weka configurations
     * <["misc.VFI", "-B 0.6", "AUC"],"">
     * <["CostSensitiveClassifier", "-cost-matrix "[0.0 ratio; 100.0 0.0]" -S 1 -W weka.classifiers.misc.VFI -- -B 0.6", "AUC"],"">
     *
     * @param classifier
     * @param options
     * @param costSensitive
     */
    private static void addClassificationToQueue(String classifier, String options) {
        String optimizers[] = Main.classificationOptimizers.split(",");
        String searchmodes[] = Main.searchmodes.split(",");
        if (Main.noFeatureSelection) {
            searchmodes = new String[]{"all"};
        } 
//        else if(weka.myData.numAttributes()>500){
//            
//        }
        

        for (String optimizer : optimizers) {//AUC, ACC, SEN, SPE, MCC, TP+FN, kappa            
            for (String searchmode : searchmodes) {//F, FB, B, BF, topk
                alClassifiers.add(new String[]{classifier, options, optimizer.trim(), searchmode.trim()});
            }
        }
        if (Main.metaCostSensitiveClassifier) {
            options = "-cost-matrix \"[0.0 ratio; 100.0 0.0]\" -S 1 -W weka.classifiers." + classifier + " -- " + options;
            for (String value : optimizers) {
                alClassifiers.add(new String[]{"meta.CostSensitiveClassifier", options, value.trim(), "f"});
                alClassifiers.add(new String[]{"meta.CostSensitiveClassifier", options, value.trim(), "bf"});
            }
        }
    }

    private static void addRegressionToQueue(String classifier, String options) {
        //String valuesToMaximizeOrMinimize[] = new String[]{"CC", "MAE", "RMSE", "RAE", "RRSE"};
        String optimizers[] = Main.regressionOptimizers.split(",");
        String searchmodes[] = Main.searchmodes.split(",");
        if (Main.noFeatureSelection) {
            searchmodes = new String[]{"all"};
        }
        for (String optimizer : optimizers) {//only CC is maximized here
            for (String searchmode : searchmodes) {
                alClassifiers.add(new String[]{classifier, options, optimizer.trim(), searchmode.trim()});
            }

        }
        if (Main.metaAdditiveRegression) {
            options = "-S 1.0 -I 10 -W weka.classifiers." + classifier + " -- " + options;
            for (String value : optimizers) {
                alClassifiers.add(new String[]{"meta.AdditiveRegression", options, value, "F"});
                alClassifiers.add(new String[]{"meta.AdditiveRegression", options, value, "FB"});
            }
        }
    }

    private static class AttributeOject {

        public Integer ID;
        public ArrayList<Integer> alAttributes = new ArrayList<>();
        public Integer Class;
        public ArrayList<Integer> retainedAttributesOnly = new ArrayList<>();

        public AttributeOject() {

        }

        private AttributeOject(ArrayList<Integer> attributes) {
            ID = attributes.get(0);
            Class = attributes.get(attributes.size() - 1);
            attributes.remove(attributes.size() - 1);
            attributes.remove(0);
            alAttributes = attributes;
        }

        private void addNewAttributeToRetainedAttributes(int index) {
            retainedAttributesOnly.add(alAttributes.get(index));
        }

        private String getRetainedAttributesIdClassInString() {
            String attributes = "";
            attributes += (ID) + ",";
            for (Integer retainedAttribute : retainedAttributesOnly) {
                attributes += (retainedAttribute) + ",";
            }
            attributes += Class;
            return attributes;
        }

        private String getAttributesIdClassInString() {
            String attributes = "";
            attributes += (ID) + ",";
            for (Integer att : alAttributes) {
                attributes += (att) + ",";
            }
            attributes += Class;
            return attributes;
        }
    }

}
