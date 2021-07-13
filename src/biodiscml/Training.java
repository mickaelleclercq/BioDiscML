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
import java.text.DecimalFormatSymbols;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import org.apache.commons.math3.stat.descriptive.moment.*;
import utils.Weka_module;
import utils.utils;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.Classifier;

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
    public static int cptPassed = 0;
    public static int cptFailed = 0;
    public boolean parrallel = true;
    public static String trainFileName = "";
    public static DecimalFormat df = new DecimalFormat();

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

        df.setMaximumFractionDigits(3);
        DecimalFormatSymbols dfs = new DecimalFormatSymbols();
        dfs.setDecimalSeparator('.');
        df.setDecimalFormatSymbols(dfs);
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
                //brute force classification, try everything in the provided classifiers and optimizers
                for (String cmd : Main.classificationBruteForceCommands) {
                    String classifier = cmd.split(" ")[0];
                    String options = cmd.replace(classifier, "").trim();
                    addClassificationToQueue(classifier, options);
                }
            }

            System.out.print("Feature selection and ranking...");
            if (new File(featureSelectionFile).exists() && Main.resumeTraining) {
                System.out.print("\nFeature selection and ranking already done... skipped by resumeTraining");
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
        //resume training, remove from alClassifier all classifiers already trained
        if (Main.resumeTraining) {
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
                }
            }
            int alClassifiersAfterRemoval = alClassifiers.size();
            int totalRemoved = alClassifiersBeforeRemoval - alClassifiersAfterRemoval;
            if (Main.debug) {
                System.out.println("Total removed from alClassifier after resumeTraining = " + totalRemoved);
            }

            System.out.println("ResumeTraining: Remains " + alClassifiersAfterRemoval
                    + "classifiers to train on the " + alClassifiersBeforeRemoval);

        }

        try {
            //PREPARE OUTPUT
            PrintWriter pw;
            if (Main.resumeTraining) {
                pw = new PrintWriter(new FileWriter(resultsFile, true));
                pw.println("Resumed here");
                pw.flush();
            } else {
                pw = new PrintWriter(new FileWriter(resultsFile));
                pw.println(resultsSummaryHeader);
            }

            //EXECUTE IN PARRALLEL
            System.out.println("total classifiers to test: " + alClassifiers.size());

            if (parrallel) {
                alClassifiers
                        .parallelStream()
                        .map((classif) -> StepWiseFeatureSelectionTraining(classif[0], classif[1], classif[2], classif[3]))
                        .map((s) -> {
                            if (!s.contains("ERROR")) {
                                pw.println(s);
                                pw.flush();
                            } else if (Main.printFailedModels) {
                                pw.println(s);
                                pw.flush();
                            }
                            return s;
                        })
                        .sorted()
                        .forEach((_item) -> {
                            pw.flush();
                        });
//                alClassifiers.stream().parallel().map((classif) -> {
//                    String s = StepWiseFeatureSelectionTraining(classif[0], classif[1], classif[2], classif[3]);
//                    if (!s.contains("ERROR")) {
//                        pw.println(s);
//                    }
//                    return classif;
//                }).forEachOrdered((_item) -> {
//                    pw.flush();
//                });

            } else {
                alClassifiers.stream().map((classif) -> {
                    String s = StepWiseFeatureSelectionTraining(classif[0], classif[1], classif[2], classif[3]);
                    if (!s.contains("ERROR")) {
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

        } catch (Exception e) {
            if (Main.debug) {
                e.printStackTrace();
            }
        } catch (Error err) {
            if (Main.debug) {
                err.printStackTrace();
            }
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
        System.out.println("[model] (" + (cptPassed++) + "/" + alClassifiers.size() + ")" + out);
        String lastOutput = "";

        boolean minimize = valueToMaximizeOrMinimize.equals("fdr")
                || valueToMaximizeOrMinimize.equals("mae")
                || valueToMaximizeOrMinimize.equals("rmse")
                || valueToMaximizeOrMinimize.equals("ber")
                || valueToMaximizeOrMinimize.equals("rae")
                || valueToMaximizeOrMinimize.equals("rrse");

        try {
            StringBuilder predictions = new StringBuilder();
            StringBuilder selectedAttributes = new StringBuilder();
            Classifier model = null;
            Object o = null;
            int numberOfAttributes = 0;
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
            o = weka.shortTestTrainClassifier(classifier, classifier_options,
                    ao.getAttributesIdClassInString(), isClassification);
            if (o.getClass().getName().equals("java.lang.String")) {
                if (Main.debug) {
                    System.out.println("SHORT TEST FAILED");
                }
                return "ERROR\t" + o.toString();
            }
            if (Main.debug) {
                System.out.println("\tGoing to Stepwise evaluations");
            }

            //all features search
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
                    cpt++;
                    //add new attribute to the set of retainedAttributes
                    ao.addNewAttributeToRetainedAttributes(i);
                    Weka_module.ClassificationResultsObject oldcr = cr;
                    //do feature selection by forward(-backward)
                    if (ao.retainedAttributesOnly.size() <= Main.maxNumberOfFeaturesInModel) {
                        o = weka.trainClassifier(classifier, classifier_options,
                                ao.getRetainedAttributesIdClassInString(), isClassification, 10);

                        if (o instanceof String || o == null) {
                            break;
                        } else if (isClassification) {
                            cr = (Weka_module.ClassificationResultsObject) o;
                        } else {
                            rr = (Weka_module.RegressionResultsObject) o;
                        }

                        //choose what we want to maximize or minimize (such as error rates)
                        //this will crash if model had an error
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
                            if (!minimize) {
                                previousMeasureToMaximize = currentMeasure;
                            } else {
                                previousMeasureToMinimize = currentMeasure;
                            }
                            if (isClassification) {
                                predictions = cr.predictions;
                                selectedAttributes = cr.features;
                                numberOfAttributes = cr.numberOfFeatures;
                                model = cr.model;
                            } else {
                                //previousMeasureToMaximize = currentMeasure; ???
                                predictions = rr.predictions;
                                selectedAttributes = rr.features;
                                numberOfAttributes = rr.numberOfFeatures;
                                model = rr.model;
                            }

                            // do backward OR forward, check if we have an improvement if we remove previously chosen features
                            if (doForwardBackward_OR_BackwardForward && numberOfAttributes > 1) {
                                oldcr = cr;
                                ArrayList<Integer> attributesToTestInBackward = ao.getRetainedAttributesIdClassInArrayList();
                                for (int j = 1/*skip ID*/;
                                        j < attributesToTestInBackward.size() - 2/*skip last attribute we added by forward and class*/; j++) {
                                    attributesToTestInBackward.remove(j);

                                    String featuresToTest = utils.arrayToString(attributesToTestInBackward, ",");
                                    //train
                                    if (isClassification) {
                                        cr = (Weka_module.ClassificationResultsObject) weka.trainClassifier(classifier, classifier_options,
                                                featuresToTest, isClassification, 10);
                                    } else {
                                        rr = (Weka_module.RegressionResultsObject) weka.trainClassifier(classifier, classifier_options,
                                                featuresToTest, isClassification, 10);
                                    }
                                    //get measure
                                    double measureWithRemovedFeature = getValueToMaximize(valueToMaximizeOrMinimize, cr, rr);
                                    //check if we have improvement
                                    if (minimize) {
                                        modelIsImproved = (measureWithRemovedFeature <= currentMeasure)
                                                || measureWithRemovedFeature < currentMeasure;
                                    } else {
                                        modelIsImproved = (measureWithRemovedFeature >= currentMeasure)
                                                || measureWithRemovedFeature > currentMeasure;
                                    }

                                    if (modelIsImproved) {
                                        //if model is improved definitly discard feature is improvement
                                        ao.changeRetainedAttributes(featuresToTest);
                                        oldcr = cr;
                                        currentMeasure = measureWithRemovedFeature;
                                    } else {
                                        //restore the feature
                                        attributesToTestInBackward = ao.getRetainedAttributesIdClassInArrayList();
                                        cr = oldcr;
                                    }
                                }
                            }
//
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
                        }
                    }
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
                if (Main.debug) {
                    System.out.println("\tRepeated Holdout on Train set " + Main.bootstrapAndRepeatedHoldoutFolds + " times");
                }
                if (isClassification) {
                    for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                        Weka_module.ClassificationResultsObject cro
                                = (Weka_module.ClassificationResultsObject) weka.trainClassifierHoldOutValidation(classifier, classifier_options,
                                        ao.getRetainedAttributesIdClassInString(), isClassification, i);
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

                        eproRHTrain.alCCs.add(Double.valueOf(rro.CC));
                        eproRHTrain.alMAEs.add(Double.valueOf(rro.MAE));
                        eproRHTrain.alRMSEs.add(Double.valueOf(rro.RMSE));
                        eproRHTrain.alRAEs.add(Double.valueOf(rro.RAE));
                        eproRHTrain.alRRSEs.add(Double.valueOf(rro.RRSE));

                    }
                }
                eproRHTrain.computeMeans();

                //BOOTSTRAP AND BOOTSTRAP .632+ rule TRAIN
                if (Main.debug) {
                    System.out.println("\tBootstrapping on Train set " + Main.bootstrapAndRepeatedHoldoutFolds + " times");
                }
                double bootstrapTrain632plus = 1000;
                Weka_module.evaluationPerformancesResultsObject eproBSTrain
                        = new Weka_module.evaluationPerformancesResultsObject();
                if (isClassification) {
                    bootstrapTrain632plus = weka.trainClassifierBootstrap632plus(classifier, classifier_options,
                            ao.getRetainedAttributesIdClassInString());
                    for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                        Weka_module.ClassificationResultsObject cro
                                = (Weka_module.ClassificationResultsObject) weka.trainClassifierBootstrap(classifier, classifier_options,
                                        ao.getRetainedAttributesIdClassInString(), isClassification, i);
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

                        eproBSTrain.alCCs.add(Double.valueOf(rro.CC));
                        eproBSTrain.alMAEs.add(Double.valueOf(rro.MAE));
                        eproBSTrain.alRMSEs.add(Double.valueOf(rro.RMSE));
                        eproBSTrain.alRAEs.add(Double.valueOf(rro.RAE));
                        eproBSTrain.alRRSEs.add(Double.valueOf(rro.RRSE));
                    }
                }
                eproBSTrain.computeMeans();

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
                    if (Main.doSampling) {
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
                    if (Main.doSampling) {
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
                                        = (Weka_module.RegressionResultsObject) weka2.trainClassifierHoldOutValidation(
                                                classifier, classifier_options,
                                                null, isClassification, i);
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
                    double BERs[] = null;
                    if (Main.loocv) {
                        BERs = new double[]{Double.valueOf(cr.BER),
                            Double.valueOf(eproRHTrain.meanBERs),
                            Double.valueOf(eproBSTrain.meanBERs),
                            Double.valueOf(crLoocv.BER)};
                        if (Main.doSampling) {
                            BERs = new double[]{
                                Double.valueOf(cr.BER),
                                Double.valueOf(crLoocv.BER),
                                Double.valueOf(eproRHTrain.meanBERs),
                                Double.valueOf(eproBSTrain.meanBERs),
                                Double.valueOf(tro.BER),
                                Double.valueOf(eproRHTrainTest.meanBERs),
                                Double.valueOf(eproBSTrainTest.meanBERs)};
                        }
                    } else {
                        BERs = new double[]{Double.valueOf(cr.BER),
                            Double.valueOf(eproRHTrain.meanBERs),
                            Double.valueOf(eproBSTrain.meanBERs)};
                        if (Main.doSampling) {
                            BERs = new double[]{
                                Double.valueOf(cr.BER),
                                Double.valueOf(eproRHTrain.meanBERs),
                                Double.valueOf(eproBSTrain.meanBERs),
                                Double.valueOf(tro.BER),
                                Double.valueOf(eproRHTrainTest.meanBERs),
                                Double.valueOf(eproBSTrainTest.meanBERs)};
                        }
                    }

                    StandardDeviation std = new StandardDeviation();
                    std.setData(BERs);
                    double StdBER = std.evaluate();

                    Mean m = new Mean();
                    m.setData(BERs);
                    double averageBER = m.evaluate();

                    //average MAE
                    double MAEs[] = null;
                    if (Main.loocv) {
                        MAEs = new double[]{Double.valueOf(cr.MAE),
                            Double.valueOf(eproRHTrain.meanMAEs),
                            Double.valueOf(eproBSTrain.meanMAEs),
                            Double.valueOf(crLoocv.MAE)};
                        if (Main.doSampling) {
                            MAEs = new double[]{
                                Double.valueOf(cr.MAE),
                                Double.valueOf(crLoocv.MAE),
                                Double.valueOf(eproRHTrain.meanMAEs),
                                Double.valueOf(eproBSTrain.meanMAEs),
                                Double.valueOf(tro.MAE),
                                Double.valueOf(eproRHTrainTest.meanMAEs),
                                Double.valueOf(eproBSTrainTest.meanMAEs)};
                        }
                    } else {
                        MAEs = new double[]{Double.valueOf(cr.MAE),
                            Double.valueOf(eproRHTrain.meanMAEs),
                            Double.valueOf(eproBSTrain.meanMAEs)};
                        if (Main.doSampling) {
                            MAEs = new double[]{
                                Double.valueOf(cr.MAE),
                                Double.valueOf(eproRHTrain.meanMAEs),
                                Double.valueOf(eproBSTrain.meanMAEs),
                                Double.valueOf(tro.MAE),
                                Double.valueOf(eproRHTrainTest.meanMAEs),
                                Double.valueOf(eproBSTrainTest.meanMAEs)};
                        }
                    }

                    std = new StandardDeviation();
                    std.setData(MAEs);
                    double StdMAE = std.evaluate();

                    m = new Mean();
                    m.setData(MAEs);
                    double averageMAE = m.evaluate();

                    //average MCC
                    double[] MCCs;
                    if (Main.loocv) {
                        MCCs = new double[]{Double.valueOf(cr.MCC),
                            Double.valueOf(eproRHTrain.meanMCCs),
                            Double.valueOf(eproBSTrain.meanMCCs),
                            Double.valueOf(crLoocv.MCC)};
                        if (Main.doSampling) {
                            MCCs = new double[]{
                                Double.valueOf(cr.MCC),
                                Double.valueOf(crLoocv.MCC),
                                Double.valueOf(eproRHTrain.meanMCCs),
                                Double.valueOf(eproBSTrain.meanMCCs),
                                Double.valueOf(tro.MCC),
                                Double.valueOf(eproRHTrainTest.meanMCCs),
                                Double.valueOf(eproBSTrainTest.meanMCCs)};
                        }
                    } else {
                        MCCs = new double[]{Double.valueOf(cr.MCC),
                            Double.valueOf(eproRHTrain.meanMCCs),
                            Double.valueOf(eproBSTrain.meanMCCs)};
                        if (Main.doSampling) {
                            MCCs = new double[]{
                                Double.valueOf(cr.MCC),
                                Double.valueOf(eproRHTrain.meanMCCs),
                                Double.valueOf(eproBSTrain.meanMCCs),
                                Double.valueOf(tro.MCC),
                                Double.valueOf(eproRHTrainTest.meanMCCs),
                                Double.valueOf(eproBSTrainTest.meanMCCs)};
                        }
                    }

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
                    if (bt632.equals(1000)) {
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
                    double[] CCs;
                    if (Main.loocv) {
                        CCs = new double[]{
                            Double.valueOf(rr.CC),
                            Double.valueOf(eproRHTrain.meanCCs),
                            Double.valueOf(eproBSTrain.meanCCs),
                            Double.valueOf(rrLoocv.CC)};
                        if (Main.doSampling) {
                            CCs = new double[]{
                                Double.valueOf(rr.CC),
                                Double.valueOf(rrLoocv.CC),
                                Double.valueOf(eproRHTrain.meanCCs),
                                Double.valueOf(eproBSTrain.meanCCs),
                                Double.valueOf(tro.CC),
                                Double.valueOf(eproRHTrainTest.meanCCs),
                                Double.valueOf(eproBSTrainTest.meanCCs)};
                        }
                    } else {
                        CCs = new double[]{
                            Double.valueOf(rr.CC),
                            Double.valueOf(eproRHTrain.meanCCs),
                            Double.valueOf(eproBSTrain.meanCCs)};
                        if (Main.doSampling) {
                            CCs = new double[]{
                                Double.valueOf(rr.CC),
                                Double.valueOf(eproRHTrain.meanCCs),
                                Double.valueOf(eproBSTrain.meanCCs),
                                Double.valueOf(tro.CC),
                                Double.valueOf(eproRHTrainTest.meanCCs),
                                Double.valueOf(eproBSTrainTest.meanCCs)};
                        }
                    }

                    StandardDeviation std = new StandardDeviation();
                    std.setData(CCs);
                    double StdCC = std.evaluate();

                    Mean m = new Mean();
                    m.setData(CCs);
                    double averageCC = m.evaluate();

                    //average MAE
                    double[] MAEs;
                    if (Main.loocv) {
                        MAEs = new double[]{
                            Double.valueOf(rr.MAE),
                            Double.valueOf(eproRHTrain.meanMAEs),
                            Double.valueOf(eproBSTrain.meanMAEs),
                            Double.valueOf(rrLoocv.MAE)};
                        if (Main.doSampling) {
                            MAEs = new double[]{
                                Double.valueOf(rr.MAE),
                                Double.valueOf(rrLoocv.MAE),
                                Double.valueOf(eproRHTrain.meanMAEs),
                                Double.valueOf(eproBSTrain.meanMAEs),
                                Double.valueOf(tro.MAE),
                                Double.valueOf(eproRHTrainTest.meanMAEs),
                                Double.valueOf(eproBSTrainTest.meanMAEs)};
                        }
                    } else {
                        MAEs = new double[]{
                            Double.valueOf(rr.MAE),
                            Double.valueOf(eproRHTrain.meanMAEs),
                            Double.valueOf(eproBSTrain.meanMAEs)};
                        if (Main.doSampling) {
                            MAEs = new double[]{
                                Double.valueOf(rr.MAE),
                                Double.valueOf(eproRHTrain.meanMAEs),
                                Double.valueOf(eproBSTrain.meanMAEs),
                                Double.valueOf(tro.MAE),
                                Double.valueOf(eproRHTrainTest.meanMAEs),
                                Double.valueOf(eproBSTrainTest.meanMAEs)};
                        }
                    }

                    std = new StandardDeviation();
                    std.setData(MAEs);
                    double StdMAE = std.evaluate();

                    m = new Mean();
                    m.setData(MAEs);
                    double averageMAE = m.evaluate();

                    //average RMSE
                    double[] RMSEs;
                    if (Main.loocv) {
                        RMSEs = new double[]{Double.valueOf(rr.RMSE),
                            Double.valueOf(eproRHTrain.meanRMSEs),
                            Double.valueOf(eproBSTrain.meanRMSEs),
                            Double.valueOf(rrLoocv.RMSE)};
                        if (Main.doSampling) {
                            RMSEs = new double[]{
                                Double.valueOf(rr.RMSE),
                                Double.valueOf(rrLoocv.RMSE),
                                Double.valueOf(eproRHTrain.meanRMSEs),
                                Double.valueOf(eproBSTrain.meanRMSEs),
                                Double.valueOf(tro.RMSE),
                                Double.valueOf(eproRHTrainTest.meanRMSEs),
                                Double.valueOf(eproBSTrainTest.meanRMSEs)};
                        }
                    } else {
                        RMSEs = new double[]{Double.valueOf(rr.RMSE),
                            Double.valueOf(eproRHTrain.meanRMSEs),
                            Double.valueOf(eproBSTrain.meanRMSEs)};
                        if (Main.doSampling) {
                            RMSEs = new double[]{
                                Double.valueOf(rr.RMSE),
                                Double.valueOf(eproRHTrain.meanRMSEs),
                                Double.valueOf(eproBSTrain.meanRMSEs),
                                Double.valueOf(tro.RMSE),
                                Double.valueOf(eproRHTrainTest.meanRMSEs),
                                Double.valueOf(eproBSTrainTest.meanRMSEs)};
                        }
                    }

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
                Random r = new Random();
                int randomNumber = r.nextInt(1000 - 10) + 10;
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
                if (o.toString().equals("null")) {
                    out += "\t " + classifier + " " + classifier_options + " | " + searchMethod + " | Error probably because of number of features inferior to topX";
                }
            }

        } catch (Exception e) {
            out = "ERROR\t" + classifier + " " + classifier_options + " | " + searchMethod + " | " + e.getMessage();
            cptFailed++;
            if (Main.debug) {
                e.printStackTrace();
            }
        }
        Instant finish = Instant.now();
        long s = Duration.between(start, finish).toMillis();
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

        switch (valueWanted.toLowerCase()) {
            //classification
            case "auc":
                return Double.parseDouble(cr.AUC);
            case "pauc":
                return Double.parseDouble(cr.pAUC);
            case "acc":
                return Double.parseDouble(cr.ACC);
            case "sen":
                return Double.parseDouble(cr.TPR);
            case "spe":
                return Double.parseDouble(cr.TNR);
            case "mcc":
                return Double.parseDouble(cr.MCC);
            case "kappa":
                return Double.parseDouble(cr.kappa);
            case "aupcr":
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
        return 0;
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

        for (String optimizer : optimizers) {
            //AUC, ACC, SEN, SPE, MCC, TP+FN, kappa
            for (String searchmode : searchmodes) {
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

        private ArrayList getRetainedAttributesIdClassInArrayList() {
            ArrayList<Integer> al = new ArrayList<>();
            al.add(ID);
            for (Integer retainedAttribute : retainedAttributesOnly) {
                al.add(retainedAttribute);
            }
            al.add(Class);
            return al;
        }

        private void changeRetainedAttributes(String featuresToTest) {
            String features[] = featuresToTest.split(",");
            retainedAttributesOnly = new ArrayList<>();
            for (int i = 1; i < features.length - 1; i++) {//skip ID and class indexes
                retainedAttributesOnly.add(Integer.valueOf(features[i]));
            }
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
