/*
 * execute machine learning algorithms in parrallel
 * Works with regression and classification
 */
package biodiscml;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import org.apache.commons.math3.stat.descriptive.moment.*;
import utils.Weka_module;
import utils.utils;
import static utils.utils.getMean;
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
                    + "\tTRAIN_LOOCV_BER"
                    //bootstrap TRAIN
                    + "\tTRAIN_BS_ACC"
                    + "\tTRAIN_BS_AUC"
                    + "\tTRAIN_BS_AUPRC"
                    + "\tTRAIN_BS_SEN"
                    + "\tTRAIN_BS_SPE"
                    + "\tTRAIN_BS_MCC"
                    + "\tTRAIN_BS_BER"
                    //test
                    + "\tTEST_ACC"
                    + "\tTEST_AUC"
                    + "\tTEST_AUPRC"
                    + "\tTEST_SEN"
                    + "\tTEST_SPE"
                    + "\tTEST_MCC"
                    + "\tTEST_BER"
                    //bootstrap TRAIN_TEST
                    + "\tTRAIN_TEST_BS_ACC"
                    + "\tTRAIN_TEST_BS_AUC"
                    + "\tTRAIN_TEST_BS_AUPRC"
                    + "\tTRAIN_TEST_BS_SEN"
                    + "\tTRAIN_TEST_BS_SPE"
                    + "\tTRAIN_TEST_BS_MCC"
                    + "\tTRAIN_TEST_BS_BER"
                    //stats
                    + "\tAVG_BER"
                    + "\tSTD_BER"
                    + "\tAVG_MCC"
                    + "\tSTD_MCC"
                    + "\tAttributeList";

            //fast way classification. Predetermined commands.
            if (Main.classificationFastWay) {
                //hmclassifiers.put(new String[]{"misc.VFI", "-B 0.6", "AUC"}, "");
                //hmclassifiers.put(new String[]{"meta.CostSensitiveClassifier", "-cost-matrix \"[0.0 ratio; 100.0 0.0]\" -S 1 -W weka.classifiers.misc.VFI -- -B 0.4", "MCC"}, "");

                for (String cmd : Main.classificationFastWayCommands) {
                    String optimizer = cmd.split(":")[1];
                    String classifier = cmd.split(":")[0].split(" ")[0];
                    String options = cmd.split(":")[0].replace(classifier, "");
                    if (optimizer.equals("ALL")) {
                        addClassifierToQueue(classifier, options);
                    } else {
                        alClassifiers.add(new String[]{classifier, options, optimizer, "F"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "FB"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "B"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "BF"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top5"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top10"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top15"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top20"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top30"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top40"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top50"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top75"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top100"});
                    }

                }
            } else {
                //brute force classification, try everything in the provided classifiers and optimizers
                for (String cmd : Main.classificationBruteForceCommands) {
                    String classifier = cmd.split(" ")[0];
                    String options = cmd.replace(classifier, "");
                    addClassifierToQueue(classifier, options);
                }
            }

            System.out.print("Feature selection and ranking...");
            //ATTRIBUTE SELECTION for classification
            weka.attributeSelectionByInfoGainRankingAndSaveToCSV(featureSelectionFile);
            //get the rank of attributes
            ranking = weka.featureRankingForClassification();
            System.out.println("[done]");

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
                    //BOOTSRAP
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
                    //BOOTSRAP TRAIN_TEST
                    + "\tTRAIN_TEST_BS_CC"
                    + "\tTRAIN_TEST_BS_MAE"
                    + "\tTRAIN_TEST_BS_RMSE"
                    + "\tTRAIN_TEST_BS_RAE"
                    + "\tTRAIN_TEST_BS_RRSE"
                    //stats
                    + "\tAVG_CC"
                    + "\tSTD_CC"
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
                    String classifier = cmd.split(":")[0].split(" ")[0];
                    String options = cmd.split(":")[0].replace(classifier, "");
                    if (optimizer.equals("ALL")) {
                        addRegressionToQueue(classifier, options);
                    } else {
                        alClassifiers.add(new String[]{classifier, options, optimizer, "F"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "FB"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "B"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "BF"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top5"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top10"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top15"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top20"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top30"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top40"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top50"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top75"});
                        alClassifiers.add(new String[]{classifier, options, optimizer, "top100"});
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
            weka.attributeSelectionByRelieFFAndSaveToCSV(featureSelectionFile);
            System.out.println("[done]");
        }

        //reset arff and keep compatible header
        weka.setCSVFile(new File(featureSelectionFile));
        weka.csvToArff(isClassification);
        weka.makeCompatibleARFFheaders(dataToTrainModel.replace("data_to_train.csv", "data_to_train.arff"),
                featureSelectionFile.replace("infoGain.csv", "infoGain.arff"));
        weka.setARFFfile(featureSelectionFile.replace("infoGain.csv", "infoGain.arff"));
        weka.setDataFromArff();

        try {
            //PREPARE OUTPUT
            PrintWriter pw = new PrintWriter(new FileWriter(resultsFile));
            pw.println(resultsSummaryHeader);

            //EXECUTE IN PARRALLEL
            System.out.println("total classifiers to test: " + alClassifiers.size());
            if (parrallel) {
                alClassifiers.parallelStream().map((classif) -> {
                    String s = StepWiseFeatureSelectionTraining(classif[0], classif[1], classif[2], classif[3]);
                    if (!s.contains("ERROR")) {
                        pw.println(s);
                    }
                    return classif;
                }).forEach((_item) -> {
                    pw.flush();
                });
            } else {
                alClassifiers.stream().map((classif) -> {
                    String s = StepWiseFeatureSelectionTraining(classif[0], classif[1], classif[2], classif[3]);
                    if (!s.contains("ERROR")) {
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
    private static String StepWiseFeatureSelectionTraining(String classifier, String classifier_options, String valueToMaximizeOrMinimize,
            String searchMethod) {
        String out = classifier + "\t" + classifier_options + "\t" + valueToMaximizeOrMinimize.toUpperCase() + "\t" + searchMethod;

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
            //TOP X features search
            if (searchMethod.startsWith("top")) {
                int top = Integer.valueOf(searchMethod.replace("top", ""));
                if (top <= ao.alAttributes.size()) {
                    //Create attribute list
                    for (int i = 0; i < top; i++) { //from ID to class (excluded)
                        //add new attribute to the set of retainedAttributes
                        ao.addNewAttributeToRetainedAttributes(i);
                    }
                    //TRAIN
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

            }

            //FORWARD AND FORWARD-BACKWARD search AND BACKWARD AND BACKWARD-FORWARD search
            if (searchMethod.startsWith("F") || searchMethod.startsWith("B")) {
                boolean doForwardBackward_OR_BackwardForward = searchMethod.equals("FB") || searchMethod.equals("BF");
                boolean StartBackwardInsteadForward = searchMethod.startsWith("B");
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
                                // System.out.println(lastOutput);

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
                } else {
                    rrLoocv = (Weka_module.RegressionResultsObject) oLoocv;
                }

                //BOOTSTRAP
                Weka_module.bootstrapResultsObject broTrain = new Weka_module.bootstrapResultsObject();
                if (isClassification) {
                    for (int i = 0; i < Main.bootstrapFolds; i++) {
                        Weka_module.ClassificationResultsObject cro
                                = (Weka_module.ClassificationResultsObject) weka.trainClassifierHoldOutCVandTest(classifier, classifier_options,
                                        ao.getRetainedAttributesIdClassInString(), isClassification, i);
                        broTrain.alAUCs.add(Double.valueOf(cro.AUC));
                        broTrain.alpAUCs.add(Double.valueOf(cro.pAUC));
                        broTrain.alAUPRCs.add(Double.valueOf(cro.AUPRC));
                        broTrain.alACCs.add(Double.valueOf(cro.ACC));
                        broTrain.alSEs.add(Double.valueOf(cro.TPR));
                        broTrain.alSPs.add(Double.valueOf(cro.TNR));
                        broTrain.alMCCs.add(Double.valueOf(cro.MCC));
                        broTrain.alBERs.add(Double.valueOf(cro.BER));
                    }
                } else {
                    for (int i = 0; i < Main.bootstrapFolds; i++) {
                        Weka_module.RegressionResultsObject rro
                                = (Weka_module.RegressionResultsObject) weka.trainClassifierHoldOutCVandTest(classifier, classifier_options,
                                        ao.getRetainedAttributesIdClassInString(), isClassification, i);

                        broTrain.alCCs.add(Double.valueOf(rro.CC));
                        broTrain.alMAEs.add(Double.valueOf(rro.MAE));
                        broTrain.alRMSEs.add(Double.valueOf(rro.RMSE));
                        broTrain.alRAEs.add(Double.valueOf(rro.RAE));
                        broTrain.alRRSEs.add(Double.valueOf(rro.RRSE));

                    }
                }
                broTrain.computeMeans();

                // TEST SET
                Weka_module.testResultsObject tro = new Weka_module.testResultsObject();
                String testResults = null;
                if (isClassification) {
                    testResults = "\t" + "\t" + "\t" + "\t" + "\t" + "\t";
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
                                cr2 = (Weka_module.ClassificationResultsObject) weka2.testClassifierFromModel(
                                        cr.model,
                                        trainFileName.replace("data_to_train.csv", "data_to_test.csv"),//test file
                                        Main.isclassification, out);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                            tro.ACC = cr2.ACC;
                            tro.AUC = cr2.AUC;
                            tro.AUPRC = cr2.AUPRC;
                            tro.SE = cr2.TPR;
                            tro.SP = cr2.TNR;
                            tro.MCC = cr2.MCC;
                            tro.BER = cr2.BER;
                            testResults = tro.toStringClassification();

                        } else {
                            Weka_module.RegressionResultsObject rr2
                                    = (Weka_module.RegressionResultsObject) weka2.testClassifierFromModel(
                                            rr.model,
                                            trainFileName.replace("data_to_train.csv", "data_to_test.csv"),//test file
                                            Main.isclassification, out);
                            tro.CC = rr2.CC;
                            tro.RMSE = rr2.RMSE;
                            testResults = tro.toStringRegression();
                        }
                    }
                } catch (Exception e) {
                    if (Main.debug) {
                        e.printStackTrace();
                    }
                }

                //BOOTSTRAP TRAIN_TEST
                Weka_module.bootstrapResultsObject broTrainTest = new Weka_module.bootstrapResultsObject();
                try {
                    if (Main.doSampling) {
                        Weka_module weka2 = new Weka_module();
                        weka2.setARFFfile(trainFileName.replace("data_to_train.csv", "all_data.arff"));
                        weka2.setDataFromArff();
                        weka2.myData = weka2.extractFeaturesFromDatasetBasedOnModel(cr.model, weka2.myData);

                        if (isClassification) {
                            for (int i = 0; i < Main.bootstrapFolds; i++) {
                                Weka_module.ClassificationResultsObject cro
                                        = (Weka_module.ClassificationResultsObject) weka2.trainClassifierHoldOutCVandTest(
                                                classifier, classifier_options,
                                                null, isClassification, i);

                                broTrainTest.alAUCs.add(Double.valueOf(cro.AUC));
                                broTrainTest.alpAUCs.add(Double.valueOf(cro.pAUC));
                                broTrainTest.alAUPRCs.add(Double.valueOf(cro.AUPRC));
                                broTrainTest.alACCs.add(Double.valueOf(cro.ACC));
                                broTrainTest.alSEs.add(Double.valueOf(cro.TPR));
                                broTrainTest.alSPs.add(Double.valueOf(cro.TNR));
                                broTrainTest.alMCCs.add(Double.valueOf(cro.MCC));
                                broTrainTest.alBERs.add(Double.valueOf(cro.BER));
                            }
                        } else {
                            for (int i = 0; i < Main.bootstrapFolds; i++) {
                                Weka_module.RegressionResultsObject rro
                                        = (Weka_module.RegressionResultsObject) weka2.trainClassifierHoldOutCVandTest(classifier, classifier_options,
                                                ao.getRetainedAttributesIdClassInString(), isClassification, i);
                                broTrainTest.alCCs.add(Double.valueOf(rro.CC));
                                broTrainTest.alMAEs.add(Double.valueOf(rro.MAE));
                                broTrainTest.alRMSEs.add(Double.valueOf(rro.RMSE));
                                broTrainTest.alRAEs.add(Double.valueOf(rro.RAE));
                                broTrainTest.alRRSEs.add(Double.valueOf(rro.RRSE));
                            }
                        }
                        broTrainTest.computeMeans();
                    }
                } catch (Exception e) {
                    if (Main.debug) {
                        e.printStackTrace();
                    }
                }

                //STATISTICS AND OUTPUT
                if (isClassification) {
                    //statistics
                    double BERs[] = {Double.valueOf(cr.BER), Double.valueOf(broTrain.meanBERs), Double.valueOf(crLoocv.BER)};
                    if (Main.doSampling) {
                        BERs = new double[]{Double.valueOf(cr.BER), Double.valueOf(crLoocv.BER),
                            Double.valueOf(broTrain.meanBERs),
                            Double.valueOf(tro.BER),
                            Double.valueOf(broTrainTest.meanBERs)};
                    }
                    StandardDeviation std = new StandardDeviation();
                    std.setData(BERs);
                    double StdBER = std.evaluate();

                    Mean m = new Mean();
                    m.setData(BERs);
                    double averageBER = m.evaluate();

                    double[] MCCs = {Double.valueOf(cr.MCC), Double.valueOf(broTrain.meanMCCs), Double.valueOf(crLoocv.MCC)};
                    if (Main.doSampling) {
                        MCCs = new double[]{Double.valueOf(cr.MCC), Double.valueOf(crLoocv.MCC),
                            Double.valueOf(broTrain.meanMCCs),
                            Double.valueOf(tro.MCC),
                            Double.valueOf(broTrainTest.meanMCCs),};
                    }
                    std = new StandardDeviation();
                    std.setData(MCCs);
                    double StdMCC = std.evaluate();

                    m = new Mean();
                    m.setData(MCCs);
                    double averageMCC = m.evaluate();

                    String stats = df.format(averageBER) + "\t"
                            + df.format(StdBER) + "\t"
                            + df.format(averageMCC) + "\t"
                            + df.format(StdMCC);

                    //output
                    lastOutput = out
                            + "\t" + cr.numberOfFeatures + "\t" + cr.toString()
                            + "\t" + crLoocv.toStringShort()
                            + "\t" + broTrain.toStringClassification()
                            + "\t" + testResults
                            + "\t" + broTrainTest.toStringClassification()
                            + "\t" + stats + "\t" + ao.getRetainedAttributesIdClassInString();
                } else {
                    //statistics
                    double[] CCs = {Double.valueOf(rr.CC), Double.valueOf(broTrain.meanCCs), Double.valueOf(rrLoocv.CC)};
                    if (Main.doSampling) {
                        CCs = new double[]{Double.valueOf(rr.CC), Double.valueOf(rrLoocv.CC),
                            Double.valueOf(broTrain.meanCCs),
                            Double.valueOf(tro.CC),
                            Double.valueOf(broTrainTest.meanCCs)};
                    }
                    StandardDeviation std = new StandardDeviation();
                    std.setData(CCs);
                    double StdCC = std.evaluate();

                    Mean m = new Mean();
                    m.setData(CCs);
                    double averageCC = m.evaluate();

                    double[] RMSEs = {Double.valueOf(rr.RMSE),
                        Double.valueOf(broTrain.meanRMSEs), Double.valueOf(rrLoocv.RMSE)};
                    if (Main.doSampling) {
                        RMSEs = new double[]{Double.valueOf(rr.RMSE),
                            Double.valueOf(rrLoocv.RMSE),
                            Double.valueOf(broTrain.meanRMSEs),
                            Double.valueOf(tro.RMSE),
                            Double.valueOf(broTrainTest.meanRMSEs)};
                    }
                    std = new StandardDeviation();
                    std.setData(RMSEs);
                    double StdRMSE = std.evaluate();

                    m = new Mean();
                    m.setData(RMSEs);
                    double averageRMSE = m.evaluate();

                    String stats = df.format(averageCC) + "\t"
                            + df.format(StdCC) + "\t"
                            + df.format(averageRMSE) + "\t"
                            + df.format(StdRMSE);
                    lastOutput = out
                            + "\t" + rr.numberOfFeatures + "\t" + rr.toString()
                            + "\t" + rrLoocv.toString()
                            + "\t" + broTrain.toStringRegression()
                            + "\t" + testResults
                            + "\t" + broTrainTest.toStringRegression()
                            + "\t" + stats + "\t" + ao.getRetainedAttributesIdClassInString();
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
            }

        } catch (Exception e) {
            out = "ERROR";
            cptFailed++;
            if (Main.debug) {
                e.printStackTrace();
            }
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

        switch (valueWanted) {
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
    private static void addClassifierToQueue(String classifier, String options) {
        String valuesToMaximize[] = Main.classificationOptimizers.split(",");

        for (String value : valuesToMaximize) {
            //AUC, ACC, SEN, SPE, MCC, TP+FN, kappa
            alClassifiers.add(new String[]{classifier, options, value.trim(), "F"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "FB"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "B"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "BF"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top1"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top5"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top10"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top15"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top20"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top25"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top30"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top40"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top50"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top75"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top100"});
        }
        if (Main.metaCostSensitiveClassifier) {
            options = "-cost-matrix \"[0.0 ratio; 100.0 0.0]\" -S 1 -W weka.classifiers." + classifier + " -- " + options;
            for (String value : valuesToMaximize) {
                alClassifiers.add(new String[]{"meta.CostSensitiveClassifier", options, value.trim(), "F"});
                alClassifiers.add(new String[]{"meta.CostSensitiveClassifier", options, value.trim(), "BF"});
            }
        }
    }

    private static void addRegressionToQueue(String classifier, String options) {
        //String valuesToMaximizeOrMinimize[] = new String[]{"CC", "MAE", "RMSE", "RAE", "RRSE"};
        String valuesToMaximizeOrMinimize[] = Main.regressionOptimizers.split(",");
        for (String value : valuesToMaximizeOrMinimize) {//only CC is maximized here
            alClassifiers.add(new String[]{classifier, options, value, "F"});
            alClassifiers.add(new String[]{classifier, options, value, "FB"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "B"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "BF"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top1"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top5"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top10"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top15"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top20"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top25"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top30"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top40"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top50"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top75"});
            alClassifiers.add(new String[]{classifier, options, value.trim(), "top100"});
        }
        if (Main.metaAdditiveRegression) {
            options = "-S 1.0 -I 10 -W weka.classifiers." + classifier + " -- " + options;
            for (String value : valuesToMaximizeOrMinimize) {
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
    }

}
