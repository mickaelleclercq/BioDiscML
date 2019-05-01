/*
 * Select only models getting specific min MCC or max RMSE
 * Select all features from selected models
 * Retrain a model (best classifier of all tested) with the unique selected features with LOOCV 75/25.
 * Do it 10 times with various seeds, report the average scores (ex: AUC)
 * Explore biology behind the set of selected features

 */
package biodiscml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.TreeMap;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import utils.UpSetR;
import utils.Weka_module;
import weka.core.SerializationHelper;

/**
 *
 * @author Mickael
 */
public class BestModelSelectionAndReport {

    public static String wd = Main.wd;
    public static Weka_module weka = new Weka_module();
    public static boolean retrieveCorrelatedGenes = true;
    public static HashMap<String, Integer> hmResultsHeaderNames = new HashMap<>();
    public static HashMap< Integer, String> hmResultsHeaderIndexes = new HashMap<>();
    static DecimalFormat df = new DecimalFormat();

    /**
     *
     * @param trainFileName
     * @param featureSelectionFile
     * @param predictionsResultsFile
     * @param type
     */
    public BestModelSelectionAndReport(String trainFileName,
            String featureSelectionFile,
            String predictionsResultsFile,
            String type) {
        df.setMaximumFractionDigits(3);
        DecimalFormatSymbols dfs = new DecimalFormatSymbols();
        dfs.setDecimalSeparator('.');
        df.setDecimalFormatSymbols(dfs);
        String bestOrCombine = "Select best ";
        if (Main.combineModels) {
            bestOrCombine = "Combine ";
        }
        String sign = " >= ";
        boolean metricToMinimize = (Main.bestModelsSortingMetric.contains("RMSE")
                || Main.bestModelsSortingMetric.contains("BER")
                || Main.bestModelsSortingMetric.contains("FPR")
                || Main.bestModelsSortingMetric.contains("FNR")
                || Main.bestModelsSortingMetric.contains("FDR")
                || Main.bestModelsSortingMetric.contains("MAE")
                || Main.bestModelsSortingMetric.contains("RAE")
                || Main.bestModelsSortingMetric.contains("RRSE"));
        if (metricToMinimize) {
            sign = " <= ";
        }

        System.out.println("## " + bestOrCombine + " models using " + Main.bestModelsSortingMetric + " as sorting metric.\n"
                + "## Parameters: " + Main.numberOfBestModels + " best models and "
                + Main.bestModelsSortingMetric + sign + Main.bestModelsSortingMetricThreshold);

        //Read results file
        boolean classification = type.equals("classification");
        Object trainingOutput = null;

        if (classification) {
            init(featureSelectionFile.replace("infoGain.csv", "infoGain.arff"), classification);
        } else {
            init(featureSelectionFile.replace("RELIEFF.csv", "RELIEFF.arff"), classification);
        }

        try {
            BufferedReader br = new BufferedReader(new FileReader(predictionsResultsFile));
            TreeMap<String, Object> tmModels = new TreeMap<>(); //<metric modelID, classification/regression Object>
            HashMap<String, Object> hmModels = new HashMap<>(); //<modelID, classification/regression Object>

            //in case of RMSE or BER, we want the minimum value instead of the maximal one
            if (!metricToMinimize) {
                tmModels = new TreeMap<>(Collections.reverseOrder());
            }
            String line = br.readLine();

            //fill header mapping
            String header = line;
            int cpt = 0;
            for (String s : header.split("\t")) {
                hmResultsHeaderNames.put(s, cpt);
                hmResultsHeaderIndexes.put(cpt, s);
                cpt++;
            }
            if (!hmResultsHeaderNames.containsKey(Main.bestModelsSortingMetric)) {
                System.err.println("[error] " + Main.bestModelsSortingMetric + " column does not exist in the results file.");
                System.exit(0);
            }

            //read results
            while (br.ready()) {
                line = br.readLine();
                if (!line.trim().isEmpty()) {
                    if (classification) {
                        try {
                            classificationObject co = new classificationObject(line);
                            tmModels.put(Double.valueOf(co.hmValues.get(Main.bestModelsSortingMetric)) + " " + co.hmValues.get("ID"), co);
                            hmModels.put(co.hmValues.get("ID"), co);
                        } catch (Exception e) {
                            if (Main.debug) {
                                e.printStackTrace();
                            }
                        }
                    } else {
                        try {
                            regressionObject ro = new regressionObject(line);
                            tmModels.put(Double.valueOf(ro.hmValues.get(Main.bestModelsSortingMetric)) + " " + ro.hmValues.get("ID"), ro);
                            hmModels.put(ro.hmValues.get("ID"), ro);
                        } catch (Exception e) {
                            if (Main.debug) {
                                e.printStackTrace();
                            }
                        }
                    }
                }
            }
            br.close();

            //control available models
            if (Main.numberOfBestModels > tmModels.size()) {
                System.out.println("Only " + tmModels.size() + " available models. You have configured " + Main.numberOfBestModels + " best models");
                Main.numberOfBestModels = tmModels.size();
            }

            // get best models list
            ArrayList<Object> alBestClassifiers = new ArrayList<>();
            cpt = 0;
            if (Main.hmTrainingBestModelList.isEmpty()) {
                for (String metricAndModel : tmModels.keySet()) {
                    cpt++;
                    boolean condition = false;
                    if (metricToMinimize) {
                        condition = Double.valueOf(metricAndModel.split(" ")[0]) < Main.bestModelsSortingMetricThreshold;
                    } else {
                        condition = Double.valueOf(metricAndModel.split(" ")[0]) > Main.bestModelsSortingMetricThreshold;
                    }
                    if (condition && cpt <= Main.numberOfBestModels) {
                        if (classification) {
                            alBestClassifiers.add(((classificationObject) tmModels.get(metricAndModel)));
                        } else {
                            alBestClassifiers.add(((regressionObject) tmModels.get(metricAndModel)));
                        }
                    }
                }
            } else {
                for (String modelID : Main.hmTrainingBestModelList.keySet()) {
                    if (classification) {
                        alBestClassifiers.add(((classificationObject) hmModels.get(modelID)));
                    } else {
                        alBestClassifiers.add(((regressionObject) hmModels.get(modelID)));
                    }
                }
            }

            //if model combination vote
            if (Main.combineModels) {
                if (classification) {
                    classificationObject co = new classificationObject();
                    co.buildVoteClassifier(alBestClassifiers);
                    alBestClassifiers = new ArrayList<>();
                    alBestClassifiers.add(co);
                } else {
                    regressionObject ro = new regressionObject();
                    ro.buildVoteClassifier(alBestClassifiers);
                    alBestClassifiers = new ArrayList<>();
                    alBestClassifiers.add(ro);
                }
            }

            //perform evaluations
            //in the same time, save models
            PrintWriter pw = null;
            for (Object classifier : alBestClassifiers) {
                System.out.print("\n# Model ");
                if (Main.debug) {
                    System.out.println("Save model ");
                }
                String modelFilename;
                Weka_module.ClassificationResultsObject cr = null;
                Weka_module.RegressionResultsObject rr = null;
                classificationObject co = null;
                regressionObject ro = null;
                String classifierName = "";

                //saving files
                if (classification) {
                    co = (classificationObject) classifier;
                    //train model
                    modelFilename = Main.wd + Main.project
                            + "d." + co.classifier + "_" + co.printOptions() + "_"
                            + co.optimizer.toUpperCase().trim() + "_" + co.mode;
                    trainingOutput = weka.trainClassifier(co.classifier, co.options,
                            co.featuresSeparatedByCommas, classification, 10);
                    cr = (Weka_module.ClassificationResultsObject) trainingOutput;

                    classifierName = co.classifier + "_" + co.printOptions() + "_"
                            + co.optimizer.toUpperCase().trim() + "_" + co.mode;
                    //save feature file
                    weka.saveFilteredDataToCSV(co.featuresSeparatedByCommas, classification, modelFilename + ".train_features.csv");
                    //call ranking function
                    cr.featuresRankingResults = weka.featureRankingForClassification(modelFilename + ".train_features.csv");
                    //save model
                    try {
                        SerializationHelper.write(modelFilename + ".model", cr.model);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    System.out.println(modelFilename);
                } else {
                    ro = (regressionObject) classifier;
                    modelFilename = Main.wd + Main.project
                            + "d." + ro.classifier + "_" + ro.printOptions() + "_"
                            + ro.optimizer.toUpperCase().trim() + "_" + ro.mode;
                    trainingOutput = weka.trainClassifier(ro.classifier, ro.options,
                            ro.featuresSeparatedByCommas, classification, 10);
                    rr = (Weka_module.RegressionResultsObject) trainingOutput;
                    classifierName = ro.classifier + "_" + ro.options.replace(" ", "").replace("\\", "").replace("\"", "") + "_"
                            + ro.optimizer.toUpperCase().trim() + "_" + ro.mode;
                    //save feature file
                    weka.saveFilteredDataToCSV(ro.featuresSeparatedByCommas, classification, modelFilename + ".train_features.csv");
                    //call ranking function
                    rr.featuresRankingResults = weka.featureRankingForRegression(modelFilename + ".train_features.csv");
                    //save model and features
                    try {
                        SerializationHelper.write(modelFilename + ".model", rr.model);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    System.out.println(modelFilename);
                }

                //header
                pw = new PrintWriter(new FileWriter(modelFilename + ".details.txt"));
                pw.println("## Generated by BioDiscML (Leclercq et al. 2019)##");
                pw.println("# Project: " + Main.project.substring(0, Main.project.length() - 1));

                if (classification) {
                    pw.println("# ID: " + co.identifier);
                    System.out.println("# ID: " + co.identifier);
                    pw.println("# Classifier: " + co.classifier + " " + co.options
                            + "\n# Optimizer: " + co.optimizer.toUpperCase()
                            + "\n# Feature search mode: " + co.mode);
                } else {
                    pw.println("# ID: " + ro.identifier);
                    System.out.println("# ID: " + ro.identifier);
                    pw.println("# Classifier: " + ro.classifier + " " + ro.options
                            + "\n# Optimizer: " + ro.optimizer.toUpperCase()
                            + "\n# Feature search mode: " + ro.mode);
                }

                //show combined models in case of combined vote
                if (Main.combineModels) {
                    pw.println("# Combined classifiers:");
                    String combOpt = "";
                    if (classification) {
                        combOpt = co.options;
                    } else {
                        combOpt = ro.options;
                    }
                    for (String s : combOpt.split("-B ")) {
                        if (s.startsWith("\"weka.classifiers.meta.FilteredClassifier")) {
                            String usedFeatures = s.split("Remove -V -R ")[1];
                            usedFeatures = usedFeatures.split("\\\\")[0];
                            String model = s.substring(s.indexOf("-W ") + 2).trim()
                                    .replace("-- ", "")
                                    .replace("\\\"", "\"")
                                    .replace("\\\\\"", "\\\"");
                            model = model.substring(0, model.length() - 1);
                            pw.println(model + " (features: " + usedFeatures + ")");
                        }
                    }
                }
                pw.flush();

                //UpSetR
                if (Main.UpSetR) {
                    utils.UpSetR up = new UpSetR();
                    up.creatUpSetRDatasetFromSignature(co, featureSelectionFile, predictionsResultsFile);
                }

                //10CV performance
                System.out.println("# 10 fold cross validation performance");
                pw.println("\n# 10 fold cross validation performance");
                if (classification) {
                    System.out.println(cr.toStringDetails());
                    pw.println(cr.toStringDetails().replace("[score_training] ", ""));
                } else {
                    System.out.println(rr.toStringDetails());
                    pw.println(rr.toStringDetails().replace("[score_training] ", ""));
                }
                pw.flush();

                //LOOCV performance
                System.out.println("# LOOCV (Leave-One-Out cross validation) performance");
                pw.println("\n# LOOCV (Leave-One-Out Cross Validation) performance");
                if (classification) {
                    Weka_module.ClassificationResultsObject cr2 = (Weka_module.ClassificationResultsObject) weka.trainClassifier(co.classifier, co.options,
                            co.featuresSeparatedByCommas, classification, weka.myData.numInstances());
                    System.out.println(cr2.toStringDetails());
                    pw.println(cr2.toStringDetails().replace("[score_training] ", ""));
                } else {
                    Weka_module.RegressionResultsObject rr2 = (Weka_module.RegressionResultsObject) weka.trainClassifier(ro.classifier, ro.options,
                            ro.featuresSeparatedByCommas, classification, weka.myData.numInstances());
                    System.out.println(rr2.toStringDetails());
                    pw.println(rr2.toStringDetails().replace("[score_training] ", ""));
                }
                pw.flush();

                //REPEATED HOLDOUT performance TRAIN set
                ArrayList<Object> alROCs = new ArrayList<>();
                Weka_module.evaluationPerformancesResultsObject eproRHTrain = new Weka_module.evaluationPerformancesResultsObject();
                if (classification) {
                    System.out.println("Repeated Holdout evaluation on TRAIN set of " + co.classifier + " " + co.options
                            + " optimized by " + co.optimizer + "...");
                    pw.println("#Repeated Holdout evaluation performance on TRAIN set, "
                            + Main.bootstrapAndRepeatedHoldoutFolds + " times weighted average (and standard deviation) on random seeds");
                    for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                        Weka_module.ClassificationResultsObject cro
                                = (Weka_module.ClassificationResultsObject) weka.trainClassifierHoldOutValidation(co.classifier, co.options,
                                        co.featuresSeparatedByCommas, classification,i);
                        eproRHTrain.alAUCs.add(Double.valueOf(cro.AUC));
                        eproRHTrain.alpAUCs.add(Double.valueOf(cro.pAUC));
                        eproRHTrain.alAUPRCs.add(Double.valueOf(cro.AUPRC));
                        eproRHTrain.alACCs.add(Double.valueOf(cro.ACC));
                        eproRHTrain.alSEs.add(Double.valueOf(cro.TPR));
                        eproRHTrain.alSPs.add(Double.valueOf(cro.TNR));
                        eproRHTrain.alMCCs.add(Double.valueOf(cro.MCC));
                        eproRHTrain.alBERs.add(Double.valueOf(cro.BER));
                        alROCs.add(cro);
                        // System.out.println(i+"\t"+Double.valueOf(cro.AUC));
                    }
                    eproRHTrain.computeMeans();
                    System.out.println(eproRHTrain.toStringClassificationDetails());
                    pw.println(eproRHTrain.toStringClassificationDetails().replace("[score_training] ", ""));

                    if (Main.ROCcurves) {
                        rocCurveGraphs.createRocCurvesWithConfidence(alROCs, classification, modelFilename, ".roc_train.png");
                    }
                } else {
                    System.out.println("Repeated Holdout evaluation on TRAIN set of " + ro.classifier + " "
                            + ro.options + "optimized by " + ro.optimizer.toUpperCase());
                    pw.println("\n\n#Repeated Holdout evaluation performance on TRAIN set, " + Main.bootstrapAndRepeatedHoldoutFolds + " times average on random seeds");
                    for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                        Weka_module.RegressionResultsObject rro
                                = (Weka_module.RegressionResultsObject) weka.trainClassifierHoldOutValidation(ro.classifier, ro.options,
                                        ro.featuresSeparatedByCommas, classification,i);
                        eproRHTrain.alCCs.add(Double.valueOf(rro.CC));
                        eproRHTrain.alMAEs.add(Double.valueOf(rro.MAE));
                        eproRHTrain.alRMSEs.add(Double.valueOf(rro.RMSE));
                        eproRHTrain.alRAEs.add(Double.valueOf(rro.RAE));
                        eproRHTrain.alRRSEs.add(Double.valueOf(rro.RRSE));
                    }
                    eproRHTrain.computeMeans();
                    pw.println(eproRHTrain.toStringRegressionDetails().replace("[score_training] ", ""));
                }
                pw.flush();

                //BOOTSTRAP performance TRAIN set
                double bootstrapTrain632plus = -1;
                Weka_module.evaluationPerformancesResultsObject eproBSTrain = new Weka_module.evaluationPerformancesResultsObject();
                if (classification) {
                    System.out.println("Bootstrap evaluation on TRAIN set of " + co.classifier + " " + co.options
                            + " optimized by " + co.optimizer + "...");
                    pw.println("\n#Bootstrap evaluation performance on TRAIN set, "
                            + Main.bootstrapAndRepeatedHoldoutFolds + " times weighted average (and standard deviation) on random seeds");
                    for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                        Weka_module.ClassificationResultsObject cro
                                = (Weka_module.ClassificationResultsObject) weka.trainClassifierBootstrap(co.classifier, co.options,
                                        co.featuresSeparatedByCommas, classification,i);
                        eproBSTrain.alAUCs.add(Double.valueOf(cro.AUC));
                        eproBSTrain.alpAUCs.add(Double.valueOf(cro.pAUC));
                        eproBSTrain.alAUPRCs.add(Double.valueOf(cro.AUPRC));
                        eproBSTrain.alACCs.add(Double.valueOf(cro.ACC));
                        eproBSTrain.alSEs.add(Double.valueOf(cro.TPR));
                        eproBSTrain.alSPs.add(Double.valueOf(cro.TNR));
                        eproBSTrain.alMCCs.add(Double.valueOf(cro.MCC));
                        eproBSTrain.alBERs.add(Double.valueOf(cro.BER));
                        alROCs.add(cro);
                        // System.out.println(i+"\t"+Double.valueOf(cro.AUC));
                    }
                    eproBSTrain.computeMeans();
                    System.out.println(eproBSTrain.toStringClassificationDetails());
                    pw.println(eproBSTrain.toStringClassificationDetails().replace("[score_training] ", ""));

                    //632+ rule
                    System.out.println("Bootstrap .632+ rule calculated on TRAIN set of " + co.classifier + " " + co.options
                            + " optimized by " + co.optimizer + "...");
                    pw.println("\n#Bootstrap .632+ rule calculated on TRAIN set, "
                            + Main.bootstrapAndRepeatedHoldoutFolds + " folds with random seeds");

                    bootstrapTrain632plus = weka.trainClassifierBootstrap632plus(co.classifier, co.options,
                            co.featuresSeparatedByCommas);
                    System.out.println(df.format(bootstrapTrain632plus));
                    pw.println(df.format(bootstrapTrain632plus));

                    if (Main.ROCcurves) {
                        rocCurveGraphs.createRocCurvesWithConfidence(alROCs, classification, modelFilename, ".roc_train.png");
                    }
                } else {
                    System.out.println("Bootstrap evaluation on TRAIN set of " + ro.classifier + " "
                            + ro.options + "optimized by " + ro.optimizer.toUpperCase());
                    pw.println("\n#Bootstrap evaluation performance on TRAIN set, "
                            + Main.bootstrapAndRepeatedHoldoutFolds + " times average on random seeds");
                    for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                        Weka_module.RegressionResultsObject rro
                                = (Weka_module.RegressionResultsObject) weka.trainClassifierBootstrap(ro.classifier, ro.options,
                                        ro.featuresSeparatedByCommas, classification,i);
                        eproBSTrain.alCCs.add(Double.valueOf(rro.CC));
                        eproBSTrain.alMAEs.add(Double.valueOf(rro.MAE));
                        eproBSTrain.alRMSEs.add(Double.valueOf(rro.RMSE));
                        eproBSTrain.alRAEs.add(Double.valueOf(rro.RAE));
                        eproBSTrain.alRRSEs.add(Double.valueOf(rro.RRSE));
                    }
                    eproBSTrain.computeMeans();
                    pw.println(eproBSTrain.toStringRegressionDetails().replace("[score_training] ", ""));
                }
                pw.flush();

                // TEST SET
                try {
                    if (Main.doSampling) {
                        alROCs = new ArrayList<>();
                        System.out.println("Evaluation performance on test set");
                        pw.println("\n#Evaluation performance on test set");
                        //get arff test filename generated before training
                        String arffTestFile = trainFileName.replace("data_to_train.csv", "data_to_test.arff");
                        //set the outfile of the extracted features needed to test the current model
                        String arffTestFileWithExtractedModelFeatures = "";
                        if (classification) {
                            arffTestFileWithExtractedModelFeatures = arffTestFile.
                                    replace("_a.classification.data_to_test.arff", "_d." + classifierName + ".test_features.arff");
                        } else {
                            arffTestFileWithExtractedModelFeatures = arffTestFile.
                                    replace("_a.regression.data_to_test.arff", "_d." + classifierName + ".test_features.arff");
                        }
                        //check if test set is here
                        if (!new File(arffTestFile).exists()) {
                            pw.println("Test file " + arffTestFile + " not found");
                        }
                        //adapt test file to model (extract the needed features)
                        //test file come from original dataset preprocessed in AdaptDatasetToTraining, so the arff is compatible
                        Weka_module weka2 = new Weka_module();
                        weka2.setARFFfile(arffTestFile);
                        weka2.setDataFromArff();
                        if (Main.combineModels) {
                            //combined model contains the filters, we need to keep the same
                            //features indexes as the b.featureSelection.infoGain.arff
                            weka2.extractFeaturesFromTestFileBasedOnSelectedFeatures(weka.myData,
                                    weka2.myData, arffTestFileWithExtractedModelFeatures);
                        } else {
                            weka2.extractFeaturesFromTestFileBasedOnModel(modelFilename + ".model",
                                    weka2.myData, arffTestFileWithExtractedModelFeatures);
                        }

                        //testing
                        weka2 = new Weka_module();
                        weka2.setARFFfile(arffTestFileWithExtractedModelFeatures);
                        weka2.setDataFromArff();
                        if (classification) {
                            Weka_module.ClassificationResultsObject cr2
                                    = (Weka_module.ClassificationResultsObject) weka2.testClassifierFromFileSource(new File(weka2.ARFFfile),
                                            modelFilename + ".model", Main.isclassification);
                            alROCs.add(cr2);
                            alROCs.add(cr2);
                            System.out.println("[score_testing] ACC: " + cr2.ACC);
                            System.out.println("[score_testing] AUC: " + cr2.AUC);
                            System.out.println("[score_testing] AUPRC: " + cr2.AUPRC);
                            System.out.println("[score_testing] SEN: " + cr2.TPR);
                            System.out.println("[score_testing] SPE: " + cr2.TNR);
                            System.out.println("[score_testing] MCC: " + cr2.MCC);
                            System.out.println("[score_testing] BER: " + cr2.BER);

                            pw.println("AUC: " + cr2.AUC);
                            pw.println("ACC: " + cr2.ACC);
                            pw.println("AUPRC: " + cr2.AUPRC);
                            pw.println("SEN: " + cr2.TPR);
                            pw.println("SPE: " + cr2.TNR);
                            pw.println("MCC: " + cr2.MCC);
                            pw.println("BER: " + cr2.BER);

                            if (Main.ROCcurves) {
                                rocCurveGraphs.createRocCurvesWithConfidence(alROCs, classification, modelFilename, ".roc_test.png");
                            }
                        } else {
                            Weka_module.RegressionResultsObject rr2
                                    = (Weka_module.RegressionResultsObject) weka2.testClassifierFromFileSource(new File(weka2.ARFFfile),
                                            modelFilename + ".model", classification);
                            System.out.println("[score_testing] Average CC: " + rr2.CC);
                            System.out.println("[score_testing] Average RMSE: " + rr2.RMSE);
                            //
                            pw.println("Average CC: " + rr2.CC);
                            pw.println("Average RMSE: " + rr2.RMSE);
                        }

                        //REPEATED HOLDOUT TRAIN_TEST
                        Weka_module.evaluationPerformancesResultsObject eproRHTrainTest = new Weka_module.evaluationPerformancesResultsObject();
                        try {
                            alROCs = new ArrayList<>();
                            Weka_module weka3 = new Weka_module();
                            weka3.setARFFfile(trainFileName.replace("data_to_train.csv", "all_data.arff"));
                            weka3.setDataFromArff();

                            if (classification) {
                                weka3.myData = weka3.extractFeaturesFromDatasetBasedOnModel(cr.model, weka3.myData);
                                System.out.println("Repeated Holdout evaluation on TRAIN AND TEST sets of " + co.classifier + " " + co.options
                                        + " optimized by " + co.optimizer);
                                pw.println("\n#Repeated Holdout evaluation performance on TRAIN AND TEST set, "
                                        + Main.bootstrapAndRepeatedHoldoutFolds + " times weighted average (and standard deviation) on random seeds");
                                for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                                    Weka_module.ClassificationResultsObject cro
                                            = (Weka_module.ClassificationResultsObject) weka3.trainClassifierHoldOutValidation(co.classifier, co.options,
                                                    null, classification,i);
                                    eproRHTrainTest.alAUCs.add(Double.valueOf(cro.AUC));
                                    eproRHTrainTest.alpAUCs.add(Double.valueOf(cro.pAUC));
                                    eproRHTrainTest.alAUPRCs.add(Double.valueOf(cro.AUPRC));
                                    eproRHTrainTest.alACCs.add(Double.valueOf(cro.ACC));
                                    eproRHTrainTest.alSEs.add(Double.valueOf(cro.TPR));
                                    eproRHTrainTest.alSPs.add(Double.valueOf(cro.TNR));
                                    eproRHTrainTest.alMCCs.add(Double.valueOf(cro.MCC));
                                    eproRHTrainTest.alBERs.add(Double.valueOf(cro.BER));
                                    alROCs.add(cro);
                                }
                                eproRHTrainTest.computeMeans();
                                System.out.println(eproRHTrainTest.toStringClassificationDetails());
                                pw.println(eproRHTrainTest.toStringClassificationDetails().replace("[score_training] ", ""));
                                if (Main.ROCcurves) {
                                    rocCurveGraphs.createRocCurvesWithConfidence(alROCs, classification, modelFilename, ".roc.png");
                                }
                            } else {
                                weka3.myData = weka3.extractFeaturesFromDatasetBasedOnModel(rr.model, weka3.myData);
                                System.out.println("Repeated Holdout evaluation on TRAIN AND TEST sets of " + ro.classifier + " "
                                        + ro.options + "optimized by " + ro.optimizer.toUpperCase());
                                pw.println("\n#Repeated Holdout evaluation performance on TRAIN AND TEST set, "
                                        + Main.bootstrapAndRepeatedHoldoutFolds + " times weighted average (and standard deviation) on random seeds");
                                for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                                    Weka_module.RegressionResultsObject rro
                                            = (Weka_module.RegressionResultsObject) weka3.trainClassifierHoldOutValidation(ro.classifier, ro.options,
                                                    null, classification,i);
                                    eproRHTrainTest.alCCs.add(Double.valueOf(rro.CC));
                                    eproRHTrainTest.alMAEs.add(Double.valueOf(rro.MAE));
                                    eproRHTrainTest.alRMSEs.add(Double.valueOf(rro.RMSE));
                                    eproRHTrainTest.alRAEs.add(Double.valueOf(rro.RAE));
                                    eproRHTrainTest.alRRSEs.add(Double.valueOf(rro.RRSE));
                                }
                                eproRHTrainTest.computeMeans();
                                System.out.println(eproRHTrainTest.toStringRegressionDetails());
                                pw.println(eproRHTrainTest.toStringRegressionDetails().replace("[score_training] ", ""));
                            }
                            eproRHTrainTest.computeMeans();
                        } catch (Exception e) {
                            if (Main.debug) {
                                e.printStackTrace();
                            }
                        }

                        //BOOTSRAP TRAIN_TEST
                        Weka_module.evaluationPerformancesResultsObject eproBSTrainTest = new Weka_module.evaluationPerformancesResultsObject();
                        try {
                            alROCs = new ArrayList<>();
                            Weka_module weka4 = new Weka_module();
                            weka4.setARFFfile(trainFileName.replace("data_to_train.csv", "all_data.arff"));
                            weka4.setDataFromArff();

                            if (classification) {
                                weka4.myData = weka4.extractFeaturesFromDatasetBasedOnModel(cr.model, weka4.myData);
                                System.out.println("Bootstrap evaluation on TRAIN AND TEST sets of " + co.classifier + " " + co.options
                                        + " optimized by " + co.optimizer);
                                pw.println("\n#Bootstrap evaluation performance on TRAIN AND TEST set, "
                                        + Main.bootstrapAndRepeatedHoldoutFolds + " times weighted average (and standard deviation) on random seeds");
                                for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                                    Weka_module.ClassificationResultsObject cro
                                            = (Weka_module.ClassificationResultsObject) weka4.trainClassifierBootstrap(co.classifier, co.options,
                                                    null, classification,i);
                                    eproBSTrainTest.alAUCs.add(Double.valueOf(cro.AUC));
                                    eproBSTrainTest.alpAUCs.add(Double.valueOf(cro.pAUC));
                                    eproBSTrainTest.alAUPRCs.add(Double.valueOf(cro.AUPRC));
                                    eproBSTrainTest.alACCs.add(Double.valueOf(cro.ACC));
                                    eproBSTrainTest.alSEs.add(Double.valueOf(cro.TPR));
                                    eproBSTrainTest.alSPs.add(Double.valueOf(cro.TNR));
                                    eproBSTrainTest.alMCCs.add(Double.valueOf(cro.MCC));
                                    eproBSTrainTest.alBERs.add(Double.valueOf(cro.BER));
                                    alROCs.add(cro);
                                }
                                eproBSTrainTest.computeMeans();
                                System.out.println(eproBSTrainTest.toStringClassificationDetails());
                                pw.println(eproBSTrainTest.toStringClassificationDetails().replace("[score_training] ", ""));

                                //632+ rule
                                System.out.println("Bootstrap .632+ rule calculated on TRAIN AND TEST set of " + co.classifier + " " + co.options
                                        + " optimized by " + co.optimizer + "...");
                                pw.println("\n#Bootstrap .632+ rule calculated on TRAIN AND TEST set, "
                                        + Main.bootstrapAndRepeatedHoldoutFolds + " folds with random seeds");

                                bootstrapTrain632plus = weka4.trainClassifierBootstrap632plus(co.classifier, co.options,
                                        null);
                                System.out.println(df.format(bootstrapTrain632plus));
                                pw.println(df.format(bootstrapTrain632plus));

                                if (Main.ROCcurves) {
                                    rocCurveGraphs.createRocCurvesWithConfidence(alROCs, classification, modelFilename, ".roc.png");
                                }
                            } else {
                                weka4.myData = weka4.extractFeaturesFromDatasetBasedOnModel(rr.model, weka4.myData);
                                System.out.println("Bootstrap evaluation on TRAIN AND TEST sets of " + ro.classifier + " "
                                        + ro.options + "optimized by " + ro.optimizer.toUpperCase());
                                pw.println("\n#Bootstrap evaluation performance on TRAIN AND TEST set, "
                                        + Main.bootstrapAndRepeatedHoldoutFolds + " times weighted average (and standard deviation) on random seeds");
                                for (int i = 0; i < Main.bootstrapAndRepeatedHoldoutFolds; i++) {
                                    Weka_module.RegressionResultsObject rro
                                            = (Weka_module.RegressionResultsObject) weka4.trainClassifierBootstrap(ro.classifier, ro.options,
                                                    null, classification,i);
                                    eproBSTrainTest.alCCs.add(Double.valueOf(rro.CC));
                                    eproBSTrainTest.alMAEs.add(Double.valueOf(rro.MAE));
                                    eproBSTrainTest.alRMSEs.add(Double.valueOf(rro.RMSE));
                                    eproBSTrainTest.alRAEs.add(Double.valueOf(rro.RAE));
                                    eproBSTrainTest.alRRSEs.add(Double.valueOf(rro.RRSE));
                                }
                                eproBSTrainTest.computeMeans();
                                System.out.println(eproBSTrainTest.toStringRegressionDetails());
                                pw.println(eproBSTrainTest.toStringRegressionDetails().replace("[score_training] ", ""));
                            }
                            eproBSTrainTest.computeMeans();
                        } catch (Exception e) {
                            if (Main.debug) {
                                e.printStackTrace();
                            }
                        }

                        //remove test file arff once done
                        new File(arffTestFileWithExtractedModelFeatures).delete();
                    }
                } catch (Exception e) {
                    if (Main.debug) {
                        e.printStackTrace();
                    }
                }
                pw.flush();
                
                // show average metrics and standard deviation

                //output features
                if (classification) {
                    try {
                        pw.print("\n# Selected Attributes (Total attributes:" + cr.numberOfFeatures + "). "
                                + "Occurrences are shown if you chose combined model\n");
                        pw.print(cr.features);
                        pw.println("\n# Attribute ranking by merit calculated by information gain");
                        pw.print(cr.getFeatureRankingResults());

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                } else {
                    try {
                        pw.print("\n# Selected Attributes\t(Total attributes:" + rr.numberOfFeatures + "). "
                                + "Occurrences are shown if you chose combined model\n");
                        pw.print(rr.features);
                        pw.println("\n# Attribute ranking by merit calculated by RELIEFF");
                        pw.print(rr.getFeatureRankingResults());

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                pw.flush();

                //retrieve correlated genes
                if (retrieveCorrelatedGenes) {
                    if (new File(trainFileName).exists()) {
                        System.out.print("Search correlated features (spearman)...");
                        pw.println("\n# Correlated features (Spearman)");
                        pw.println("FeatureInSignature\tSpearmanCorrelationScore\tCorrelatedFeature");
                        TreeMap<String, Double> tmsCorrelatedgenes
                                = RetreiveCorrelatedGenes.spearmanCorrelation(modelFilename + ".train_features.csv", trainFileName);
                        for (String correlation : tmsCorrelatedgenes.keySet()) {
                            pw.println(correlation);
                        }
                        if (tmsCorrelatedgenes.isEmpty()) {
                            pw.println("#nothing found !");
                        }
                        System.out.println("[done]");

                        System.out.print("Search correlated features (pearson)...");
                        pw.println("\n# Correlated features (Pearson)");
                        pw.println("FeatureInSignature\tSpearmanCorrelationScore\tCorrelatedFeature");
                        TreeMap<String, Double> tmpCorrelatedgenes
                                = RetreiveCorrelatedGenes.pearsonCorrelation(modelFilename + ".train_features.csv", trainFileName);
                        for (String correlation : tmpCorrelatedgenes.keySet()) {
                            pw.println(correlation);
                        }
                        if (tmpCorrelatedgenes.isEmpty()) {
                            pw.println("#nothing found !");
                        }
                        System.out.println("[done]");
                    } else {
                        System.out.println("Feature file " + trainFileName + " not found. Unable to calculate correlated genes");
                    }
                    pw.flush();

                    //retreive rankings
                    String ranking = "";
                    if (classification) {
                        ranking = weka.featureRankingForClassification(trainFileName.replace("csv", "arff"));
                    } else {
                        ranking = weka.featureRankingForRegression(trainFileName.replace("csv", "arff"));
                    }
                    String lines[] = ranking.split("\n");
                    HashMap<String, ArrayList<RankerObject>> hmRanks = new HashMap();
                    for (String s : lines) {
                        s = s.replaceAll(" +", " ");
                        if (!s.startsWith("\t") && !s.trim().isEmpty() && s.trim().split(" ").length == 3) {
                            RankerObject rankero = new RankerObject(s.trim());
                            if (hmRanks.containsKey(rankero.roundedScore)) {
                                ArrayList<RankerObject> alRankero = hmRanks.get(rankero.roundedScore);
                                alRankero.add(rankero);
                                hmRanks.put(rankero.roundedScore, alRankero);
                            } else {
                                ArrayList<RankerObject> alRankero = new ArrayList<>();
                                alRankero.add(rankero);
                                hmRanks.put(rankero.roundedScore, alRankero);
                            }
                        }
                    }

                    if (Main.retreiveCorrelatedGenesByRankingScore) {
                        System.out.print("Search similar ranking scores (infogain for classification or relieFf) in the original dataset...");

                        pw.println("\n# Similar ranking score (maximal difference: " + Main.maxRankingScoreDifference + ")");
                        pw.println("FeatureInSignature\tRankingScore\tFeatureInDataset\tRankingScore");
                        String rankedFeaturesSign[];
                        if (classification) {
                            rankedFeaturesSign = cr.getFeatureRankingResults().split("\n");
                        } else {
                            rankedFeaturesSign = rr.getFeatureRankingResults().split("\n");
                        }
                        for (String featureSign : rankedFeaturesSign) {
                            String featureSignIG = featureSign.split("\t")[0];
                            String featureSignIGrounded = df.format(Double.valueOf(featureSignIG));
                            if (hmRanks.containsKey(featureSignIGrounded)) {
                                for (RankerObject alio : hmRanks.get(featureSignIGrounded)) {
                                    if (!featureSign.contains(alio.feature)) {
                                        //max difference between infogains: 0.005
                                        if (Math.abs(Double.parseDouble(featureSignIG) - Double.parseDouble(alio.infogain)) <= Main.maxRankingScoreDifference) {
                                            pw.println(featureSign.split("\t")[1] + "\t"
                                                    + featureSignIG + "\t"
                                                    + alio.feature + "\t"
                                                    + alio.infogain);
                                        }

                                    }
                                }
                            }
                        }
                        System.out.println("[done]");
                    }

                    //close file
                    pw.println("\n\n## End of file ##");
                    pw.close();

                    //export enriched signature
                    LinkedHashMap<String, String> lhmCorrFeaturesNames = new LinkedHashMap<>();//signature + correlated features
                    LinkedHashMap<String, String> lhmFeaturesNames = new LinkedHashMap<>();//signature only
                    try {
                        br = new BufferedReader(new FileReader(modelFilename + ".details.txt"));
                        line = "";
                        //go to selected attributes
                        while (!line.startsWith("# Selected Attributes")) {
                            line = br.readLine();
                        }
                        line = br.readLine();
                        //add attributes to hashmap
                        while (!line.startsWith("#")) {
                            if (!line.isEmpty()) {
                                lhmCorrFeaturesNames.put(line.trim(), "");
                                lhmFeaturesNames.put(line.trim(), "");
                            }
                            line = br.readLine();
                        }
                        //go to spearman correlated attributes
                        while (!line.startsWith("FeatureInSignature")) {
                            line = br.readLine();
                        }
                        line = br.readLine();
                        //add attributes to hashmap
                        while (!line.startsWith("#")) {
                            if (!line.isEmpty()) {
                                lhmCorrFeaturesNames.put(line.split("\t")[2].trim(), "");
                            }
                            line = br.readLine();
                        }
                        //go to pearson correlated attributes
                        while (!line.startsWith("FeatureInSignature")) {
                            line = br.readLine();
                        }
                        line = br.readLine();
                        //add attributes to hashmap
                        while (!line.startsWith("#")) {
                            if (!line.isEmpty()) {
                                lhmCorrFeaturesNames.put(line.split("\t")[2].trim(), "");
                            }
                            line = br.readLine();
                        }
                        if (Main.retreiveCorrelatedGenesByRankingScore) {
                            //go to infogain correlated attributes
                            while (!line.startsWith("FeatureInSignature")) {
                                line = br.readLine();
                            }
                            line = br.readLine();
                            //add attributes to hashmap
                            while (br.ready()) {
                                if (!line.isEmpty()) {
                                    lhmCorrFeaturesNames.put(line.split("\t")[0].trim(), "");
                                    lhmCorrFeaturesNames.put(line.split("\t")[2].trim(), "");
                                }
                                line = br.readLine();
                            }
                        }
                        br.close();
                        //write correlated feature file from training file
                        writeFeaturesFile(lhmCorrFeaturesNames, trainFileName,
                                classification, modelFilename + ".train_corrFeatures.csv");
                        if (Main.doSampling) {
                            //write correlated feature file from all data file
                            //if we have done a sampling, then we can't go from trainFeaturesFile
                            //but to allFeaturesFile, which contain test data
                            writeFeaturesFile(lhmCorrFeaturesNames, trainFileName.replace("data_to_train", "all_data"),
                                    classification, modelFilename + ".all_corrFeatures.csv");

                            //short signature
                            writeFeaturesFile(lhmFeaturesNames, trainFileName.replace("data_to_train", "all_data"),
                                    classification, modelFilename + ".all_features.csv");
                        }

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        } catch (ClassCastException e) {
            e.printStackTrace();
            System.err.println("Unable to train selected best model(s). Check input files. ");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     *
     * @param lhm feature names in order
     * @param originFile the training file or all data file
     * @param classification if we are doing a classification
     * @param outfile outfile name
     */
    private void writeFeaturesFile(LinkedHashMap<String, String> lhm, String originFile, boolean classification, String outfile) {
        //find columns indices
        String featuresSeparatedByCommas = "1";
        try {
            BufferedReader br = new BufferedReader(new FileReader(originFile));
            String header = br.readLine();
            String features[] = header.split(utils.utils.detectSeparator(originFile));
            for (int i = 0; i < features.length; i++) {
                String feature = features[i];
                if (lhm.containsKey(feature)) {
                    featuresSeparatedByCommas += "," + (i + 1);
                }
            }
            featuresSeparatedByCommas += "," + features.length;
        } catch (Exception e) {
            e.printStackTrace();
        }

        //extract columns using weka filter
        Weka_module weka2 = new Weka_module();
        weka2.setARFFfile(originFile.replace("csv", "arff"));
        weka2.setDataFromArff();
        weka2.saveFilteredDataToCSV(featuresSeparatedByCommas,
                classification, outfile);

    }

    /**
     * initialize weka
     *
     * @param infile
     * @param classification
     */
    private static void init(String infile, boolean classification) {
        //convert csv to arff
        if (infile.endsWith(".csv")) {
            weka.setCSVFile(new File(infile));
            weka.csvToArff(classification);
        } else {
            weka.setARFFfile(infile.replace(".csv", ".arff"));
        }

        //set local variable of weka object from ARFFfile
        weka.setDataFromArff();
//        // check if class has numeric values, hence regression, instead of nominal class (classification)
        classification = weka.isClassification();
    }

    /**
     * calculate mean and standard deviation of an array of doubles
     *
     * @param al
     * @return
     */
    private static String getMeanAndStandardDeviation(ArrayList<Double> al) {

        double d[] = new double[al.size()];
        for (int i = 0; i < al.size(); i++) {
            d[i] = (double) al.get(i);
        }

        StandardDeviation sd = new StandardDeviation();
        Mean m = new Mean();

        return df.format(m.evaluate(d)) + " (" + df.format(sd.evaluate(d)) + ")";
    }

    /**
     * classification object
     */
    public static class classificationObject {

        public ArrayList<String> featureList = new ArrayList<>();
        public String featuresSeparatedByCommas = "";
        public String optimizer = "";
        public String mode = "";
        public String classifier = "";
        public String options = "";
        public String identifier = "";
        public TreeMap<Integer, Integer> tmFeatures;
        public HashMap<String, String> hmValues = new HashMap<>(); //Column name, value

        public classificationObject() {
        }

        public classificationObject(String line) {

            identifier = line.split("\t")[hmResultsHeaderNames.get("ID")];
            classifier = line.split("\t")[hmResultsHeaderNames.get("classifier")];
            options = line.split("\t")[hmResultsHeaderNames.get("Options")];
            optimizer = line.split("\t")[hmResultsHeaderNames.get("OptimizedValue")];
            mode = line.split("\t")[hmResultsHeaderNames.get("SearchMode")];

            featureList.addAll(Arrays.asList(line.split("\t")[hmResultsHeaderNames.get("AttributeList")].split(",")));
            featuresSeparatedByCommas = line.split("\t")[hmResultsHeaderNames.get("AttributeList")];
            String s[] = line.split("\t");
            for (int i = 0; i < s.length; i++) {
                hmValues.put(hmResultsHeaderIndexes.get(i), s[i]);
            }

        }

        /**
         * for combined models
         *
         * @param alBestClassifiers
         * @param NumberOfTopModels
         */
        private void buildVoteClassifier(ArrayList<Object> alBestClassifiers) {
            classifier = "meta.Vote"; //Combination rule: average of probabilities
            options = "-S 1 -R " + Main.combinationRule;

            tmFeatures = new TreeMap<>();
            for (Object c : alBestClassifiers) {
                classificationObject co = (classificationObject) c;
                //create filteredclassifier with selected attributes
                String filteredClassifierOptions = "-B \"weka.classifiers.meta.FilteredClassifier -F \\\"weka.filters.unsupervised.attribute.Remove -V -R "
                        + co.featuresSeparatedByCommas.substring(2) + "\\\"";
                //add classifier and its options
                String classif = "-W weka.classifiers." + co.classifier + " --";
                String classifOptions = co.options.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
                options += " " + filteredClassifierOptions + " " + classif + " " + classifOptions;

                //get all features seen and get their number of views
                for (String f : co.featuresSeparatedByCommas.split(",")) {
                    if (tmFeatures.containsKey(Integer.valueOf(f))) {
                        int i = tmFeatures.get(Integer.valueOf(f));
                        i++;
                        tmFeatures.put(Integer.valueOf(f), i);
                    } else {
                        tmFeatures.put(Integer.valueOf(f), 1);
                    }
                }
            }
            //set features lists
            for (Integer f : tmFeatures.keySet()) {
                featureList.add(f.toString());
                featuresSeparatedByCommas += "," + f;
            }
            featuresSeparatedByCommas = featuresSeparatedByCommas.substring(1);

            //set other variables
            optimizer = "COMB";
            mode = Main.numberOfBestModels + "_" + Main.bestModelsSortingMetric + "_" + Main.bestModelsSortingMetricThreshold;
        }

        /**
         * printable version of options
         *
         * @return
         */
        private String printOptions() {
            if (classifier.contains("meta.Vote")) {
                return options.substring(0, options.indexOf("-B")).replace(" ", "");
            } else {
                return options.replace(" ", "").replace("\\", "").replace("\"", "");
            }
        }

    }

    public static class regressionObject {

        public ArrayList<String> featureList = new ArrayList<>();
        public String featuresSeparatedByCommas="";
        public String classifier;
        public String optimizer;
        public String options;
        public String mode;
        public String identifier;
        public TreeMap<Integer, Integer> tmFeatures;
        public HashMap<String, String> hmValues = new HashMap<>(); //Column name, value

        public regressionObject() {
        }

        private regressionObject(String line) {
            identifier = line.split("\t")[hmResultsHeaderNames.get("ID")];
            classifier = line.split("\t")[hmResultsHeaderNames.get("classifier")];
            options = line.split("\t")[hmResultsHeaderNames.get("Options")];
            optimizer = line.split("\t")[hmResultsHeaderNames.get("OptimizedValue")];
            mode = line.split("\t")[hmResultsHeaderNames.get("SearchMode")];
            featureList.addAll(Arrays.asList(line.split("\t")[hmResultsHeaderNames.get("AttributeList")].split(",")));
            featuresSeparatedByCommas = line.split("\t")[hmResultsHeaderNames.get("AttributeList")];

            if (options.startsWith("\"")) {
                options = options.substring(1); //remove first "
                options = options.replace("\\\"\"", "\\\""); //  replace \"" by \"
                options = options.replace("\"\"\"", "\""); // replace """ by "
                options = options.replace("\"\"weka", "\"weka"); // replace  "" by "
            }

            String s[] = line.split("\t");
            for (int i = 0; i < s.length; i++) {
                hmValues.put(hmResultsHeaderIndexes.get(i), s[i]);
            }
        }

        /**
         * for combined models
         *
         * @param alBestClassifiers
         * @param cc
         * @param NumberOfTopModels
         */
        private void buildVoteClassifier(ArrayList<Object> alBestClassifiers) {
            classifier = "meta.Vote"; //Combination rule: average of probabilities
            options = "-S 1 -R " + Main.combinationRule;

            tmFeatures = new TreeMap<>();
            int cpt = 0;
            for (Object r : alBestClassifiers) {
                regressionObject ro = (regressionObject) r;
                cpt++;
                //create filteredclassifier with selected attributes
                String filteredClassifierOptions = "-B \"weka.classifiers.meta.FilteredClassifier -F \\\"weka.filters.unsupervised.attribute.Remove -V -R "
                        + ro.featuresSeparatedByCommas.substring(2) + "\\\"";
                //add classifier and its options
                String classif = "-W weka.classifiers." + ro.classifier + " --";
                String classifOptions = ro.options.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
                options += " " + filteredClassifierOptions + " " + classif + " " + classifOptions;

                //get all features seen and get their number of views
                for (String f : ro.featuresSeparatedByCommas.split(",")) {
                    if (tmFeatures.containsKey(Integer.valueOf(f))) {
                        int i = tmFeatures.get(Integer.valueOf(f));
                        i++;
                        tmFeatures.put(Integer.valueOf(f), i);
                    } else {
                        tmFeatures.put(Integer.valueOf(f), 1);
                    }
                }

            }

            //set features lists
            for (Integer f : tmFeatures.keySet()) {
                featureList.add(f.toString());
                featuresSeparatedByCommas += "," + f;
            }
            featuresSeparatedByCommas = featuresSeparatedByCommas.substring(1);

            //set other variables
            optimizer = "COMB";
            mode = Main.numberOfBestModels + "_" + Main.bestModelsSortingMetric + "_" + Main.bestModelsSortingMetricThreshold;
        }

        /**
         * printable version of options
         *
         * @return
         */
        private String printOptions() {
            if (classifier.contains("meta.Vote")) {
                return options.substring(0, options.indexOf("-B")).replace(" ", "");
            } else {
                return options.replace(" ", "").replace("\\", "").replace("\"", "");
            }
        }

    }

    private static class RankerObject {

        public String infogain;
        public String roundedScore;
        public String feature;

        public RankerObject() {

        }

        private RankerObject(String s) {
            infogain = s.split(" ")[0];
            roundedScore = df.format(Double.valueOf(s.split(" ")[0]));
            feature = s.split(" ")[2];
        }
    }

}
