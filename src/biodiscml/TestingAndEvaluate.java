/*
 * Test and evaluate
 */
package biodiscml;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import utils.Weka_module;

/**
 *
 * @author Mickael
 */
public class TestingAndEvaluate {

    TestingAndEvaluate() {

    }

    /**
     * class exist and we know the outcome, so we evaluate the predictor
     *
     * @param modelFile
     * @param TEST_FILE
     * @param TEST_RESULTS_FILE
     */
    public void TestingAndEvaluate(String modelFile, String TEST_FILE, String TEST_RESULTS_FILE) {
        Weka_module weka = new Weka_module();
        weka.setCSVFile(new File(TEST_FILE));
        weka.csvToArff(Main.isclassification);
        weka.setDataFromArff();

        Weka_module.ClassificationResultsObject cr = null;
        Weka_module.RegressionResultsObject rr = null;

        try {
            PrintWriter pw = new PrintWriter(new FileWriter(TEST_RESULTS_FILE));
            System.out.println("Test results stored in " + TEST_RESULTS_FILE);

            if (Main.isclassification) {
                cr = (Weka_module.ClassificationResultsObject) weka.testClassifierFromFileSource(new File(weka.ARFFfile), modelFile, Main.isclassification);
                cr.getPredictions();
                pw.println("instance\tactual\tpredicted\terror\tprobability\n" + cr.predictions);
                System.out.println(cr.toStringDetails());
                Main.bench_AUC = cr.AUC;
                pw.println(cr.toStringDetails());// TODO check format
            } else {
                rr = (Weka_module.RegressionResultsObject) weka.testClassifierFromFileSource(new File(weka.ARFFfile), modelFile, Main.isclassification);
                System.out.println(rr.predictions);
                pw.println("instance\tactual\tpredicted\terror\tprobability\n" + rr.predictions);
                rr.toStringDetails();
                pw.println(rr.toStringDetails());
            }
            pw.flush();
            pw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * class is missing, so we make predictions
     *
     * @param modelFile
     * @param TEST_FILE
     * @param TEST_RESULTS_FILE
     */
    public void TestingAndMakePredictions(String modelFile, String TEST_FILE, String TEST_RESULTS_FILE) {
        Weka_module weka = new Weka_module();
        weka.setCSVFile(new File(TEST_FILE));
        weka.csvToArff(Main.isclassification);
        weka.setDataFromArff();
        StringBuffer sb = weka.makePredictions(new File(weka.ARFFfile), modelFile);
        System.out.println("Test results stored in " + TEST_RESULTS_FILE);
        try {
            PrintWriter pw = new PrintWriter(new FileWriter(TEST_RESULTS_FILE));
            pw.println(sb);
            pw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
