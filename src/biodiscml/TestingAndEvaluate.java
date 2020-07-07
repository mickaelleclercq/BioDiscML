/*
 * Test and evaluate
 */
package biodiscml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
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
    public void TestingAndEvaluate(String modelFile, String TEST_FILE,
            String TEST_RESULTS_FILE, boolean missingClass) {
        Weka_module weka = new Weka_module();
        weka.setCSVFile(new File(TEST_FILE));

        if (missingClass) {
            weka.csvToArff(Main.isClassification);
            getClassesFromModel(modelFile, weka);
        } else {
            weka.csvToArff(Main.isClassification);
        }
        weka.setDataFromArff();

        Weka_module.ClassificationResultsObject cr = null;
        Weka_module.RegressionResultsObject rr = null;

        try {
            PrintWriter pw = new PrintWriter(new FileWriter(TEST_RESULTS_FILE));
            System.out.println("Test results stored in " + TEST_RESULTS_FILE);

            if (Main.isClassification) {
                cr = (Weka_module.ClassificationResultsObject) weka.testClassifierFromFileSource(new File(weka.ARFFfile), modelFile, Main.isClassification);
                System.out.println("instance\tactual\tpredicted\terror\t" + cr.classes + "\n" + cr.predictions);
                pw.println("instance\tactual\tpredicted\terror\t" + cr.classes + "\n" + cr.predictions);
                if (!missingClass) {
                    System.out.println(cr.toStringDetailsTesting());
                    pw.println(cr.toStringDetailsTesting());
                }
            } else {
                rr = (Weka_module.RegressionResultsObject) weka.testClassifierFromFileSource(new File(weka.ARFFfile), modelFile, Main.isClassification);
                System.out.println("instance\tactual\tpredicted\terror\t" + cr.classes + "\n" + rr.predictions);
                pw.println("instance\tactual\tpredicted\terror\t" + cr.classes + "\n" + rr.predictions);
                if (!missingClass) {
                    System.out.println(rr.toStringDetails());
                    pw.println(rr.toStringDetails());
                }
            }
            pw.flush();
            pw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     *
     *
     * @param modelFile
     * @param TEST_FILE
     * @param TEST_RESULTS_FILE
     */
    public void getClassesFromModel(String modelFile, Weka_module weka) {
        //get classes from model
        String classes = weka.getClassesFromClassifier(modelFile, false).get(0);

        //load model
        File arffFile = new File(weka.ARFFfile);
        File newarffFile = new File(weka.ARFFfile + ".tmp");
        try {
            BufferedReader br = new BufferedReader(new FileReader(arffFile));
            PrintWriter pw = new PrintWriter(new FileWriter(newarffFile));
            String line = "";
            while (br.ready()) {
                line = br.readLine();
                if (line.contains("@attribute class {")) {
                    pw.println(classes);
                } else {
                    pw.println(line);
                }
            }
            pw.close();
            br.close();
            arffFile.delete();
            newarffFile.renameTo(arffFile);
        } catch (Exception e) {
            if (Main.debug) {
                e.printStackTrace();
            }
        }

    }

}
