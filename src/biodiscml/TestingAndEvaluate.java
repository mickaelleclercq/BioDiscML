/*
 * Test and evaluate
 */
package biodiscml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.HashMap;
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

        weka.csvToArff(Main.isClassification);
        getClassesAndFeaturesFromModel(modelFile, weka);
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
    public void getClassesAndFeaturesFromModel(String modelFile, Weka_module weka) {
        //get classes from model
        String classes = weka.getClassesFromClassifier(modelFile, false).get(0);
        HashMap<String, String> hmFeatures = weka.getFullFeaturesFromClassifier(modelFile);

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
                    if (line.startsWith("@attribute Instance ")) {
                        pw.println(line);
                    } else if (line.startsWith("@attribute ") && line.contains("{") && line.contains("}")) {
                        String featureName = line.replaceAll("\\{.*", "").replace(("@attribute"), "").trim();
                        pw.println("@attribute " + featureName + " " + hmFeatures.get(featureName));
                    } else {
                        pw.println(line);
                    }
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
