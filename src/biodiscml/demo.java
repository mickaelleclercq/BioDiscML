/*
 *
 */
package biodiscml;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.HashMap;

/**
 *
 * @author mik
 */
public class demo {

    // public static String folder = "/home/mickael/ownCloud/";
    public static String folder = "E:\\cloud\\";
    // public static String folder = "C:\\Users\\Mickael\\ownCloud\\";

    public static void main(String[] args) {
        System.out.println("=== Demo mode ===");
        //trainingExecution();
        //testingExecution();
        bestModel();
        //benchmark();
    }

    private static void trainingExecution() {
        try {
            //String s[] = {"-config " + folder + "Data\\TCGA_PRAD\\datamining\\config.conf -train"};
            //String s[] = {"-config " + folder + "Data\\TCGA_PRAD\\datamining\\time\\config.conf -train"};
            //String s[] = {"-config " + folder + "Projects/loreal/VESPA/datamining//config_vespa.conf -train"};
            //String s[] = {"-config " + folder + "/Projects/Benjamin/Collaboration-CHUL-Quebec/1_Prostate/READY_TO_USE_for_Brute_force_X/datamining/2_Genes+clinic/config.conf -train"};

            //String s[] = {"-config config_example_2class.conf -train"};
            String s[] = {"-config " + folder + "Code\\BruteForceML\\benchmark\\CNS_test/config.conf -train"};
            //String s[] = {"-config " + folder + "Code/BruteForceML/benchmark/Benjamin_signature/config.conf -train"};
            //String s[] = {"-config " + folder + "Projects\\bacteria\\datamining\\config.conf -train"};
            Main.main(s);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void bestModel() {
        Main m = new Main();
        //demo Benjamin
//        m.wd = folder + "Projects\\Benjamin\\Collaboration - CHUL - Quebec\\1_Prostate\\READY_TO_USE_for_Brute_force_X\\datamining\\";
//        m.configFile = m.wd + "config.conf";
//        m.setConfiguration();
//        m.wd = folder + "Projects\\Benjamin\\Collaboration - CHUL - Quebec\\1_Prostate\\READY_TO_USE_for_Brute_force_X\\datamining\\";
//        String CLASSIFICATION_FILE = m.wd + m.project + "a.classification.data_to_train.csv"; // output of Training(), models performances
//        String TRAINING_RESULTS_FILE = m.wd + m.project + "c.classification.results.csv"; // output of Training(), models performances
//        String FEATURE_SELECTION_FILE = m.wd + m.project + "b.featureSelection.infoGain.csv"; // output of Training(), feature selection result
//        BestModelSelectionAndReport b = new BestModelSelectionAndReport(CLASSIFICATION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
//                "classification");
//        //mint
//        m.wd = folder + "Code/BruteForceML/benchmark/mint/";
//        m.configFile = m.wd + "config.conf";
//        m.setConfiguration();
//        m.wd = folder + "Code/BruteForceML/benchmark/mint/";
//        m.hmTrainingBestModelList.put("trees.RandomForest_AUC_BF_16_0.9571_77", "1");
//        String CLASSIFICATION_FILE = m.wd + m.project + "a.classification.data_to_train.csv"; // output of Training(), models performances
//        String TRAINING_RESULTS_FILE = m.wd + m.project + "c.classification.results.csv"; // output of Training(), models performances
//        String FEATURE_SELECTION_FILE = m.wd + m.project + "b.featureSelection.infoGain.csv"; // output of Training(), feature selection result
//        BestModelSelectionAndReport b = new BestModelSelectionAndReport(CLASSIFICATION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
//                "classification");
        //        //mint
        m.wd = folder + "Code/BruteForceML/benchmark/CNS_test/";
        m.configFile = m.wd + "config.conf";
        m.setConfiguration();
        m.wd = folder + "Code/BruteForceML/benchmark/CNS_test/";
        String CLASSIFICATION_FILE = m.wd + m.project + "a.regression.data_to_train.csv"; // output of Training(), models performances
        String TRAINING_RESULTS_FILE = m.wd + m.project + "c.regression.results.csv"; // output of Training(), models performances
        String FEATURE_SELECTION_FILE = m.wd + m.project + "b.featureSelection.RELIEFF.csv"; // output of Training(), feature selection result
        BestModelSelectionAndReport b = new BestModelSelectionAndReport(CLASSIFICATION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
                "regression");
        //bacteria
//        m.wd = folder + "Projects\\bacteria\\datamining\\";
//        m.configFile = m.wd + "config.conf";
//        m.setConfiguration();
//        m.wd = folder + "Projects\\bacteria\\datamining\\";
//        m.hmTrainingBestModelList.put("trees.RandomForest_AUC_B_25_0.9531_907", "1");
//        String CLASSIFICATION_FILE = m.wd + m.project + "a.classification.data_to_train.csv"; // output of Training(), models performances
//        String TRAINING_RESULTS_FILE = m.wd + m.project + "c.classification.results.csv"; // output of Training(), models performances
//        String FEATURE_SELECTION_FILE = m.wd + m.project + "b.featureSelection.infoGain.csv"; // output of Training(), feature selection result
//        BestModelSelectionAndReport b = new BestModelSelectionAndReport(CLASSIFICATION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
//                "classification");
//        //benjamin
//        m.wd = folder + "Code/BruteForceML/benchmark/Benjamin_prostate/";
//        m.configFile = m.wd + "config.conf";
//        m.setConfiguration();
//        m.wd = folder + "Code/BruteForceML/benchmark/Benjamin_prostate/";
//        String CLASSIFICATION_FILE = m.wd + m.project + "a.classification.data_to_train.csv"; // output of Training(), models performances
//        String TRAINING_RESULTS_FILE = m.wd + m.project + "c.classification.results.csv"; // output of Training(), models performances
//        String FEATURE_SELECTION_FILE = m.wd + m.project + "b.featureSelection.infoGain.csv"; // output of Training(), feature selection result
//        BestModelSelectionAndReport b = new BestModelSelectionAndReport(CLASSIFICATION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
//                "classification");
//       //golub
//        m.wd = folder + "Code/BruteForceML/benchmark/brain/";
//        m.configFile = m.wd + "config.conf";
//        m.setConfiguration();
//        m.wd = folder + "Code/BruteForceML/benchmark/brain/";
//        String CLASSIFICATION_FILE = m.wd + m.project + "a.classification.data_to_train.csv"; // output of Training(), models performances
//        String TRAINING_RESULTS_FILE = m.wd + m.project + "c.classification.results.csv"; // output of Training(), models performances
//        String FEATURE_SELECTION_FILE = m.wd + m.project + "b.featureSelection.infoGain.csv"; // output of Training(), feature selection result
//        BestModelSelectionAndReport b = new BestModelSelectionAndReport(CLASSIFICATION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
//                "classification");
//        //demo dreamchallenge
//        m.wd = folder + "Projects\\dreamchallenge\\proteogenomics\\SUB2_ML\\";
//        m.configFile = m.wd + "config.conf";
//        m.setConfiguration();
//        m.wd = folder + "Projects\\dreamchallenge\\proteogenomics\\SUB2_ML\\";
//        String CLASSIFICATION_FILE = m.wd + m.project + "a.regression.data_to_train.csv"; // output of Training(), models performances
//        String TRAINING_RESULTS_FILE = m.wd + m.project + "c.regression.results.csv"; // output of Training(), models performances
//        String FEATURE_SELECTION_FILE = m.wd + m.project + "b.featureSelection.RELIEFF.csv"; // output of Training(), feature selection result
//        BestModelSelectionAndReport b = new BestModelSelectionAndReport(CLASSIFICATION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
//                "regression");
        //demo vespa
//        m.wd = folder + "Projects\\loreal\\VESPA\\datamining\\";
//        m.configFile = m.wd + "config_vespa.conf";
//        m.setConfiguration();
//        m.wd = folder + "Projects\\loreal\\VESPA\\datamining\\";
//        String CLASSIFICATION_FILE = m.wd + m.project + "a.classification.data_to_train.csv"; // output of Training(), models performances
//        String TRAINING_RESULTS_FILE = m.wd + m.project + "c.classification.results.csv"; // output of Training(), models performances
//        String FEATURE_SELECTION_FILE = m.wd + m.project + "b.featureSelection.infoGain.csv"; // output of Training(), feature selection result
//        BestModelSelectionAndReport b = new BestModelSelectionAndReport(CLASSIFICATION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
//                "classification");
        //demo DATA
//        m.configFile = "config_example_2class.conf";
//        m.setConfiguration();
//        m.wd = "";
//        String CLASSIFICATION_FILE = m.wd + m.project + "a.classification.data_to_train.csv"; // output of Training(), models performances
//        String TRAINING_RESULTS_FILE = m.wd + m.project + "c.classification.results.csv"; // output of Training(), models performances
//        String FEATURE_SELECTION_FILE = m.wd + m.project + "b.featureSelection.infoGain.csv"; // output of Training(), feature selection result
//        BestModelSelectionAndReport b = new BestModelSelectionAndReport(CLASSIFICATION_FILE, FEATURE_SELECTION_FILE, TRAINING_RESULTS_FILE,
//                "classification");
    }

    public static void testingExecution() {
        try {
//            String s[] = {"-test -model gdx_data_.misc.VFI_-B0.6.txt.model "
//                + "-testfiles gdx.545patients.clinical.csv gdx.1742patients.expr.csv "
//                //+ "-prefixes clin expr "
//                + "-mergingID patient -separator \\t -classification -keyword BCR_sensor"};
            String s[] = {"-test -model " + folder + "Data\\TCGA_PRAD\\datamining\\TCGA_BCR_.misc.VFI_-B0.6.txt.model "
                + "-testfiles " + folder + "Data\\TCGA_PRAD\\datamining\\geneExpression.log2RUVg.csv"
                + " " + folder + "Data\\TCGA_PRAD\\datamining\\clinical_test.csv "
                + "-mergingID Patient -separator \\t -classification -keyword BCR_sensor"};
            for (String s1 : s) {
                System.out.print(s1);
            }
            System.out.println("");
            Main.main(s);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void benchmark() {
        try {
            PrintWriter pw = new PrintWriter(new FileWriter("benchmark_3.txt"));
            pw.println("FeaturesLimit\tAUC_Train\tAUC_Test");
            for (int i = 5; i <= 200; i = i + 5) {
                //train
                Main.hmTrainFiles = new HashMap<>();
                Main.needConfigFile = true;
                Main.testing = false;
                Main.training = true;
                System.out.println("\n-------------\nTRAIN " + i);
                String s[] = {"-config " + folder + "Data\\TCGA_PRAD\\datamining\\config_opt.conf -train"};
                Main.maxNumberOfFeaturesInModel = i;
                Main.main(s);
                String train = Main.bench_AUC;
                //test
                Main.hmTrainFiles = new HashMap<>();
                Main.configFile = "";
                Main.needConfigFile = false;
                Main.testing = true;
                Main.training = false;
                Main.project = "outfile";
                System.out.println("\n-------------\nTEST " + i);
                String s2[] = {"-test -model " + folder + "Data\\TCGA_PRAD\\datamining\\bench_.misc.VFI_-B0.6.txt.model "
                    + "-testfiles " + folder + "Data\\TCGA_PRAD\\datamining\\geneExpression.log2RUVg.csv"
                    + " " + folder + "Data\\TCGA_PRAD\\datamining\\clinical_test.csv "
                    + "-mergingID Patient -separator \\t -classification -keyword BCR_sensor"};
                Main.main(s2);
                String test = Main.bench_AUC;
                pw.println(i + "\t" + train + "\t" + test);
                pw.flush();
            }
        } catch (Exception e) {
        }
    }

}
