/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utils;

import biodiscml.BestModelSelectionAndReport;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

/**
 *
 * @author mickael
 */
public class UpSetR {

    public void creatUpSetRDataset(String featureSelectionFile, String predictionsResultsFile) {
        System.out.println("# create UpSetR file");
        String outfile = predictionsResultsFile.replace(".csv", ".UpSetR.csv");
        //create header
        ArrayList<String> featuresHeader = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(featureSelectionFile.replace(".csv", ".arff")));
            String line = br.readLine(); //relation
            br.readLine(); //empty line
            line = br.readLine(); // @attribute
            while (line.startsWith("@attribute")) {
                featuresHeader.add(line.replace("@attribute ", "").replaceAll(" \\w+$", ""));
                line = br.readLine();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        featuresHeader.remove(featuresHeader.size() - 1);
        featuresHeader.remove(0);

        //output
        try {
            BufferedReader br = new BufferedReader(new FileReader(predictionsResultsFile));
            PrintWriter pw = new PrintWriter(new FileWriter(outfile));
            pw.println("ID,"
                    + "TRAIN_10CV_MCC"
                    + ",TRAIN_LOOCV_MCC"
                    + ",TRAIN_BS_MCC"
                    + ",TEST_MCC"
                    + ",TRAIN_TEST_BS_MCC"
                    + ",AVG_MCC,"
                    + featuresHeader.toString().replace("[", "").replace("]", "").trim() + "");

            pw.flush();
            String line = br.readLine();

            while (br.ready()) {
                line = br.readLine();
                BestModelSelectionAndReport.classificationObject co = new BestModelSelectionAndReport.classificationObject(line);

                //get ID
                ArrayList<String> featureList = co.featureList;
                featureList.remove(featureList.size() - 1);
                featureList.remove(0);
                int[] tab = new int[featuresHeader.size()];
                try {
                    for (String index : featureList) {
                        tab[Integer.valueOf(index) - 2] = 1;
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
                String features = "";
                for (int i : tab) {
                    features += "," + i;
                }

                String out = "" + co.identifier + ""
                        + "," + co.hmValues.get("TRAIN_10CV_MCC")
                        + "," + co.hmValues.get("TRAIN_LOOCV_MCC")
                        + "," + co.hmValues.get("TRAIN_BS_MCC")
                        + "," + co.hmValues.get("TEST_MCC")
                        + "," + co.hmValues.get("TRAIN_TEST_BS_MCC")
                        + "," + co.hmValues.get("AVG_MCC") + features + "";
                pw.println(out);
            }
            pw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("UpSetR file: " + outfile);
    }

    public void creatUpSetRDatasetFromSignature(BestModelSelectionAndReport.classificationObject co_model,
            String featureSelectionFile, String predictionsResultsFile) {
        System.out.println("# create UpSetR file");
        String outfile = predictionsResultsFile.replace(".csv", ".UpSetR.csv");
        //create header
        ArrayList<String> alFeaturesOrder = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(featureSelectionFile.replace(".csv", ".arff")));
            String line = br.readLine(); //relation
            br.readLine(); //empty line
            line = br.readLine(); // @attribute            
            while (line.startsWith("@attribute")) {
                alFeaturesOrder.add(line.replace("@attribute ", "").replaceAll(" \\w+$", ""));
                line = br.readLine();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        alFeaturesOrder.remove(alFeaturesOrder.size() - 1);

        //retreive signature
        String featuresHeader = "";
        for (int i = 1; i < co_model.featureList.size() - 1; i++) {
            featuresHeader += "," + alFeaturesOrder.get(Integer.valueOf(co_model.featureList.get(i)) - 1);
        }

        //output
        try {
            //header
            BufferedReader br = new BufferedReader(new FileReader(predictionsResultsFile));
            PrintWriter pw = new PrintWriter(new FileWriter(outfile));
            pw.println("ID,"
                    + "TRAIN_10CV_MCC"
                    + ",TRAIN_LOOCV_MCC"
                    + ",TRAIN_BS_MCC"
                    + ",TEST_MCC"
                    + ",TRAIN_TEST_BS_MCC"
                    + ",AVG_MCC"
                    + featuresHeader);

            pw.flush();
            String line = br.readLine();

            //content
            while (br.ready()) {
                line = br.readLine();
                BestModelSelectionAndReport.classificationObject co_line = new BestModelSelectionAndReport.classificationObject(line);
                ArrayList<String> featureList = co_line.featureList;
                if (co_line.identifier.equals("trees.RandomForest_AUC_FB_19_0.9571_877")){
                    System.out.println("");
                }

                HashMap<String, String> hmFeaturesLine = new HashMap<>();
                for (int i = 1; i < co_line.featureList.size() - 1; i++) {
                    hmFeaturesLine.put(co_line.featureList.get(i),"");
                }
                String presence = "";
                for (int i = 1; i < co_model.featureList.size() - 1; i++) {                   
                    if (hmFeaturesLine.containsKey(co_model.featureList.get(i) + "")) {
                        presence += ",1";
                    } else {
                        presence += ",0";
                    }
                }
                String out = "" + co_line.identifier + ""
                        + "," + co_line.hmValues.get("TRAIN_10CV_MCC")
                        + "," + co_line.hmValues.get("TRAIN_LOOCV_MCC")
                        + "," + co_line.hmValues.get("TRAIN_BS_MCC")
                        + "," + co_line.hmValues.get("TEST_MCC")
                        + "," + co_line.hmValues.get("TRAIN_TEST_BS_MCC")
                        + "," + co_line.hmValues.get("AVG_MCC") + presence + "";
                pw.println(out);
            }
            pw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("UpSetR file: " + outfile);
    }

}
