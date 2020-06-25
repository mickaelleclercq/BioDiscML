/*
 * Get clinical and genes expression
 * Make some feature extraction
 */
package biodiscml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;
import utils.Weka_module;
import utils.utils.TableObject;
import static utils.utils.readTable;

/**
 *
 * @author Mickael
 */
public class AdaptDatasetToTraining {

    public static boolean debug = Main.debug;

    public AdaptDatasetToTraining() {
    }

    /**
     * create dataset with a determined class
     *
     * @param trainFile
     */
    public AdaptDatasetToTraining(String trainFile) {
        //create the adapted training file
        System.out.println("# Training file(s)");
        if (Main.doClassification) {
            createFileCompatibleForWeka(Main.classificationClassName, Main.hmTrainFiles, trainFile, Main.separator);
        } else {
            createFileCompatibleForWeka(Main.regressionClassName, Main.hmTrainFiles, trainFile, Main.separator);
        }
        //create the adapted tested file
        if (Main.doSampling) {
            System.out.println("## Apply sampling configuration");
            //if a test file is provided
            String trainAndTestFile = trainFile.replace("data_to_train", "all_data");
            String testFile = trainFile.replace("data_to_train.csv", "data_to_test.csv");
            Weka_module weka = new Weka_module();
            String trainSetRange = "";
            if (!Main.hmNewDataFiles.isEmpty()) {
                System.out.println("# Testing file(s)");
                if (Main.doClassification) {
                    createFileCompatibleForWeka(Main.classificationClassName, Main.hmNewDataFiles, testFile, Main.separator);
                } else {
                    createFileCompatibleForWeka(Main.regressionClassName, Main.hmNewDataFiles, testFile, Main.separator);
                }
                //if a test file is provided, we need to merge it to the train file and
                // split it again to preserve a compatible arff format between train and test sets
                trainSetRange = mergeTrainAndTestFiles(trainFile, testFile, trainAndTestFile);

            } else {// else we split
                // just rename the train file to a file that contains all train and test data
                new File(trainFile).renameTo(new File(trainAndTestFile));
            }

            //perform sampling
            weka.sampling(trainAndTestFile, trainFile, testFile, Main.isClassification, trainSetRange);
        }
        System.out.println("");

    }

    /**
     * get common ids between infiles
     *
     * @param al_tables
     * @return
     */
    public HashMap<String, String> getCommonIds(ArrayList<TableObject> al_tables) {
        HashMap<String, String> hm_ids = new HashMap<>();
        //if many infiles
        if (al_tables.size() > 0) {
            HashMap<String, Integer> hm_counts = new HashMap<>();
            //get all ids, count how many times each one is seen
            for (TableObject table : al_tables) {
                for (String s : table.hmIDsList.keySet()) {
                    s = s.toLowerCase();
                    if (hm_counts.containsKey(s)) {
                        int tmp = hm_counts.get(s);
                        tmp++;
                        hm_counts.put(s, tmp);
                    } else {
                        hm_counts.put(s, 1);
                    }
                }
            }
            //check number of times ids have been seen
            for (String s : hm_counts.keySet()) {
                if (hm_counts.get(s) == al_tables.size()) {
                    hm_ids.put(s, "");
                }
            }
        } else {//for one infile
            for (String s : al_tables.get(0).hmIDsList.keySet()) {
                hm_ids.put(s, "");
            }
        }

        return hm_ids;
    }

    private void createFileCompatibleForWeka(String theClass, HashMap<String, String> infiles, String outfile, String separator) {
        //convert hashmap to list
        String[] files = new String[infiles.size()];
        String[] prefixes = new String[infiles.size()];
        int cpt = 0;
        for (String f : infiles.keySet()) {
            files[cpt] = f;
            prefixes[cpt] = infiles.get(f);
            cpt++;
        }

        //load datasets of features
        if (debug) {
            System.out.println("loading files");
        }

        ArrayList<TableObject> al_tables = new ArrayList<>();
        int classIndex = -1;

        for (int i = 0; i < files.length; i++) {
            String file = files[i];
            if (debug) {
                System.out.println(file);
            }

            TableObject tbo = new TableObject(readTable(file, separator));
            //locate class
            if (tbo.containsClass(theClass)) {
                classIndex = i;
            }
            al_tables.add(tbo);
        }

        //extract class
        ArrayList<String> myClass = new ArrayList<>();
        try {
            myClass = al_tables.get(classIndex).getTheClass(theClass);
        } catch (Exception e) {
            System.err.println("[error] Class " + theClass + " not found. Error in the input file.");
            if (Main.debug) {
                e.printStackTrace();
            }
            System.exit(0);
        }
        //remove useless features having 100% the same value
        try {
            for (TableObject tbo : al_tables) {
                for (String s : tbo.getSortedHmDataKeyset()) {
                    HashMap<String, String> hm = new HashMap<>();
                    for (String value : tbo.hmData.get(s)) {
                        hm.put(value, value);
                    }
                    if (hm.size() == 1) {
                        tbo.hmData.remove(s);
                        if (hm.keySet().toArray()[0].equals("?")) {
                            System.out.println("Removing feature " + s + " "
                                    + "because 100% of values are missing");
                        } else {
                            System.out.println("Removing feature " + s + " "
                                    + "because 100% of values have the same value "
                                    + "{" + hm.keySet().toArray()[0] + "}");
                        }
                    }

                }
                cpt++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        //create outfile
        if (debug) {
            System.out.println("create outfile " + outfile);
        }
        try {
            PrintWriter pw = new PrintWriter(new FileWriter(outfile));
            ///////// PRINT HEADER
            pw.print(Main.mergingID);
            cpt = 0;
            for (TableObject tbo : al_tables) {
                for (String s : tbo.getSortedHmDataKeyset()) {
                    if (!Main.hmExcludedFeatures.containsKey(s)) {
                        if (!prefixes[cpt].isEmpty()) {
                            pw.print("\t" + prefixes[cpt] + "__" + s);
                        } else {
                            pw.print("\t" + s);
                        }
                    }
                }
                cpt++;
            }
            pw.println("\tclass");
            pw.flush();

            //search for ids present in all datasets
            HashMap<String, String> hm_ids = getCommonIds(al_tables);
            if (debug && al_tables.size() > 1) {
                System.out.println("Total number of common instances between files: " + hm_ids.size());
            } else {
                System.out.println("Total number of instances between files: " + hm_ids.size());
            }
            System.out.println("Total number of features: " + al_tables.get(0).hmData.size());

            ///////PRINT CONTENT
            TreeMap<String, Integer> tm = new TreeMap<>();
            tm.putAll(al_tables.get(0).hmIDsList);
            for (String id : tm.keySet()) {
                if (hm_ids.containsKey(id) && !id.equals(Main.mergingID.toLowerCase())) {
                    pw.print(id);
                    for (TableObject tbo : al_tables) {
                        int idIndex = tbo.hmIDsList.get(id);
                        for (String s : tbo.getSortedHmDataKeyset()) {
                            if (!Main.hmExcludedFeatures.containsKey(s)) { //if it is not a rejected feature
                                // print values and replace , by .
                                String out = tbo.hmData.get(s).get(idIndex).replace(",", ".");
                                if (out.isEmpty() || out.equals("NA") || out.equals("na")
                                        || out.equals("N/A") || out.equals("n/a")) {
                                    out = "?";
                                }
                                pw.print("\t" + out);
                            }
                        }
                    }
                    pw.print("\t" + myClass.get(al_tables.get(classIndex).hmIDsList.get(id)));
                    //pw.print("\t" + myClass.get(idIndex).replace("1", "true").replace("0", "false"));
                    pw.println();
                }
            }
            pw.flush();

            if (debug) {
                System.out.println("closing outfile " + outfile);
            }
            pw.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     *
     * @param outfile
     * @param replace
     * @return range of the train set (ex: 1-100)
     */
    private String mergeTrainAndTestFiles(String trainFile, String testFile, String trainAndTestFile) {
        int cpt = -1;
        try {
            //create train+set file
            PrintWriter pw = new PrintWriter(new FileWriter(trainAndTestFile));

            //read train
            BufferedReader br = new BufferedReader(new FileReader(trainFile));

            while (br.ready()) {
                pw.println(br.readLine());
                cpt++;
            }
            br.close();
            pw.flush();

            //read test
            br = new BufferedReader(new FileReader(testFile));
            br.readLine(); // skip header
            while (br.ready()) {
                pw.println(br.readLine());
            }
            br.close();
            pw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "1-" + cpt;
    }

}
