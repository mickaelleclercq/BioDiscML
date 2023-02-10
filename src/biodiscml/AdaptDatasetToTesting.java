/*
 * Get clinical and genes expression
 * Make some feature extraction
 */
package biodiscml;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;
import utils.Weka_module;
import utils.utils.TableObject;
import static utils.utils.readTable;
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;

/**
 *
 * @author Mickael
 */
public class AdaptDatasetToTesting {

    public static boolean debug = Main.debug;
    public static boolean missingClass = false;

    public AdaptDatasetToTesting() {
    }

    public boolean isMissingClass() {
        return missingClass;
    }

    /**
     * create dataset with a determined class
     *
     * @param theClass
     * @param infiles
     * @param outfile
     */
    public AdaptDatasetToTesting(String theClass, HashMap<String, String> infiles, String outfile, String separator, String model) {
        //get model features
        Weka_module weka = new Weka_module();
        ArrayList<String> alModelFeatures = weka.getFeaturesFromClassifier(model); //features in the right order
        HashMap<String, String> hmModelFeatures = new HashMap<>();//indexed hashed features
        System.out.println("# Model features: ");

        Boolean voteModel = false;
        try {
            voteModel = (((Classifier) SerializationHelper.read(model)).getClass().toString().contains("weka.classifiers.meta.Vote"));
        } catch (Exception e) {
            if (Main.debug) {
                e.printStackTrace();
            }
        }
        if (voteModel) {
            System.out.println("    Combined Vote model");
        }
        for (String f : alModelFeatures) {
            if (!f.startsWith("Model") && !f.startsWith("class")) {
                hmModelFeatures.put(f, f);
            }
            System.out.println("\t" + f);
        }
        hmModelFeatures.put("class", "class");

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
            String f = "";
            for (String s : infiles.keySet()) {
                f += " " + s;
            }
            System.out.println("Class " + theClass + " not found in " + f + ". Class values are filled by ?");
            missingClass = true;
        }

        // replace spaces by _ in class
        for (int i = 0; i < myClass.size(); i++) {
            String c = myClass.get(i).replace(" ", "_");
            myClass.set(i, c);
        }

        //remove feature that are not needed by the model
        //create outfile
        System.out.println("create outfile " + outfile);

        HashMap<String, ArrayList<String>> hmOutput = new HashMap<>();
        try {
            PrintWriter pw = new PrintWriter(new FileWriter(outfile));
            ///////// PRINT HEADER
            //pw.print(Main.mergingID);
            hmOutput.put(Main.mergingID, new ArrayList<>());
            for (int i = 0; i < al_tables.size(); i++) {
                TableObject tbo = al_tables.get(i);
                for (String s : tbo.hmData.keySet()) {
                    String head = null;
                    if (!prefixes[i].isEmpty()) {
                        head = prefixes[i] + "__" + s;
                    } else {
                        head = s;
                    }
                    if (hmModelFeatures.containsKey(head)) {
                        //pw.print("\t" + head);
                        hmOutput.put(head, new ArrayList<>());
                    }
                }
            }

            //pw.println("\tclass");
            hmOutput.put("class", new ArrayList<>());
            //pw.flush();

            //search for ids present in all datasets
            HashMap<String, String> hm_ids = getCommonIds(al_tables);
            if (al_tables.size() > 1) {
                System.out.println("Total number of common instances between files:" + hm_ids.size());
            } else {
                System.out.println("Total number of instances between files:" + hm_ids.size());
            }

            ///////PREPARE CONTENT FOR PRINTING
            TreeMap<String, Integer> tm = new TreeMap<>();
            tm.putAll(al_tables.get(0).hmIDsList);
            for (String id : tm.keySet()) {
                if (hm_ids.containsKey(id)) { //if sample exist in all files
                    //pw.print(id);
                    hmOutput.get(Main.mergingID).add(id);
                    for (int i = 0; i < al_tables.size(); i++) {
                        TableObject tbo = al_tables.get(i);
                        int idIndex = tbo.hmIDsList.get(id); //get index for sample
                        for (String s : tbo.hmData.keySet()) { //for all features
                            String feature = null;
                            if (!prefixes[i].isEmpty()) {
                                feature = prefixes[i] + "__" + s;
                            } else {
                                feature = s;
                            }
                            if (hmModelFeatures.containsKey(feature)) {
                                //pw.print("\t" + tbo.hmData.get(s).get(idIndex));
                                // print values and replace , per .
                                String out = tbo.hmData.get(s).get(idIndex).replace(",", ".").trim();
                                if (out.isEmpty()) {
                                    out = Main.missingValueToReplace;
                                }
                                hmOutput.get(feature).add(out);
                            }
                        }
                    }

                    if (missingClass) {
                        hmOutput.get("class").add("?");
                    } else {
                        //pw.print("\t" + myClass.get(al_tables.get(classIndex).hmIDsList.get(id)));
                        hmOutput.get("class").add(myClass.get(al_tables.get(classIndex).hmIDsList.get(id)));
                        //pw.print("\t" + myClass.get(idIndex).replace("1", "true").replace("0", "false"));
                    }

                    //pw.println();
                }
            }
            //PRINTING CONTENT IN THE RIGHT ORDER
            if (voteModel) {
                //ensure class is at the end of the list
                alModelFeatures = new ArrayList<>();
                for (String feature : hmModelFeatures.keySet()) {
                    if (!feature.equals("class")) {
                        alModelFeatures.add(feature);
                    }
                }
                alModelFeatures.add("class");
            }
            //header
            pw.print(Main.mergingID);
            for (String feature : alModelFeatures) {
                pw.print("\t" + feature);
            }
            pw.println();
            pw.flush();
            //content
            for (int i = 0; i < hm_ids.size(); i++) {//for every instance
                pw.print(hmOutput.get(Main.mergingID).get(i));
                for (String feature : alModelFeatures) {
                    try {
                        pw.print("\t" + hmOutput.get(feature).get(i));
                    } catch (Exception e) {
                        System.out.println("Feature "+feature+" is not present in the test set. Replacing with missing data");
                        pw.print("\t?");
                    }
                }
                pw.println();
            }
            pw.flush();
            pw.close();
            if (debug) {
                System.out.println("closing outfile " + outfile);
            }
            pw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * get common ids between infiles
     *
     * @param al_tables
     * @return
     */
    public static HashMap<String, String> getCommonIds(ArrayList<TableObject> al_tables) {
        HashMap<String, String> hm_ids = new HashMap<>();
        //if many infiles
        if (al_tables.size() > 0) {
            HashMap<String, Integer> hm_counts = new HashMap<>();
            //get all ids, count how many times each one is seen
            for (TableObject table : al_tables) {
                for (String s : table.hmIDsList.keySet()) {
                    //s = s.toLowerCase();
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

}
