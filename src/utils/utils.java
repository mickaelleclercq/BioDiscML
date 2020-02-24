/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utils;

import biodiscml.Main;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.TreeMap;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

/**
 *
 * @author Administrator
 */
public class utils {

    public static HashMap<String, ArrayList<String>> convertDataColumnsToHashMap(ArrayList<String[]> altable) {
        HashMap<String, ArrayList<String>> hmData = new HashMap<>();
        for (int i = 1; i < altable.get(0).length; i++) { //for column
            String header = altable.get(0)[i]; //get column header
            ArrayList<String> al = new ArrayList<>(); //data list of the column
            for (int j = 1; j < altable.size(); j++) { //for each value in the column
                try {
                    al.add(altable.get(j)[i]);
                } catch (Exception e) {
                    al.add("");
                }
            }
            hmData.put(header, al);
        }
        return hmData;
    }

    public static ArrayList<String[]> readTable(String file, String separator) {
        if (separator.isEmpty()) {
            separator = detectSeparator(file);
        }
        ArrayList<String[]> altable = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            while (br.ready()) {
                altable.add(br.readLine().split(separator));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return altable;
    }

    public static ArrayList<String[]> readTable(String file) {
        String separator = detectSeparator(file);
        if (Main.debug) {
            System.out.println("reading table " + file);
        }
        ArrayList<String[]> altable = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            while (br.ready()) {
                altable.add(br.readLine().split(separator));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return altable;
    }

    public static String[][] transposeMatrix(String[][] m) {
        String[][] temp = new String[m[0].length][m.length];
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[0].length; j++) {
                temp[j][i] = m[i][j];
            }
        }
        return temp;
    }

    /**
     *
     * @param doubles
     * @return
     */
    public static double[] convertStringListToDoubles(ArrayList<String> doubles) {
        double[] ret = new double[doubles.size()];
        Iterator<String> iterator = doubles.iterator();
        int i = 0;
        while (iterator.hasNext()) {
            ret[i] = Double.valueOf(iterator.next());
            i++;
        }
        return ret;
    }

    /**
     * auto detection of the delimiter
     *
     * @param infile
     * @return
     */
    public static String detectSeparator(String infile) {
        if (Main.debug) {
            //System.out.println("Delimiter not specified, auto-detection 10 first lines among these delimiters: \"\\t\" \" \" \";\" \",\" \"~\" \":\" \"/\" \"\\|\"");
            System.out.println("Delimiter not specified, auto-detection in the 10 first lines...");

        }
        String delimiter = "";
        String potentialDelimiters[] = {"\\t", " ", ";", ",", "~", ":", "/", "\\|"};
        for (String potentialDelimiter : potentialDelimiters) {
            try {
                BufferedReader br = new BufferedReader(new FileReader(infile));
                boolean sameNumberAsPreviousLine = true;
                String line = br.readLine();
                int init = line.split(potentialDelimiter).length;
                if (init > 1) {
                    int cpt = 0;
                    while (br.ready() & sameNumberAsPreviousLine && cpt < 10) {
                        cpt++;
                        line = br.readLine();
                        if (!line.trim().isEmpty()) {
                            int split = line.split(potentialDelimiter).length;
                            if (split == init) {
                                init = split;
                            } else {
                                sameNumberAsPreviousLine = false;
                            }
                        }
                    }
                    if (sameNumberAsPreviousLine) {
                        delimiter = potentialDelimiter;
                        break;
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        if (delimiter.isEmpty()) {
            System.err.println("[error] CSV separator not detected. Guessing there is just one column. "
                    + "If not, change your CSV separator to a standard one (ex: tabulation or comma), or check the file consistency");
            delimiter = " ";
        }
        if (Main.debug) {
            System.out.println("Delimiter found:" + delimiter);
        }
        return delimiter;
    }

    public static void copyFileUsingStream(File source, File dest) throws IOException {
        InputStream is = null;
        OutputStream os = null;
        try {
            is = new FileInputStream(source);
            os = new FileOutputStream(dest);
            byte[] buffer = new byte[1024];
            int length;
            while ((length = is.read(buffer)) > 0) {
                os.write(buffer, 0, length);
            }
        } finally {
            is.close();
            os.close();
        }
    }

    public static void copyFileUsingChannel(File source, File dest) throws IOException {
        FileChannel sourceChannel = null;
        FileChannel destChannel = null;
        try {
            sourceChannel = new FileInputStream(source).getChannel();
            destChannel = new FileOutputStream(dest).getChannel();
            destChannel.transferFrom(sourceChannel, 0, sourceChannel.size());
        } finally {
            sourceChannel.close();
            destChannel.close();
        }
    }

    public static String arrayToString(String[] array, String separator) {
        String s = "";
        for (int i = 0; i < array.length; i++) {
            if (array[i] != null) {
                s += array[i] + separator;
            }
        }
        return s.substring(0, s.length() - 1);
    }

    public static String arrayToString(ArrayList<Integer> array, String separator) {
        String s = "";
        for (int i = 0; i < array.size(); i++) {
            s += array.get(i) + separator;
        }
        return s.substring(0, s.length() - 1);
    }

    /**
     * Table Object to manipulate table
     */
    public static class TableObject {

        public HashMap<String, ArrayList<String>> hmData = new HashMap<>();//<ColName,Values>
        public HashMap<String, Integer> hmIDsList = new HashMap<>(); //<ID,index>

        public TableObject(ArrayList<String[]> altable) {
            if (Main.debug) {
                System.out.println("converting table");
            }
            hmData = convertDataColumnsToHashMap(altable);
            int ID_index = getIdIndex(altable);
            for (int i = 1; i < altable.size(); i++) { //for each ID
                hmIDsList.put(altable.get(i)[ID_index].toLowerCase(), i - 1);
            }
        }

        public ArrayList<String> getTheClass(String theclass) {
            ArrayList<String> toreturn = hmData.get(theclass);
            hmData.remove(theclass);
            return toreturn;
        }

        public boolean containsClass(String theClass) {
            return hmData.containsKey(theClass);
        }

        public int getIdIndex(ArrayList<String[]> altable) {
            int cpt = 0;
            for (String s : altable.get(0)) {
                if (s.equals(Main.mergingID)) {
                    return cpt;
                } else {
                    cpt++;
                }
            }
            return 0;
        }

        public ArrayList<String> getSortedHmDataKeyset() {
            ArrayList<String> al = new ArrayList<>();
            TreeMap<String, ArrayList<String>> tm = new TreeMap<>();
            tm.putAll(hmData);
            for (String key : tm.keySet()) {
                al.add(key);
            }
            return al;
        }
    }

    /**
     * calculate mean of an array of doubles
     *
     * @param al
     * @return
     */
    public static String getMean(ArrayList<Double> al) {

        if (!al.isEmpty()) {
            double d[] = new double[al.size()];
            for (int i = 0; i < al.size(); i++) {
                if (!al.get(i).isNaN()) {
                    d[i] = (double) al.get(i);
                }
            }

            Mean m = new Mean();
            DecimalFormat df = new DecimalFormat();
            df.setMaximumFractionDigits(3);
            DecimalFormatSymbols dfs = new DecimalFormatSymbols();
            dfs.setDecimalSeparator('.');
            df.setDecimalFormatSymbols(dfs);
            return df.format(m.evaluate(d));
        } else {
            return "";
        }
    }

}
