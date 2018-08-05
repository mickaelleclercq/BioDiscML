/*
 * Get all classifiers from weka libs
 */
package utils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import weka.classifiers.*;
import weka.core.Utils;

/**
 *
 * @author Administrator
 */
public class RetreiveClassifiersFromWeka {

    public static void main(String[] args) {
        ArrayList<String> al = new ArrayList<>();
        //get all possible models from lib folder
        File lib = new File("lib/weka/");
        for (File jar : lib.listFiles()) {
            if (jar.getName().endsWith("jar")) {
                try {
                    JarFile jarFile = new JarFile(jar);
                    Enumeration<JarEntry> entries = jarFile.entries();
                    while (entries.hasMoreElements()) {
                        JarEntry entry = entries.nextElement();
                        String n = entry.getName();
                        if (n.startsWith("weka/classifiers/")
                                && !n.contains("$")
                                && n.endsWith("class")
                                && n.split("/").length == 4
                                && (n.contains("/bayes/")
                                || n.contains("/functions/")
                                || n.contains("/rules/")
                                || n.contains("/trees/")
                                || n.contains("/lazy")
                                || n.contains("/misc"))) {
                            //System.out.println(jar.getName() + "\t" + n);
                            al.add(n);
                        }
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        Collections.sort(al);

        for (String classifier : al) {
            try {
                classifier = classifier.replace(".class", "").replace("/", ".").replace("weka/classifiers/", "");
                weka.classifiers.AbstractClassifier ac = (AbstractClassifier) Utils.forName(Classifier.class, classifier, args);
                String options = "";
                for (String option : ac.getOptions()) {
                    options += " " + option;
                }
                String cap = ac.getCapabilities().getClassCapabilities().toString();                
                System.out.println(
                        classifier.replaceAll("^weka.classifiers.", "").replaceFirst("\\.", "\t") + "\t"
                        + ac.getCapabilities().getAttributeCapabilities().toString()
                        .replaceAll("(Capabilities: \\[|attributes)", "")
                        .replace("Empty nominal ,", "")
                        .replace(" ,", ",").replace("  ", " ").replaceAll("\\].*", "").split("\n")[0] + "\t"
                        + ac.getCapabilities().getClassCapabilities().toString()
                        .replaceAll("(Capabilities: \\[|class)", "")
                        .replace("Missing  values", "")
                        .replace(" ,", ",").replace("  ", " ").replaceAll("\\].*", "").split("\n")[0].replaceAll(", $", " "));
                //+ac.getCapabilities().getClassCapabilities().toString());
//                if (cap.contains("Nominal class") && cap.contains("Numeric class")) {
//                    System.out.println(classifier.replaceAll("^weka.classifiers.", "ccmd=") + options);
//                    System.out.println(classifier.replaceAll("^weka.classifiers.", "rcmd=") + options);
//                } else if (cap.contains("Nominal class")) {
//                    System.out.println(classifier.replaceAll("^weka.classifiers.", "ccmd=") + options);
//                } else {
//                    System.out.println(classifier.replaceAll("^weka.classifiers.", "rcmd=") + options);
//                }
//
//                ac.listOptions();

            } catch (Exception e) {
                //e.printStackTrace();
            }

        }
    }

}
