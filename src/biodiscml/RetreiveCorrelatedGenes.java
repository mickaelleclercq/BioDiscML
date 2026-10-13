package biodiscml;

/*
 * Retrieve correlated genes
 */


import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentSkipListMap;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import utils.utils.TableObject;
import static utils.utils.convertStringListToDoubles;
import static utils.utils.newDecimalFormat;
import static utils.utils.readTable;

/**
 *
 * @author Mickael
 */
public class RetreiveCorrelatedGenes {

    public static void correlation(String signatureFile, String allFeaturesFile, String bestFeatures) {
        TableObject tboAllFeatures = new TableObject(readTable(allFeaturesFile));
        TableObject tboSignature = new TableObject(readTable(signatureFile));
        TableObject tboBestFeatures = new TableObject(readTable(bestFeatures));

        //store best features
        HashMap<String, String> hmBestFeatures = new HashMap<>();
        for (String gene : tboBestFeatures.hmData.keySet()) {
            hmBestFeatures.put(gene, gene);
        }

        System.out.println("# Looking for correlated genes and mutual information");

        tboSignature.hmData.keySet().parallelStream().forEach((signatureGene) -> {
            //iterate through all signature genes
            // DecimalFormat is not thread-safe: one instance per task
            DecimalFormat df = newDecimalFormat();
            double max = 0;
            for (String otherGene : tboAllFeatures.hmData.keySet()) { //iterate through all features
                if (!otherGene.equals(signatureGene)) { //do not compare the signature gene against itself
                    try {
                        double corr = new SpearmansCorrelation().correlation(
                                convertStringListToDoubles(tboSignature.hmData.get(signatureGene)),
                                convertStringListToDoubles(tboAllFeatures.hmData.get(otherGene)));
                        if (corr >= Main.spearmanCorrelation_upper || corr <= Main.spearmanCorrelation_lower) {
                            boolean bf = false;
                            if (hmBestFeatures.containsKey(otherGene)) {
                                bf = true;
                            }
                            System.out.println(signatureGene + "\t" + otherGene + "\t" + df.format(corr) + "\t" + bf);
                        }
                        if (corr > max) {
                            max = corr;
                        }
                    } catch (Exception e) {
                    }
                }
            }
        });
    }

    public static TreeMap<String, Double> spearmanCorrelation(String signatureFile, String allFeaturesFile) {
        // Filled concurrently by the parallel stream below: TreeMap is not thread-safe
        // (concurrent puts can corrupt it into a cycle, then iterating it never ends)
        ConcurrentSkipListMap<String, Double> tm = new ConcurrentSkipListMap<>();

        TableObject tboAllFeatures = new TableObject(readTable(allFeaturesFile));
        TableObject tboSignature = new TableObject(readTable(signatureFile));

        tboSignature.hmData.keySet().parallelStream().forEach((signatureGene) -> {
            //iterate through all signature genes
            // DecimalFormat is not thread-safe: one instance per task
            DecimalFormat df = newDecimalFormat();
            double max = 0;
            for (String otherGene : tboAllFeatures.hmData.keySet()) { //iterate through all features

                if (!otherGene.equals(signatureGene)) { //do not compare the signature gene against itself
                    try {
                        double corr = new SpearmansCorrelation().correlation(
                                convertStringListToDoubles(tboSignature.hmData.get(signatureGene)),
                                convertStringListToDoubles(tboAllFeatures.hmData.get(otherGene)));
                        if (corr >= Main.spearmanCorrelation_upper || corr <= Main.spearmanCorrelation_lower) {
                            tm.put(signatureGene + "\t" + df.format(corr) + "\t" + otherGene, corr);
                        }
                        if (corr > max) {
                            max = corr;
                        }
                    } catch (Exception e) {
                    }
                }
            }
        });

        return new TreeMap<>(tm);
    }

    public static TreeMap<String, Double> pearsonCorrelation(String signatureFile, String allFeaturesFile) {
        // Filled concurrently by the parallel stream below: TreeMap is not thread-safe
        // (concurrent puts can corrupt it into a cycle, then iterating it never ends)
        ConcurrentSkipListMap<String, Double> tm = new ConcurrentSkipListMap<>();

        TableObject tboAllFeatures = new TableObject(readTable(allFeaturesFile));
        TableObject tboSignature = new TableObject(readTable(signatureFile));

        tboSignature.hmData.keySet().parallelStream().forEach((signatureGene) -> {
            //iterate through all signature genes
            // DecimalFormat is not thread-safe: one instance per task
            DecimalFormat df = newDecimalFormat();
            double max = 0;
            for (String otherGene : tboAllFeatures.hmData.keySet()) { //iterate through all features

                if (!otherGene.equals(signatureGene)) { //do not compare the signature gene against itself
                    try {
                        double corr = new PearsonsCorrelation().correlation(
                                convertStringListToDoubles(tboSignature.hmData.get(signatureGene)),
                                convertStringListToDoubles(tboAllFeatures.hmData.get(otherGene)));
                        if (corr >= Main.pearsonCorrelation_upper || corr <= Main.pearsonCorrelation_lower) {
                            tm.put(signatureGene + "\t" + df.format(corr) + "\t" + otherGene, corr);
                        }
                        if (corr > max) {
                            max = corr;
                        }
                    } catch (Exception e) {
                    }
                }
            }
        });

        return new TreeMap<>(tm);
    }

}
