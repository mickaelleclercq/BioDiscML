/*
 * Create roc curve graph
 */
package biodiscml;

import java.awt.Color;
import java.io.File;
import java.util.ArrayList;
import java.util.TreeMap;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.DeviationRenderer;
import org.jfree.chart.title.LegendTitle;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RectangleEdge;
import org.jfree.ui.RectangleInsets;
import utils.Weka_module;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;

/**
 *
 * @author Administrator
 */
public class rocCurveGraphs {

    public static void createRocCurvesWithConfidence(ArrayList<Object> alFoldsROCs, boolean classification, String outfile, String extension) {
        TreeMap<String, RocObject> tmFoldsROCs = new TreeMap<>();
        // retreive roc data of each class
        for (Object ROC : alFoldsROCs) {
            Evaluation eval;
            Instances data;
            if (classification) {
                Weka_module.ClassificationResultsObject cr = (Weka_module.ClassificationResultsObject) ROC;
                eval = cr.evaluation;
                data = cr.dataset;
            } else {
                Weka_module.RegressionResultsObject rr = (Weka_module.RegressionResultsObject) ROC;
                eval = rr.evaluation;
                data = rr.dataset;
            }

            boolean hasMoreClasses = true;
            int classIndex = 0;
            // for each class
            for (int i = 0; i < data.numClasses(); i++) {
                try {
                    //get class name
                    String className = data.classAttribute().value(classIndex);
                    //get roc object (or create it if first time)
                    RocObject ro;
                    if (tmFoldsROCs.containsKey(className)) {
                        ro = tmFoldsROCs.get(className);
                    } else {
                        ro = new RocObject();
                    }
                    //get roc values
                    Instances rocCurve = new ThresholdCurve().getCurve(eval.predictions(), classIndex);

                    //iterate through roc values and store them into Roc objects
                    ArrayList<Double> alTPR = new ArrayList<>();
                    ArrayList<Double> alFPR = new ArrayList<>();
                    for (String s : rocCurve.toString().split("\n")) {
                        if (!s.startsWith("@") && !s.trim().isEmpty()) {
                            alTPR.add(Double.parseDouble(s.split(",")[rocCurve.attribute("True Positive Rate").index()]));
                            alFPR.add(Double.parseDouble(s.split(",")[rocCurve.attribute("False Positive Rate").index()]));
                        }
                    }
                    ro.TPRvalues.add(alTPR);
                    ro.FPRvalues.add(alFPR);
                    tmFoldsROCs.put(className, ro);
                } catch (Exception e) {
                    e.printStackTrace();
                    hasMoreClasses = false;
                }
                classIndex++;
            }
        }
        //create picture
        XYDataset data = createRocDataset(tmFoldsROCs);
        String title = outfile.replace(Main.project + ".", "").replace(Main.wd, "");
        JFreeChart chart = createChart(data, title);
        exportChartToPNG(chart, outfile + extension);
    }

    public static XYDataset createRocDataset(TreeMap<String, RocObject> tm) {

        XYSeriesCollection dataset = new XYSeriesCollection();

        //for each fold
        for (String classe : tm.keySet()) {
            XYSeries serie = new XYSeries(classe, false);
            RocObject ro = tm.get(classe);
            for (int i = 0; i < ro.FPRvalues.size(); i++) {
                //XYSeries serie = new XYSeries(classe + "_fold" + i,false);
                for (int j = 0; j < ro.FPRvalues.get(i).size(); j++) {
                    serie.add(ro.FPRvalues.get(i).get(j), ro.TPRvalues.get(i).get(j));
                }
                //dataset.addSeries(serie);
            }
            dataset.addSeries(serie);

        }

        return dataset;

    }

    public static void exportChartToPNG(JFreeChart chart, String outfile) {
        try {
            ChartUtilities.saveChartAsPNG(new File(outfile), chart, 1024, 768);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Creates a chart.
     *
     * @param dataset the data for the chart.
     * @return a chart.
     */
    private static JFreeChart createChart(XYDataset dataset, String title) {
        // create the chart...
        JFreeChart chart = ChartFactory.createXYLineChart(
                title, // chart title
                "False positive rate (1 - specificity)", // x axis label
                "True positive rate (sensitivity)", // y axis label
                dataset, // data
                PlotOrientation.VERTICAL,
                true, // include legend
                true, // tooltips
                false // urls
        );

        chart.setBackgroundPaint(Color.white);
        LegendTitle legend = chart.getLegend();
        legend.setPosition(RectangleEdge.RIGHT);

        // get a reference to the plot for further customisation...
        XYPlot plot = (XYPlot) chart.getPlot();
        plot.setBackgroundPaint(Color.white);
        plot.setAxisOffset(new RectangleInsets(5.0, 5.0, 5.0, 5.0));
        plot.setDomainGridlinePaint(Color.black);
        plot.setRangeGridlinePaint(Color.black);

        //deviation renderer
        //color codes: http://htmlcolorcodes.com/
        DeviationRenderer renderer = new DeviationRenderer(true, false);

//        renderer.setSeriesStroke(0, new BasicStroke(3.0f, BasicStroke.CAP_ROUND,
//                BasicStroke.JOIN_ROUND));
//        renderer.setSeriesStroke(0, new BasicStroke(3.0f));
//        renderer.setSeriesStroke(1, new BasicStroke(3.0f));
//        renderer.setSeriesFillPaint(0, new Color(255, 118, 118)); //light red
//        renderer.setSeriesFillPaint(1, new Color(118, 180, 255)); //light blue
        plot.setRenderer(renderer);

//        //smooth line renderer
//        XYSplineRenderer rend = new XYSplineRenderer();
//        rend.setPrecision(2);
//        plot.setRenderer(rend);
        //Axis modifications
        NumberAxis yAxis = (NumberAxis) plot.getRangeAxis();
        yAxis.setRange(-0.01, 1.01);

        NumberAxis xAxis = (NumberAxis) plot.getDomainAxis();
        xAxis.setRange(-0.01, 1.01);

        return chart;
    }

    private static class RocObject {

        public String classe;
        public ArrayList<ArrayList<Double>> TPRvalues;
        public ArrayList<ArrayList<Double>> FPRvalues;

        public RocObject() {
            TPRvalues = new ArrayList<>();
            FPRvalues = new ArrayList<>();

        }
    }

}
