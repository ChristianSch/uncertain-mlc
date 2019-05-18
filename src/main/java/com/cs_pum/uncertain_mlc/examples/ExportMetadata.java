package com.cs_pum.uncertain_mlc.examples;

import com.cs_pum.uncertain_mlc.common.LabelMetadata;
import com.cs_pum.uncertain_mlc.common.LabelSpaceReduction;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import put.mlc.classifiers.pcc.PCC;
import put.mlc.classifiers.pcc.inference.ExhaustiveInference;
import put.mlc.classifiers.pcc.inference.Inference;
import put.mlc.examples.common.Experiment;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;


public class ExportMetadata extends Experiment {

    Inference inference;
    HashMap<String, Integer> labelCounts;
    HashMap<String, Boolean> labelsFirst;

    public ExportMetadata() {
        String[] datasets = {
                "emotions",
                "enron",
                "mediamill",
                "medical",
                "scene",
                "tmc2007-500",
                "yeast",
                "IMDB-F",
                "SLASHDOT-F",
                "OHSUMED-F",
                "REUTERS-K500-EX2"
        };

        labelCounts = new HashMap<String, Integer>();
        labelCounts.put("emotions", 6);
        labelCounts.put("enron", 53);
        labelCounts.put("mediamill", 101);
        labelCounts.put("medical", 45);
        labelCounts.put("scene", 6);
        labelCounts.put("tmc2007-500", 22);
        labelCounts.put("yeast", 14);
        labelCounts.put("IMDB-F", 28);
        labelCounts.put("OHSUMED-F", 23);
        labelCounts.put("SLASHDOT-F", 22);
        labelCounts.put("REUTERS-K500-EX2", 14);

        labelsFirst = new HashMap<String, Boolean>();
        labelsFirst.put("emotions", false);
        labelsFirst.put("enron", false);
        labelsFirst.put("mediamill", false);
        labelsFirst.put("medical", false);
        labelsFirst.put("scene", false);
        labelsFirst.put("tmc2007-500", false);
        labelsFirst.put("yeast", false);
        labelsFirst.put("IMDB-F", true);
        labelsFirst.put("OHSUMED-F", true);
        labelsFirst.put("SLASHDOT-F", true);
        labelsFirst.put("REUTERS-K500-EX2", true);

        this.initDataSetsList(datasets);
        this.inference = new ExhaustiveInference();
    }

    @Override
    public void runExperiment() throws Exception {
        for (String dataset : this.dataSets) {
            MultiLabelInstances data;
            FileInputStream fileStream = null;
            File arffFile = new File("datasets/" + dataset + ".arff");
            fileStream = new FileInputStream(arffFile);
            boolean labelsFirst = (boolean) this.labelsFirst.get(dataset);

            data = new MultiLabelInstances((InputStream) fileStream,
                    (int) this.labelCounts.get(dataset),
                    labelsFirst);

            if (data.getNumLabels() > 10) {
                System.out.println("reduced labels to 10");
                data = LabelSpaceReduction.reduceLabelSpace(data, 10, labelsFirst);
            }

            Instances workingSet = new Instances(data.getDataSet());
            String[] labelNames = new String[data.getLabelsMetaData().getLabelNames().size()];
            data.getLabelsMetaData().getLabelNames().toArray(labelNames);

            System.out.println("labels:");
            System.out.println(Utils.arrayToString(labelNames));
            System.out.println("label frequencies:");
            System.out.println(Utils.arrayToString(LabelMetadata.getLabelFrequencies(data, labelsFirst)));
            System.out.println("label counts:");
            System.out.println(Utils.arrayToString(LabelMetadata.getLabelCounts(data, labelsFirst)));
            System.out.println("label cardinality:");
            System.out.println(data.getCardinality());

            System.out.println("no labels:");
            System.out.println(data.getNumLabels());
            System.out.println("no instances");
            System.out.println(data.getNumInstances());
            System.out.println("no features");
            System.out.println(data.getFeatureAttributes().size());
        }
    }

    public static void main(String[] args) throws Exception {
        // System.setErr(new PrintStream(new File("errors.txt")));
        Experiment experiment = new ExportMetadata();
        experiment.runExperiment();
    }
}
