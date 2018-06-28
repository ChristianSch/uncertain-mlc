package com.cs_pum.uncertain_mlc.examples;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

import com.cs_pum.uncertain_mlc.common.LabelSpaceReduction;
import com.cs_pum.uncertain_mlc.losses.UncertainHammingLoss;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.Evaluation;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.Measure;

import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import put.mlc.classifiers.pcc.PCC;
import put.mlc.classifiers.pcc.inference.Inference;
import put.mlc.classifiers.pcc.inference.ExhaustiveInference;
import put.mlc.examples.common.Experiment;
import put.mlc.measures.ZeroOneLossMeasure;


/**
 * This class implements the experiment reported in the respective (original) publications introducing
 * the uncertainty based hamming loss.
 *
 * @author Christian Schulze
 * @since  2018-06-25
 */
public class UHLExperiment extends Experiment {
    Inference inference;
    HashMap<String, Integer> labelCounts;
    HashMap<String, Boolean> labelsFirst;

    public UHLExperiment() {
        String[] datasets = {

                "SLASHDOT-F",
                "emotions",
                "enron",
                "mediamill",
                "medical",
                "scene",
                "tmc2007-500",
                "yeast",
                "IMDB-F",
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

    public void initMeasures(int numOfLabels) {
        this.measures = new ArrayList<Measure>();
        this.measures.add(new HammingLoss());
        this.measures.add(new UncertainHammingLoss());
        this.measures.add(new ZeroOneLossMeasure());
        /*
        this.measures.add(new InstanceBasedFMeasure());
        this.measures.add(new MicroFMeasure(numOfLabels));
        this.measures.add(new MacroFMeasure(numOfLabels));
        */
    }

    /**
     * Shuffles the instances in a data set.
     *
     * @param instances the data set
     * @return the shuffled data set
     * @throws Exception
     */
    private MultiLabelInstances shuffle(MultiLabelInstances instances) throws Exception {
        Randomize rand = new Randomize();
        rand.setInputFormat(instances.getDataSet());
        rand.setRandomSeed(2018);

        Instances data = instances.getDataSet();

        // shuffle data
        for (int i = 0; i < data.numInstances(); i++) {
            rand.input(data.instance(i));
        }

        rand.batchFinished();
        Instances shuffledData = rand.getOutputFormat();
        Instance processed;

        while ((processed = rand.output()) != null) {
            shuffledData.add(processed);
        }

        return new MultiLabelInstances(shuffledData, instances.getLabelsMetaData());
    }

    /**
     * Shuffles and splits training data into train/test splits.
     *
     * @param instances instances to shuffle and split
     * @param splitPerc percentage (between 1 and 100)
     * @return split and shuffled instances as array of list two. index zero contains the
     * train set, index 1 the test set.
     * @throws InvalidDataFormatException
     */
    private ArrayList<MultiLabelInstances> splitAndShuffle(MultiLabelInstances instances, double splitPerc) throws Exception {
        MultiLabelInstances shuffledData = shuffle(instances);
        Instances data = shuffledData.getDataSet();

        RemovePercentage split = new RemovePercentage();
        split.setPercentage(splitPerc);
        split.setInputFormat(data);
        Instances splitTrainData = split.getOutputFormat();
        Instances splitTestData = split.getOutputFormat();

        // train split
        for (int i = 0; i < data.numInstances(); i++) {
            split.input(data.instance(i));
        }

        split.batchFinished();
        Instance processed;

        while ((processed = split.output()) != null) {
            splitTrainData.add(processed);
        }

        // split data: test
        split.setInvertSelection(!split.getInvertSelection());

        // these two have to be set again due to flushing after completion of the previous filter calculations
        split.setPercentage(splitPerc);
        split.setInputFormat(data);

        // the data points have to be added again as well
        for (int i = 0; i < data.numInstances(); i++) {
            split.input(data.instance(i));
        }

        split.batchFinished();

        while ((processed = split.output()) != null) {
            splitTestData.add(processed);
        }

        ArrayList<MultiLabelInstances> out = new ArrayList<MultiLabelInstances>();
        out.add(new MultiLabelInstances(splitTrainData, instances.getLabelsMetaData()));
        out.add(new MultiLabelInstances(splitTestData, instances.getLabelsMetaData()));

        return out;
    }

    private Evaluation singleEvaluation(MultiLabelInstances instances, double splitPercentage) throws Exception {
        ArrayList<MultiLabelInstances> data = splitAndShuffle(instances, splitPercentage);
        MultiLabelInstances trainData = data.get(0);
        MultiLabelInstances testData = data.get(1);

        Evaluator eval = new Evaluator();
        PCC model = new PCC(this.inference);
        model.setBaseClassifier(new Logistic());
        model.build(trainData);

        return eval.evaluate(model, testData, this.measures);
    }

    private MultipleEvaluation crossValidation(MultiLabelInstances instances, int folds) throws Exception {
        MultiLabelInstances data = shuffle(instances);

        Evaluator eval = new Evaluator();
        PCC model = new PCC(this.inference);
        model.setBaseClassifier(new Logistic());

        return eval.crossValidate(model, data, this.measures, folds);
    }

    private void writeCSV(String csv, String fileName) throws Exception {
        File file = new File(fileName);

        if (!file.exists()) {
            file.createNewFile();
        }

        BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(file));
        out.write(csv.getBytes());
        out.close();
    }

    @Override
    public void runExperiment() throws Exception {
        for (String dataset : this.dataSets) {
            System.out.println("Experiment for \"" + dataset + "\":");
            MultiLabelInstances data;
            /*
            data = new MultiLabelInstances("datasets/" + dataset + ".arff",
                    "datasets/" + dataset + ".xml");
            */
            FileInputStream fileStream = null;
            File arffFile = new File("datasets/" + dataset + ".arff");
            fileStream = new FileInputStream(arffFile);

            if ((boolean) this.labelsFirst.get(dataset)) {
                System.out.println("labelsFirst");
            }

            data = new MultiLabelInstances((InputStream) fileStream,
                    (int) this.labelCounts.get(dataset),
                    (boolean) this.labelsFirst.get(dataset));

            System.out.println(data.getLabelsMetaData().getLabelNames());
            System.out.println(Utils.arrayToString(data.getDataSet().get(0).toDoubleArray()));

            /*
            if (data.getNumLabels() > 10) {
                System.out.println("reduced labels to 10");
                data = LabelSpaceReduction.reduceLabelSpace(data, 10);
            }
            */

            this.initMeasures(data.getNumLabels());
            System.out.println("labels:");
            System.out.println(data.getNumLabels());
            System.out.println("instances");
            System.out.println(data.getNumInstances());
            System.out.println("features");
            System.out.println(data.getFeatureAttributes().size());

            if (data.getNumInstances() >= 10000) {
                System.out.println("2/3 train-test split");
                Evaluation res = singleEvaluation(data, 33);
                writeCSV(res.toCSV(), "results/results-" + dataset + ".csv");
                System.out.println(res.toString());
            } else {
                System.out.println("3-fold cross validation");
                MultipleEvaluation res = crossValidation(data, 3);
                res.calculateStatistics();
                writeCSV(res.toCSV(), "results/results-" + dataset + ".csv");
                System.out.println(res.toString());
            }
        }
    }

    public static void main(String[] args) throws Exception {
        // System.setErr(new PrintStream(new File("errors.txt")));
        Experiment experiment = new UHLExperiment();
        experiment.runExperiment();
    }
}