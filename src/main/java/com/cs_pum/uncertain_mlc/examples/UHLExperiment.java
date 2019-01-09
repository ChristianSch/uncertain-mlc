package com.cs_pum.uncertain_mlc.examples;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.cs_pum.uncertain_mlc.common.LabelMetadata;
import com.cs_pum.uncertain_mlc.common.LabelSpaceReduction;
import com.cs_pum.uncertain_mlc.losses.UncertainHammingLoss;

import com.cs_pum.uncertain_mlc.losses.UncertainLoss;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.Evaluation;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.Measure;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math.stat.descriptive.moment.Mean;
import org.apache.commons.math.stat.descriptive.moment.StandardDeviation;
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

    public void initMeasures(int numOfLabels) {
        this.measures = new ArrayList<Measure>();
        this.measures.add(new HammingLoss());
        this.measures.add(new UncertainHammingLoss(1./3, 1./2));
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

        assert file.exists() || file.createNewFile();

        BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(file));
        out.write(csv.getBytes());
        out.close();
    }

    private String[] prependStringArr(String[] arr, String prefix) {
        for (int i = 0; i < arr.length; i++) {
            arr[i] = prefix + arr[i];
        }

        return arr;
    }

    @Override
    public void runExperiment() throws Exception {
        for (String dataset : this.dataSets) {
            System.out.println("Experiment for \"" + dataset + "\":");
            MultiLabelInstances data;
            int someFolds = 3;

            DecimalFormat formatter = new DecimalFormat("#.########");
            FileInputStream fileStream;
            File arffFile = new File("datasets/" + dataset + ".arff");
            fileStream = new FileInputStream(arffFile);
            boolean labelsFirst = this.labelsFirst.get(dataset);

            data = new MultiLabelInstances(fileStream,
                    this.labelCounts.get(dataset),
                    labelsFirst);

            if (data.getNumLabels() > 10) {
                System.out.println("reduced labels to 10");
                data = LabelSpaceReduction.reduceLabelSpace(data, 10, labelsFirst);
            }

            Instances workingSet = new Instances(data.getDataSet());
            PCC model = new PCC(this.inference);
            model.setBaseClassifier(new Logistic());
            String[] labelNames = new String[data.getLabelsMetaData().getLabelNames().size()];
            data.getLabelsMetaData().getLabelNames().toArray(labelNames);

            this.initMeasures(data.getNumLabels());

            // FIXME: if doing ex post tau optimization, we should have another validation set for that
            StringBuilder out = new StringBuilder(String.join(",",
                    this.prependStringArr(labelNames, "pred_")) + ",fold,"
                    + String.join(",", data.getLabelsMetaData().getLabelNames()) + '\n');

            List<List<double[]>> allConfidences = new ArrayList<>();
            List<List<double[]>> allGroundTruth = new ArrayList<>();

            HashMap<String, List<Double>> results = new HashMap<>();

            for(int i = 0; i < someFolds; ++i) {
                try {
                    int numLabels = data.getNumLabels();
                    int numFeatures = data.getFeatureAttributes().size();

                    Instances train = workingSet.trainCV(someFolds, i);
                    Instances test = workingSet.testCV(someFolds, i);

                    MultiLabelInstances mlTrain = new MultiLabelInstances(train, data.getLabelsMetaData());
                    // MultiLabelInstances mlTest = new MultiLabelInstances(test, data.getLabelsMetaData());

                    List<double[]> foldConfidences = new ArrayList<double[]>();
                    List<double[]> foldGroundTruth = new ArrayList<double[]>();

                    MultiLabelLearner clone = model.makeCopy();
                    clone.build(mlTrain);

                    for (int j = 0; j < test.numInstances(); j++) {
                        Instance testInstance = test.instance(j);
                        double[] inst = testInstance.toDoubleArray();
                        MultiLabelOutput res = clone.makePrediction(testInstance);
                        double[] confidences = res.getConfidences();

                        foldConfidences.add(confidences);

                        assert numLabels > 0;

                        /* everything below is for writing the csv file with the confidences */
                        for (double confidence : confidences) {
                            // predicted labels (probability y_i = 1)
                            out.append(formatter.format(confidence)).append(',');
                        }

                        // #fold
                        out.append((new Integer(i)).toString()).append(',');

                        // ground truth
                        double[] groundTruth = new double[numLabels];
                        for (int k = 0; k < numLabels; k++) {
                            if (labelsFirst) {
                                out.append(inst[k]);
                                groundTruth[k] = inst[k];
                            } else {
                                out.append(inst[k + numFeatures]);
                                groundTruth[k] = inst[k + numFeatures];
                            }

                            if (k < (numLabels - 1)) {
                                out.append(",");
                            }
                        }

                        foldGroundTruth.add(groundTruth);

                        out.append('\n');
                    }

                    allGroundTruth.add(foldGroundTruth);
                    allConfidences.add(foldConfidences);

                    assert foldGroundTruth.size() > 0;

                } catch (Exception var14) {
                    Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE, null, var14);
                }
            }

            /* save confidences of predictions (probabilistic predictions) to csv */
            this.writeCSV(out.toString(), "results/predictions-" + dataset + ".csv");

            // TODO: write result of tau optimization to csv with its losses
            // now the
            TauOptimization tOpt = new TauOptimization();
            tOpt.setMeasures(this.measures);
            List<double[]> confidences = new ArrayList<>();
            List<double[]> groundTruth = new ArrayList<>();

            for (List<double[]> confs : allConfidences) {
                for (int i = 0; i < confs.size(); i++) {
                    confidences.add(confs.get(i));
                }
            }

            for (List<double[]> grounds : allGroundTruth) {
                for (int i = 0; i < grounds.size(); i++) {
                    groundTruth.add(grounds.get(i));
                }
            }

            double optTau = tOpt.tauGridSearch(confidences, groundTruth, new UncertainHammingLoss(), .5,true);
            System.out.print(" /!\\ OPTIMAL TAU: ");
            System.out.println(optTau);

            // FIXME: UHL returning 0 all the time?
            if (this.measures.size() > 0) {
                for (Measure measure : this.measures) {
                    measure.reset();
                    String k = measure.getName();

                    if (measure instanceof UncertainHammingLoss) {
                        ((UncertainHammingLoss) measure).setTau(optTau);
                    }

                    for (int j = 0; j < allConfidences.size(); j++) {
                        List<double[]> foldConfidences = allConfidences.get(j);
                        List<double[]> foldGroundTruth = allGroundTruth.get(j);

                        for (int h = 0; h < foldConfidences.size(); h++) {
                            /* the threshold is only applicable for hamming loss, subset 0/1 loss etc */
                            MultiLabelOutput mlOutput = new MultiLabelOutput(foldConfidences.get(h), .5);
                            MultiLabelOutput gt = new MultiLabelOutput(foldGroundTruth.get(h), .5);

                            measure.update(mlOutput, new GroundTruth(gt.getBipartition()));
                        }

                        if (measure instanceof UncertainLoss) {
                            System.out.print("# uncertainty ratio: ");
                            double ucr = ((UncertainLoss) measure).getUncertainty();
                            System.out.println(ucr);
                        }

                        Double v = new Double(measure.getValue());

                        if (results.containsKey(k)) {
                            results.get(k).add(v);
                        } else {
                            List<Double> r = new ArrayList<Double>();
                            r.add(v);
                            results.put(k, r);
                        }
                    }

                    Double[] d_values = new Double[someFolds];
                    results.get(k).toArray(d_values);
                    double[] values = ArrayUtils.toPrimitive(d_values);
                    Mean m = new Mean();
                    double mean = m.evaluate(values, 0, values.length);
                    StandardDeviation sd = new StandardDeviation();

                    System.out.print(k);
                    System.out.print(": ");
                    System.out.print(mean);
                    System.out.print("+-");
                    System.out.println(sd.evaluate(values, mean));
                }
            } else {
                // evaluation[i] = this.evaluate(clone, mlTest, mlTrain);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        // System.setErr(new PrintStream(new File("errors.txt")));
        Experiment experiment = new UHLExperiment();
        experiment.runExperiment();
    }
}
