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
import weka.filters.unsupervised.instance.Randomize;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Makes and saves probabistic predictions on all the datasets. The data are split into
 * three partitions, where 2/3 are used for training, and 1/3 are used for prediction.
 * All three possible combinations are evaluated, so that each instance is used for
 * a prediction.
 */
public class MakePredictions extends Experiment  {

    Inference inference;
    HashMap<String, Integer> labelCounts;
    HashMap<String, Boolean> labelsFirst;

    public MakePredictions() {
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

    private void writeCSV(String csv, String fileName) throws Exception {
        File file = new File(fileName);

        if (!file.exists()) {
            file.createNewFile();
        }

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
            PCC model = new PCC(this.inference);
            model.setBaseClassifier(new Logistic());
            String[] labelNames = new String[data.getLabelsMetaData().getLabelNames().size()];
            data.getLabelsMetaData().getLabelNames().toArray(labelNames);

            System.out.println("labels:");
            System.out.println(Utils.arrayToString(labelNames));
            System.out.println("label frequencies:");
            System.out.println(Utils.arrayToString(LabelMetadata.getLabelFrequencies(data, labelsFirst)));
            System.out.println("label counts:");
            System.out.println(Utils.arrayToString(LabelMetadata.getLabelCounts(data, labelsFirst)));

            String out = String.join(",",
                    this.prependStringArr(labelNames, "pred_")) + ",fold,"
                    + String.join(",", data.getLabelsMetaData().getLabelNames()) + '\n';

            for(int i = 0; i < someFolds; ++i) {
                try {
                    int numLabels = data.getNumLabels();
                    int numFeatures = data.getFeatureAttributes().size();

                    Instances train = workingSet.trainCV(someFolds, i);
                    Instances test = workingSet.testCV(someFolds, i);

                    MultiLabelInstances mlTrain = new MultiLabelInstances(train, data.getLabelsMetaData());
                    MultiLabelInstances mlTest = new MultiLabelInstances(test, data.getLabelsMetaData());

                    MultiLabelLearner clone = model.makeCopy();
                    clone.build(mlTrain);

                    /*
                    // TODO: port code so that we can use it here
                    if (hasMeasures) {
                        evaluation[i] = this.evaluate(clone, mlTest, measures);
                    } else {
                        evaluation[i] = this.evaluate(clone, mlTest, mlTrain);
                    }
                    */
                    for (int j = 0; j < test.numInstances(); j++) {
                        Instance testInstance = test.instance(j);
                        double[] inst = testInstance.toDoubleArray();
                        MultiLabelOutput res = clone.makePrediction(testInstance);
                        double[] confidences = res.getConfidences();

                        for (double confidence : confidences) {
                            // predicted labels (probability y_i = 1)
                            out += formatter.format(confidence) + ',';
                        }

                        // #fold
                        out += (new Integer(i)).toString() + ',';

                        // ground truth
                        for (int k = 0; k < numLabels; k++) {
                            if (labelsFirst) {
                                out += inst[k];
                            } else if (!labelsFirst) {
                                out += inst[k + numFeatures];
                            }

                            if (k < (numLabels - 1)) {
                                out += ",";
                           }
                        }

                        out += '\n';
                    }

                } catch (Exception var14) {
                    Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE, (String) null, var14);
                }

                /* save confidences of predictions (probabilistic predictions) to csv */
                this.writeCSV(out, "results/predictions-" + dataset + ".csv");

            }
        }
    }

    public static void main(String[] args) throws Exception {
        // System.setErr(new PrintStream(new File("errors.txt")));
        Experiment experiment = new MakePredictions();
        experiment.runExperiment();
    }
}