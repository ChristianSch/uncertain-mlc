package com.cs_pum.uncertain_mlc.examples;

import com.cs_pum.uncertain_mlc.losses.UncertainHammingLoss;
import com.cs_pum.uncertain_mlc.losses.UncertainLoss;
import com.opencsv.CSVReader;
import mulan.classifier.MultiLabelOutput;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.Measure;
import put.mlc.measures.ZeroOneLossMeasure;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * This experiment implements approximate optimization of the tau parameter of an extended loss, such as the extended
 * hamming loss implementing uncertainty in its evaluation of predictions. This very experiment is implemented such that
 * predictions (in the form of confidences) are read from files (see examples.MakePredictions) and tau is (approximately)
 * optimized for one set of confidences.
 */
public class TauOptimization {
    private List<Measure> measures;


    /**
     * This function tries to find an approximation of the optimal tau for a given loss/score function.
     *
     * @param measure measure to be optimized. note that only one measure can be optimized at once. optimality of
     * tau w.r.t. to a measure does not imply optimality of tau w.r.t. to another measure
     * @param minimize whether to minimize or maximize the given metric (loss functions are to be minimized,
     * score functions to be maxmimized)
     *
     * @return approximately symmetric tau
     */
    double tauGridSearch(List<double[]> confidences, List<double[]> groundTruth, UncertainLoss measure, double omega, Boolean minimize) {
        double noCandidates = 30;
        double start = .0;
        double end = .5;
        double step = (end - start) / noCandidates;
        double optTau = 0;
        double optValue = 1;
        double optUncertainty = 0;

        if (this.measures.size() > 0) {
            for (Measure m : this.measures) {
                m.reset();
            }
        }

        for (int i = 0; i < noCandidates; i++) {
            measure.reset();
            double tau = start + ((i + 1) * step);
            System.out.print("-> tau := ");
            System.out.println(tau);
            measure.setTau(tau);
            measure.setOmega(omega);

            for (int j = 0; j < confidences.size(); j++) {
                MultiLabelOutput mlOutput = new MultiLabelOutput(confidences.get(j), .5);
                MultiLabelOutput gt = new MultiLabelOutput(groundTruth.get(j), .5);
                measure.update(mlOutput, new GroundTruth(gt.getBipartition()));
            }

            System.out.println(measure.toString());
            System.out.print("# uncertainty: ");
            System.out.println(measure.getUncertainty());
            System.out.print("# hamming loss: ");
            /**
             * using "<" allows us to use the *first* optimal value of the uncertain loss. it is indeed thinkable, that
             * multiple equal values occur throughout the process. choosing the first one however, guarantees a bigger
             * distance to 1/2 and thus creates a margin, hypothetically resulting in more abstentions.
             */
            if (measure.getValue() < optValue) {
                optValue = measure.getValue();
                optTau = tau;
                optUncertainty = measure.getUncertainty();
            }
        }

        System.out.println(optUncertainty);
        System.out.println(optValue);

        return optTau;
    }

    public static void main(String[] args) {
        String[] predictionFiles = {
                "results/predictions-emotions.csv",
                "results/predictions-IMDB-F.csv",
                "results/predictions-medical.csv",
                "results/predictions-REUTERS-K500-EX2.csv",
                "results/predictions-SLASHDOT-F.csv",
                "results/predictions-yeast.csv",
                "results/predictions-enron.csv",
                "results/predictions-mediamill.csv",
                "results/predictions-OHSUMED-F.csv",
                "results/predictions-scene.csv",
                "results/predictions-tmc2007-500.csv"
        };

        for (String fileName : predictionFiles) {
            List<double[]> confidences = new ArrayList<double[]>();
            List<double[]> groundTruth = new ArrayList<double[]>();

            System.out.print("# processing: ");
            System.out.println(fileName);

            try {
                FileReader fileReader = new FileReader(fileName);

                CSVReader reader = new CSVReader(fileReader);
                String[] nextLine;
                String[] header = null;
                int predictionCount = 0;
                int groundTruthStart = 0;

                while ((nextLine = reader.readNext()) != null) {
                    if (reader.getLinesRead() == 1) {
                        header = nextLine;

                        for (String h : header) {
                            if (h.startsWith("pred_")) {
                                predictionCount++;
                                groundTruthStart++;
                            }
                        }

                        if (header[groundTruthStart].equals("fold")) {
                            // skip "fold" header
                            groundTruthStart++;
                        }

                    } else {
                        double[] doubleValues = Arrays.stream(Arrays.copyOfRange(nextLine, 0, predictionCount))
                                .mapToDouble(Double::parseDouble)
                                .toArray();
                        confidences.add(doubleValues);

                        double[] doubleValuesGT = Arrays.stream(Arrays.copyOfRange(nextLine, groundTruthStart, header.length))
                                .mapToDouble(Double::parseDouble)
                                .toArray();
                        groundTruth.add(doubleValuesGT);
                    }
                }

            } catch (Exception e) {
                e.printStackTrace();
            }

            TauOptimization tOpt = new TauOptimization();
            List<Measure> measures = new ArrayList<Measure>();
            measures.add(new HammingLoss());
            measures.add(new UncertainHammingLoss(1./3, 1./2));
            measures.add(new ZeroOneLossMeasure());
            tOpt.setMeasures(measures);
            double optTau = tOpt.tauGridSearch(confidences, groundTruth, new UncertainHammingLoss(), .5, true);
            System.out.print(" /!\\ OPTIMAL TAU: ");
            System.out.println(optTau);
        }
    }

    public void setMeasures(List<Measure> measures) {
        this.measures = measures;
    }
}
