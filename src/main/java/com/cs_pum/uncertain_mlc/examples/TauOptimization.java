package com.cs_pum.uncertain_mlc.examples;

import com.cs_pum.uncertain_mlc.losses.UncertainHammingLoss;
import com.cs_pum.uncertain_mlc.losses.UncertainLoss;
import com.opencsv.CSVReader;
import mulan.classifier.MultiLabelOutput;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.measure.HammingLoss;

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
    static double tauGridSearch(List<double[]> confidences, List<double[]> groundTruth, UncertainLoss measure, Boolean minimize) {
        double noCandidates = 30;
        double start = .0;
        double end = .49999999;
        double step = (end - start) / noCandidates;

        HammingLoss hl = new HammingLoss();

        for (int i = 0; i < noCandidates; i++) {
            measure.reset();
            hl.reset();
            double tau = start + ((i + 1) * step);
            System.out.print("-> tau := ");
            System.out.println(tau);
            measure.setTau(tau);
            measure.setOmega(.5);

            for (int j = 0; j < confidences.size(); j++) {
                MultiLabelOutput mlOutput = new MultiLabelOutput(confidences.get(j), .5);
                MultiLabelOutput gt = new MultiLabelOutput(groundTruth.get(j), .5);
                measure.update(mlOutput, new GroundTruth(gt.getBipartition()));
                hl.update(mlOutput, new GroundTruth(gt.getBipartition()));
            }

            System.out.println(measure.toString());
            System.out.print("# uncertainty: ");
            System.out.println(measure.getUncertainty());
            System.out.print("# hamming loss: ");
            System.out.println(hl.toString());
        }

        // FIXME: return tau for which measure is minimised
        return .0;
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

            double optTau = tauGridSearch(confidences, groundTruth, new UncertainHammingLoss(), true);
        }
    }
}
