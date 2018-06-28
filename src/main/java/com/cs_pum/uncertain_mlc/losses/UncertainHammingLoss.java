package com.cs_pum.uncertain_mlc.losses;

import mulan.classifier.MultiLabelOutput;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.loss.MultiLabelLossFunction;
import mulan.evaluation.measure.Measure;


/**
 * This class implements the uncertain hamming loss. Predictions in the form of confidences or probabilities that the
 * label equals one are considered uncertain, if they are below/above a certain threshold. Uncertain confidences
 * are treated as wrong predictions of a certain weight.
 *
 * The attribute `tao` as set by `setTao(…)` controls
 * the threshold a confidence has to be below or above "1-tao" to be considered an uncertain one.
 * The attribute `omega` as set by `setOmega(…)` controls the weights associated with an uncertain
 * (and hence considered wrong) prediction in the loss calculation.
 *
 * @author Christian Schulze
 * @since  2018-06-25
 */
public class UncertainHammingLoss implements MultiLabelLossFunction, Measure {
    private double tao = 1./3;
    private double omega = 1.0;
    private double accum = 0;
    private double calls = 0;

    public double getTao() {
        return tao;
    }

    public void setTao(double tao) {
        // FIXME: throw exception when > 1
        this.tao = tao;
    }

    public double getOmega() {
        // FIXME: print warning when > 1
        return omega;
    }

    public void setOmega(double omega) {
        this.omega = omega;
    }

    @Override
    public String toString() {
        return this.getName() + ": " + String.format("%.4f", this.getValue());
    }

    public String getName() {
        return "Uncertain Hamming Loss";
    }

    public double getValue() {
        return this.accum / this.calls;
    }

    public double getIdealValue() {
        return 0;
    }

    public void update(MultiLabelOutput multiLabelOutput, GroundTruth groundTruth) {
        this.accum += this.computeLoss(multiLabelOutput, groundTruth.getTrueLabels());
        this.calls++;
    }

    public void update(MultiLabelOutput multiLabelOutput, boolean[] groundTruth) {
        this.accum += this.computeLoss(multiLabelOutput, groundTruth);
        this.calls++;
    }

    public Measure makeCopy() throws Exception {
        UncertainHammingLoss uhl = new UncertainHammingLoss();

        uhl.setOmega(this.omega);
        uhl.setTao(this.tao);
        uhl.accum = this.accum;
        uhl.calls = this.calls;

        return uhl;
    }

    public void reset() {
        this.calls = 0;
        this.accum = 0;
    }

    public boolean handlesMissingValues() {
        return false;
    }

    public double computeLoss(MultiLabelOutput prediction, boolean[] groundTruth) {
        double[] probabilities = prediction.getConfidences();
        boolean[] bipartition = prediction.getBipartition();
        double symmetricDifference = 0;
        int u = 0;

        for (int i = 0; i < groundTruth.length; i++) {
            if (probabilities[i] < tao || probabilities[i] > (1 - tao)) {
                if (bipartition[i] != groundTruth[i]) {
                    symmetricDifference++;
                }
            } else {
                u += 1;
            }
        }

        return (symmetricDifference + u) / groundTruth.length;
    }

}
