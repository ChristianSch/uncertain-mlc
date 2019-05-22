package com.cs_pum.uncertain_mlc.losses;

import mulan.classifier.MultiLabelOutput;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.measure.Measure;


/**
 * This class implements the uncertain hamming loss. Predictions in the form of confidences or probabilities that the
 * label equals one are considered uncertain, if they are below/above a certain threshold. Uncertain confidences
 * are treated as wrong predictions of a certain weight.
 *
 * The attribute `tau` as set by `setTau(…)` controls
 * the threshold a confidence has to be below or above "1-tau" to be considered an uncertain one.
 * The attribute `omega` as set by `setOmega(…)` controls the weights associated with an uncertain
 * (and hence considered wrong) prediction in the loss calculation.
 *
 * @author Christian Schulze
 * @since  2018-06-25
 */
public class UncertainHammingLoss implements UncertainLoss {
    private double tau = 1./3;
    private double omega = 1.0;
    private double accum = 0;
    private double calls = 0;
    private double uncertainty = 0;
    private double labelSize = 0;

    public UncertainHammingLoss() {}

    public UncertainHammingLoss(double tau) {
        setTau(tau);
    }

    public UncertainHammingLoss(double tau, double omega) {
        setOmega(omega);
        setTau(tau);
    }

    public double getTau() {
        return tau;
    }

    public void setTau(double tau) {
        if (tau <= 0 || tau > .5) {
            throw new IllegalArgumentException("Tau needs to be > 0. and <= 0.5");
        }

        this.tau = tau;
    }

    public double getOmega() {
        return omega;
    }

    public void setOmega(double omega) {
        if (omega <= 0. || omega > 0.5) {
            throw new IllegalArgumentException("Omega needs to be > 0. and <= 0.5");
        }

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
        if (this.calls == 0) {
            new Exception("measure has not been fed with data yet");
        }

        return this.accum / this.calls;
    }

    public double getIdealValue() {
        return 0;
    }

    public double getUncertainty() {
        return this.omega * (this.uncertainty / (this.calls * this.labelSize));
    }

    @Override
    public double getNoUncertain() {
        return this.uncertainty;
    }

    public void update(MultiLabelOutput multiLabelOutput, GroundTruth groundTruth) {
        if (this.labelSize == 0) {
            this.labelSize =  groundTruth.getTrueLabels().length;
        }

        this.accum += this.computeLoss(multiLabelOutput, groundTruth.getTrueLabels());
        this.calls++;
    }

    public Measure makeCopy() {
        UncertainHammingLoss uhl = new UncertainHammingLoss();

        uhl.setOmega(this.omega);
        uhl.setTau(this.tau);
        // FIXME: exact copy or only of parameters?
        uhl.accum = this.accum;
        uhl.calls = this.calls;

        return uhl;
    }

    public void reset() {
        this.calls = 0;
        this.accum = 0;
        this.uncertainty = 0;
    }

    public boolean handlesMissingValues() {
        return false;
    }

    public double computeLoss(MultiLabelOutput prediction, boolean[] groundTruth) {
        double[] probabilities = prediction.getConfidences();
        boolean[] bipartition = prediction.getBipartition();
        double symmetricDifference = 0;
        double u = 0;

        for (int i = 0; i < groundTruth.length; i++) {
            if (probabilities[i] < this.tau || probabilities[i] > (1 - this.tau)) {
                if (bipartition[i] != groundTruth[i]) {
                    symmetricDifference++;
                }
            } else {
                u += 1;
            }
        }

        this.uncertainty += u;

        return (symmetricDifference + (u * this.omega)) / groundTruth.length;
    }

}
