package com.cs_pum.uncertain_mlc.losses;

import mulan.evaluation.loss.MultiLabelLossFunction;
import mulan.evaluation.measure.Measure;

public interface UncertainLoss extends MultiLabelLossFunction, Measure {

    void setTao(double tao);
    void setOmega(double omega);

    double getTao();
    double getOmega();

    /**
     * Returns uncertainty component of the loss which is defined as the fraction of uncertain predictions
     * weighted by `omega`.
     *
     * @return uncertainty
     */
    double getUncertainty();

    /**
     * Returns total count of uncertain labels
     * @return
     */
    int getNoUncertain();

}
