package com.cs_pum.uncertain_mlc.common;

import mulan.data.MultiLabelInstances;
import weka.core.Instances;


public class LabelMetadata {
    public static int[] getLabelCounts(MultiLabelInstances instances, boolean labelsFirst) {
        int[] counts = new int[instances.getNumLabels()];

        Instances data = instances.getDataSet();
        int numLabels = instances.getNumLabels();
        int numInstances = instances.getNumInstances();
        int numFeatures = instances.getFeatureAttributes().size();

        for (int j = 0; j < numLabels; j++) {
            counts[j] = 0;
        }

        for (int i = 0; i < numInstances; i++) {
            for (int j = 0; j < numLabels; j++) {
                if (labelsFirst && (data.get(i).toDoubleArray()[j] >= .5)) {
                    counts[j] += 1;
                } else if (!labelsFirst && (data.get(i).toDoubleArray()[j + numFeatures] >= .5)) {
                    counts[j] += 1;
                }
            }
        }

        return counts;
    }

    public static double[] getLabelFrequencies(MultiLabelInstances instances, boolean labelsFirst) {
        int[] counts = getLabelCounts(instances, labelsFirst);
        double[] out = new double[counts.length];

        for (int i = 0; i < counts.length; i++) {
            out[i] = (1.* counts[i]) / instances.getNumInstances();
        }

        return out;
    }
}