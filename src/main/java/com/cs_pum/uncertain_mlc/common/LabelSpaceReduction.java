package com.cs_pum.uncertain_mlc.common;

import java.util.ArrayList;

import mulan.data.InvalidDataFormatException;
import mulan.data.LabelNodeImpl;
import mulan.data.LabelsMetaDataImpl;
import mulan.data.MultiLabelInstances;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.DenseInstance;


/**
 * This class implements reduction of the number of labels for a data set, as originally reported in
 * Bayes Optimal Multilabel Classification via Probabilistic Classifier Chains. ICML 2010: 279-286 by
 * Krzysztof Dembczyński, Weiwei Cheng, Eyke Hüllermeier.
 *
 * Considering the number of occurrence of a label in a data set, the top k occurring labels are retained, the
 * rest is removed in order to reduce the label space dimensionality due to computational concerns.
 *
 * @author Christian Schulze
 * @since  2018-06-25
 */
public class LabelSpaceReduction {
    /**
     * Reduces the labels of a data set to the top `k` occurring labels.
     *
     * @param instances the data set
     * @param numLabelsToKeep number of labels to retain
     * @param labelsFirst indicates if labels or attributes are at the start of the data section
     *                    (mulan datasets should have "false", meka data sets should have "true")
     * @return data set with reduced number of labels
     * @throws InvalidDataFormatException
     */
    public static MultiLabelInstances reduceLabelSpace(MultiLabelInstances instances, int numLabelsToKeep, boolean labelsFirst) throws InvalidDataFormatException {
        if (instances.getNumLabels() <= numLabelsToKeep) {
            return instances;
        }

        Instances data = instances.getDataSet();
        LabelsMetaDataImpl labelsData = new LabelsMetaDataImpl();
        int numLabels = instances.getNumLabels();
        int numInstances = instances.getNumInstances();
        int numFeatures = instances.getFeatureAttributes().size();
        boolean[] keepLabels = new boolean[numLabels];
        int[] counts = LabelMetadata.getLabelCounts(instances, labelsFirst);

        // start and end of label/feature regions in the data are set dynamically
        // to reduce boilerplate code further down. offsets hence refer to different regions
        // depending on the data type (meka/mulan).
        int featureStart;
        int labelStart;

        if (labelsFirst) {
            featureStart = numLabels;
            labelStart = 0;
        } else {
            featureStart = 0;
            labelStart = numFeatures;
        }

        int bound = 0;

        /*
        get minimal frequency (lower bound) for which there are k
        frequencies greater than the bound
        */
        for (int i = 0; i < numLabels; i++) {
            int c = 0;
            bound = counts[i];

            for (int j = 0; j < numLabels; j++) {
                if (counts[j] >= bound) {
                    c += 1;
                }
            }

            if (c == numLabelsToKeep) {
                break;
            }
        }

        for (int i = 0; i < numLabels; i++) {
            if (counts[i] > bound) {
                keepLabels[i] = true;
            } else {
                keepLabels[i] = false;
            }
        }

        ArrayList<Attribute> attrs = new ArrayList<>();

        for (int i = 0; i < numFeatures + numLabelsToKeep; i++) {
            Attribute attr = data.attribute(i).copy(data.attribute(i).name());

            if (!labelsFirst && i >= numFeatures && keepLabels[i - numFeatures]) {
                labelsData.addRootNode(new LabelNodeImpl(attr.name()));
            } else if (labelsFirst && i < numLabels && keepLabels[i]) {
                labelsData.addRootNode(new LabelNodeImpl(attr.name()));
            }

            attrs.add(attr);
        }

        System.out.println(labelsData.getLabelNames());

        Instances insts = new Instances(data.relationName(), attrs, numInstances);

        for (int i = 0; i < numInstances; i++) {
            Instance inst = data.get(i);
            DenseInstance filteredInstance = new DenseInstance(numFeatures + numLabelsToKeep);

            // copy features
            for (int j = featureStart; j < numFeatures; j++) {
                filteredInstance.setValue(featureStart, inst.value(j));
            }

            int offset = 0;

            // copy filtered labels
            for (int k = labelStart; k < numLabels; k++) {
                if (keepLabels[k]) {
                    filteredInstance.setValue(k - offset, inst.value(k));
                } else {
                    offset++;
                }
            }

            filteredInstance.setDataset(insts);
            insts.add(filteredInstance);
        }

        return new MultiLabelInstances(insts, labelsData);
    }
}