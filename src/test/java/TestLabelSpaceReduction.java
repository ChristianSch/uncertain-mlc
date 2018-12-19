import com.cs_pum.uncertain_mlc.common.LabelSpaceReduction;
import mulan.data.MultiLabelInstances;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.HashMap;

import static org.junit.Assert.assertEquals;

public class TestLabelSpaceReduction {
    HashMap<String, Integer> labelCounts;
    HashMap<String, Boolean> labelsFirst;
    String[] datasets;

    @Before
    public void setUp() {
         datasets = new String[]{
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
    }

    @Test
    public void testLabelSpaceReduction() throws Exception {
        for (String dataset : this.datasets) {
            MultiLabelInstances data;
            /*
            data = new MultiLabelInstances("datasets/" + dataset + ".arff",
                    "datasets/" + dataset + ".xml");
            */
            FileInputStream fileStream;
            File arffFile = new File("datasets/" + dataset + ".arff");
            fileStream = new FileInputStream(arffFile);
            boolean labelsFirst = this.labelsFirst.get(dataset);

            data = new MultiLabelInstances(fileStream,
                    this.labelCounts.get(dataset),
                    labelsFirst);

            // TODO: get first and last instance, compare features

            if (data.getNumLabels() > 10) {
                data = LabelSpaceReduction.reduceLabelSpace(data, 10, labelsFirst);

                assertEquals(data.getNumLabels(), 10);
                assertEquals(data.getDataSet().numAttributes(), data.getDataSet().get(0).toDoubleArray().length);
            }
        }
    }
}
