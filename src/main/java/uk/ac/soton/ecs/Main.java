package uk.ac.soton.ecs;

import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.validation.GroupedRandomisedPercentageHoldOut;

import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.util.Set;
import java.util.Map;

public class Main {

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Need at least 2 arguments. Run with: <run id> <path to dataset>");
            return;
        }
        Run instance = getInstance(args[0]);
        if (instance == null) {
            System.err.println("Invalid <run id>. Try: random, run1, run2 or run3.");
            return;
        }

        run(instance, args[1], 0.5);
    }

    /**
     * Create a new Run instance from id.
     * @param InstanceId.
     */
    public static Run getInstance(String id) {
        if (id.equals("random")) {
            return new RandomRun();
        }
        // Add these here:
        // if (id.equals("run1")) {
        //     return new run1.Run1();
        // }
        return null;
    }

    /**
     * Train and evaluate the given Run object.
     * The training set is randomly split into two equal sets, for test and training.
     * @param instance The run object to train and evaluate.
     * @param path Path to the dataset.
     * @throws Exception Any IO exception from reading the datasets.
     */
    public static void run(Run instance, String path, double percentage) throws Exception {
        GroupedDataset<String, VFSListDataset<FImage>, FImage> dataset = new VFSGroupDataset<FImage>(path, ImageUtilities.FIMAGE_READER);
        GroupedDataset<String, ListDataset<FImage>, FImage> data = GroupSampler.sample(dataset, dataset.size(), false);

        System.out.println("data = " + data.size());
        System.out.println("classes = "+ dataset.size());
        int imgs=0;
        Set<Map.Entry<String, ListDataset<FImage>>> s = data.entrySet();
        for (Map.Entry<String, ListDataset<FImage>> i : s) {
            System.out.println("For class " + i.getKey());
            ListDataset<FImage> ld = i.getValue();
            for (FImage img : ld) {
                imgs++;
            }
        }
        System.out.println("has image = "+ imgs);

        int f = 10;
        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<String, FImage>(data, f, 0, f);
        GroupedDataset<String, ListDataset<FImage>, FImage> training = splitter.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> test = splitter.getValidationDataset();

        int testPoints = test.size();
        System.out.println("Test points = " + test.size());
        System.out.println("Training points = " + training.size());

        // training
        instance.train(training);
        
        // evaluating
        int correct = 0;
        int total = 0;
        for (Map.Entry<String,ListDataset<FImage>> e : test.entrySet()) {
            System.out.println("testing..." + e.getKey());
            String real = e.getKey();
            ListDataset<FImage> ld = e.getValue();
            System.out.println("Found images here: " + ld.size());
            for (FImage img : ld) {
                String predicted = instance.classify(img);
                if (predicted.equals(real)) {
                    correct++;
                }
                total++;
            }
        }

        System.out.println("Correctly classified " + correct + " out of " + total + " (" +((float)correct*100f/total)+"%).");
    }
}
