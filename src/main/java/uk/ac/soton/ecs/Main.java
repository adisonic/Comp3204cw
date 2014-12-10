package uk.ac.soton.ecs;

import java.util.Iterator;
import java.util.Map;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import uk.ac.soton.ecs.run1.TinyImage;

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
        if (id.equals("run1")) {
            return new TinyImage();
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

        // Sample all data so we have a GroupedDataset<String, ListDataset<FImage>, FImage>, and not a group with a VFSListDataset.
        GroupedDataset<String, ListDataset<FImage>, FImage> data = GroupSampler.sample(dataset, dataset.size(), false);

        // Number of images in the first group. (since groups are roughly equal)
        Iterator<String> groupsIter = data.getGroups().iterator();
        if (!groupsIter.hasNext()) {
            throw new Exception("The dataset loaded has no classes.");
        }        
        String grp = groupsIter.next();
        int imagesInGroup = data.get(grp).size();

        // Split into test and training datasets.
        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<String, FImage>(data, imagesInGroup/2, 0, imagesInGroup/2);
        GroupedDataset<String, ListDataset<FImage>, FImage> training = splitter.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> test = splitter.getTestDataset();

        System.out.println("Test set size = " + test.size());
        System.out.println("Training set size = " + training.size());

        // training
        instance.train(training);
        
        // evaluating
        int correct = 0;
        int total = 0;
        for (Map.Entry<String,ListDataset<FImage>> e : test.entrySet()) {
            String real = e.getKey();
            ListDataset<FImage> ld = e.getValue();
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
