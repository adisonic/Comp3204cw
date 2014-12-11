package uk.ac.soton.ecs;

import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import uk.ac.soton.ecs.run1.TinyImage;
import uk.ac.soton.ecs.run2.LinearClassifier;
import uk.ac.soton.ecs.run3.Run3;

public class Main {

    /**
     * Can be run with 2 or three arguments, depending on expected behaviour.
     * To run for evaluation:
     *      Main <run id> <training test path>
     *
     * To run for submission file generation:
     *      Main <run id> <training test path> <test set path> [quiet]
     *
     *      Progress is written on stderr and Output on stdout if the last parameters is not given.
     */
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Need at least 2 arguments. Run with: <run id> <path to dataset>");
            return;
        }
        Run instance = getInstance(args[0]);
        if (instance == null) {
            System.err.println("Invalid <run id>. Try: random, run1, run2 or run3.");
            return;
        }

        if (args.length == 2) {
            run(instance, args[1]);
        } else if (args.length == 3 || args.length == 4) {
            generate(instance, args[1], args[2], args.length != 4);
        } else {
            System.err.println("Wrong arguments.");
        }
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
        if (id.equals("run2")) {
            return new LinearClassifier();
        }
        if (id.equals("run3")) {
            return new Run3();
        }
        // Add these here:
        // if (id.equals("run1")) {
        //     return new run1.Run1();
        // }
        return null;
    }

    /**
     * Generating the submission prediction files.
     * To run this, run: Main <run-id> <training-path> <test-path> > output.txt
     *
     * It writes progess status on the Standard Error and the output on Standard Output,
     * thus output redirection is your friend.
     *
     * @param instance Same as for run().
     * @param trPath Training set path.
     * @param testPath Test set path.
     * @param noisy If true, it prints progress on stderr.
     */
    public static void generate(Run instance, String trPath, String testPath, boolean noisy) throws Exception {
        GroupedDataset<String, VFSListDataset<FImage>, FImage> dataset = new VFSGroupDataset<FImage>(trPath, ImageUtilities.FIMAGE_READER);

        // Sample all data so we have a GroupedDataset<String, ListDataset<FImage>, FImage>, and not a group with a VFSListDataset.
        GroupedDataset<String, ListDataset<FImage>, FImage> data = GroupSampler.sample(dataset, dataset.size(), false);
        
        // Loading test dataset
        VFSListDataset<FImage> testSet = new VFSListDataset<FImage>(testPath, ImageUtilities.FIMAGE_READER);
        if (testSet == null) {
            System.err.println("Error loading trainig set.");
            return;
        }
        int size = 0;

        // weird NullPointerException when checking size if the path is wrong.
        try {
            size = testSet.size();
        } catch (NullPointerException e) {
            System.err.println("Error. Test path probably wrong. Check " + testPath);
            return;
        }

        if (noisy) System.err.println("Training dataset loaded. Staring training...");
        instance.train(data);

        if (noisy) System.err.println("Training complete. Now predicting... (progress on stderr, output on stout)");
        for (int j=0;j<size;j++) {
            FImage img = testSet.get(j);
            String file = testSet.getID(j);

            ClassificationResult<String> predicted = instance.classify(img);
            String[] classes = predicted.getPredictedClasses().toArray(new String[]{});

            if (noisy) {
                System.out.print(file); System.out.print(" ");
                for (String cls : classes) {
                    System.out.print(cls); System.out.print(" ");
                }
                System.out.println();
                System.err.print("\r " + Math.round((j+1)*100.0/size) + " %  ");
            }
        } 
        if (noisy) System.err.println("\n Done.");
    }

    /**
     * Train and evaluate the given Run object.
     * The training set is randomly split into two equal sets, for test and training.
     * @param instance The run object to train and evaluate.
     * @param path Path to the dataset.
     * @throws Exception Any IO exception from reading the datasets.
     */
    public static void run(Run instance, String path) throws Exception {
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
//        int imagesInGroup = 2;
        // Split into test and training datasets.
        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<String, FImage>(data, imagesInGroup/2, 0, imagesInGroup/2);
        GroupedDataset<String, ListDataset<FImage>, FImage> training = splitter.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> test = splitter.getTestDataset();

        System.out.println("Test set size = " + test.size());
        System.out.println("Training set size = " + training.size());

        // training
        instance.train(training);
        
        // OpenIMAJ evaluation method.
        ClassificationEvaluator<CMResult<String>, String, FImage> evaluator =
            new ClassificationEvaluator<CMResult<String>, String, FImage>(instance, test, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = evaluator.evaluate();
        CMResult<String> result = evaluator.analyse(guesses);

        System.out.println(result.getDetailReport());
    }
}
