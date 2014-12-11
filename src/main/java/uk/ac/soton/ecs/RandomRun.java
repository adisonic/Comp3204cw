package uk.ac.soton.ecs;

import java.util.ArrayList;
import java.util.Random;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.image.FImage;

public class RandomRun implements Run {

    private ArrayList<String> classes = new ArrayList<String>();
    private Random rnd = new Random();

    /**
     * Remember the class names.
     */
	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
        classes.addAll(trainingSet.keySet());
    }

    /**
     * Randomly return a class.
     * @param image Completely ignored parameter, required to implement interface.
     */
	public BasicClassificationResult<String> classify(FImage image) {
        String cls = classes.get(rnd.nextInt(classes.size()));
        BasicClassificationResult<String> result = new BasicClassificationResult<String>();
        result.put(cls,1.0);
        return result;
    }
}

