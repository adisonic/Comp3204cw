package uk.ac.soton.ecs;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;

import java.util.ArrayList;
import java.util.Random;

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
	public String classify(FImage image) {
        return classes.get(rnd.nextInt(classes.size()));
    }
}

