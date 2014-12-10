package uk.ac.soton.ecs;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.Classifier;

public interface Run extends Classifier<String, FImage> {
	
	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet);
	
	public ClassificationResult<String> classify(FImage image);

}
