package uk.ac.soton.ecs;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;

public interface Run {
	
	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet);
	
	public String classify(FImage image);

}
