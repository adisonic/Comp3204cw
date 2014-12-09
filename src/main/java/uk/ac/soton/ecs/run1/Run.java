package uk.ac.soton.ecs.run1;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;

public interface Run {
	
	public void train(GroupedDataset<String, ListDataset<Record<FImage>>, Record<FImage>> trainingSet);
	
	public void classify(GroupedDataset<String, ListDataset<Record<FImage>>, Record<FImage>> dataset);

}
