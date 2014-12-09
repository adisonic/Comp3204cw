package uk.ac.soton.ecs.run1;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;

import uk.ac.soton.ecs.Run;

public class TinyImage implements Run {

	public void train(GroupedDataset<String, ListDataset<Record<FImage>>, Record<FImage>> trainingSet) {

		
	}

	public String classify(FImage image) {
		return null;
	}

}
