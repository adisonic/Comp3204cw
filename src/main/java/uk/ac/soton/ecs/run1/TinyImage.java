package uk.ac.soton.ecs.run1;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.knn.ObjectNearestNeighboursExact;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.ml.clustering.assignment.soft.FloatKNNAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;

import uk.ac.soton.ecs.Run;

public class TinyImage implements Run {
	
	private static final int K = 5;
	private static final int SQUARE_SIZE = 16;
	
	private KNNAnnotator<FImage,String,FloatFV> classifier;
	
	public static void main(String[] args){
		try {
			VFSGroupDataset<FImage> dataset = new VFSGroupDataset<FImage>("zip:/Users/Tom/Desktop/training.zip", ImageUtilities.FIMAGE_READER);
					
			int nTraining = 1;
			int nTesting = 1;

			GroupedRandomSplitter<String, FImage> splits =  new GroupedRandomSplitter<String, FImage>(dataset, nTraining, 0, nTesting);
			GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
			GroupedDataset<String, ListDataset<FImage>, FImage> testing = splits.getTestDataset();
			
			TinyImage ti = new TinyImage();
			ti.train(training);
			FImage image = testing.getRandomInstance();
			String result = ti.classify(image);
			System.out.println(result);
			
		} catch (FileSystemException e) {
			e.printStackTrace();
		}
	}


	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
		classifier = new KNNAnnotator<FImage,String,FloatFV>(new VectorExtractor(),FloatFVComparison.EUCLIDEAN,K);
		classifier.train(trainingSet);
	}

	public String classify(FImage image) {
		List<ScoredAnnotation<String>> results = classifier.annotate(image);
		for(ScoredAnnotation<String> annotation : results){
			System.out.println(annotation.annotation +" - "+annotation.confidence);
		}
		
		return "hi";
	}
	
	class VectorExtractor implements FeatureExtractor<FloatFV,FImage>{

		public FloatFV extractFeature(FImage image) {
			int size = Math.min(image.width, image.height);
			DisplayUtilities.display(image);
			FImage center = image.extractCenter(size, size);
			DisplayUtilities.display(center);
			FImage small = center.process(new ResizeProcessor(SQUARE_SIZE, SQUARE_SIZE));
			float[] vector = ArrayUtils.reshape(small.pixels);
			DisplayUtilities.display(small);
			//TODO: zero means
			return new FloatFV(vector);
		}
		
	}
}
