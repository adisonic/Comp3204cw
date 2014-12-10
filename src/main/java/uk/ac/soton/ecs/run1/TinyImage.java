package uk.ac.soton.ecs.run1;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

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
import org.openimaj.knn.FloatNearestNeighbours;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.knn.ObjectNearestNeighboursExact;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.ml.clustering.assignment.soft.FloatKNNAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import uk.ac.soton.ecs.Run;

public class TinyImage implements Run {
	
	private static final int K = 5;
	private static final int SQUARE_SIZE = 16;
	
	private FloatNearestNeighboursExact knn;
	private List<String> classes;
	private List<float[]> featureVectors;
	
	public static void main(String[] args){
		try {
			VFSGroupDataset<FImage> dataset = new VFSGroupDataset<FImage>("/Users/Tom/Desktop/training", ImageUtilities.FIMAGE_READER);
					
			int nTraining = 10;
			int nTesting = 10;

			GroupedRandomSplitter<String, FImage> splits =  new GroupedRandomSplitter<String, FImage>(dataset, nTraining, 0, nTesting);
			GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
			GroupedDataset<String, ListDataset<FImage>, FImage> testing = splits.getTestDataset();
			
			TinyImage ti = new TinyImage();
			ti.train(training);
			
			FImage image = testing.getRandomInstance();
			String result = ti.classify(image);
			System.out.println("Guess = "+result);
			DisplayUtilities.display(image);
			
		} catch (FileSystemException e) {
			e.printStackTrace();
		}
	}


	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
		
		classes = new ArrayList<String>();
		featureVectors = new ArrayList<float[]>();
		
		for(String group : trainingSet.getGroups()){
			for(FImage image : trainingSet.get(group)){
				VectorExtractor ve = new VectorExtractor();
				float[] featureVector = ve.extractFeature(image).values;
				featureVectors.add(featureVector);
				classes.add(group);
			}
		}
		
		float[][] vectors = featureVectors.toArray(new float[featureVectors.size()][]);
		zeroMean(vectors);
		
		knn = new FloatNearestNeighboursExact(vectors);
	}
	
	private void zeroMean(float[][] data){
		
	}

	public String classify(FImage image) {
		VectorExtractor ve = new VectorExtractor();
		float[] featureVector = ve.extractFeature(image).values;
		List<IntFloatPair> neighbours = knn.searchKNN(featureVector, K);
		Map<String,Integer> classCount = new HashMap<String,Integer>();
		for(IntFloatPair result: neighbours){
			String resultClass = classes.get(result.first);
			
			int newCount = 1;
			if(classCount.containsKey(resultClass)){
				newCount += classCount.get(resultClass);
			}
			classCount.put(resultClass, newCount);
		}
		
		List<Map.Entry<String, Integer>> guessList = new ArrayList<Map.Entry<String, Integer>>(classCount.entrySet());
		Collections.sort(guessList, new Comparator<Map.Entry<String, Integer>>(){
			public int compare(Entry<String, Integer> o1, Entry<String, Integer> o2){
				return o2.getValue().compareTo(o1.getValue());
			}
		});
		
		for(Map.Entry<String, Integer> guess : guessList){
			System.out.println(guess.getKey() + " - "+guess.getValue());
		}
		
		String guessedClass = guessList.get(0).getKey();
		
		return guessedClass;
	}
	
	class VectorExtractor implements FeatureExtractor<FloatFV,FImage>{

		public FloatFV extractFeature(FImage image) {
			int size = Math.min(image.width, image.height);
			FImage center = image.extractCenter(size, size);
			FImage small = center.process(new ResizeProcessor(SQUARE_SIZE, SQUARE_SIZE));
			float[] vector = ArrayUtils.reshape(small.pixels);
			//TODO: zero means
			return new FloatFV(vector);
		}
		
	}
}
