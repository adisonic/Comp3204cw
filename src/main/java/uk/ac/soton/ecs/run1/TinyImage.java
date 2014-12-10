package uk.ac.soton.ecs.run1;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.rdf.serialize.SysOutRDFSerializer;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntDoublePair;
import org.openimaj.util.pair.IntFloatPair;

import uk.ac.soton.ecs.Normalisation;
import uk.ac.soton.ecs.Main;
import uk.ac.soton.ecs.Run;
import uk.ac.soton.ecs.run2.LinearClassifier;

public class TinyImage implements Run {
	
	//Number of nearest neighbours to considers
	private static final int K = 15;
	//Dimension of tiny image
	private static final int SQUARE_SIZE = 16;
	
	private DoubleNearestNeighboursExact knn;

    // Normalisation parameters
    private Normalisation norm;

	//The training feature vectors
	private List<double[]> featureVectors;
	//The classes of the training feature vectors (array indices correspond to featureVector indices)
	private List<String> classes;
	
	public static void main(String[] args) throws Exception{
		TinyImage hello = new TinyImage();
		Main.run(hello, "zip:D:/training.zip");
	}
	
	/**
	 * Train classifier
	 */
	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
		classes = new ArrayList<String>();
		featureVectors = new ArrayList<double[]>();
		
		VectorExtractor ve = new VectorExtractor();
		//For each image in each class
        
		for(String group : trainingSet.getGroups()){
			for(FImage image : trainingSet.get(group)){
				// Extract feature vector
                DoubleFV fvobj = ve.extractFeature(image);
                fvobj.normaliseFV();
				double[] fv = fvobj.values;

				featureVectors.add(fv);
				classes.add(group);
			}
		}
    
		//Array of all feature vectors
		double[][] vectors = featureVectors.toArray(new double[][]{});

		//New knn with training feature vectors
		knn = new DoubleNearestNeighboursExact(vectors);
	}
	
	/**
	 * Return guess of image class
	 */
	public BasicClassificationResult<String> classify(FImage image) {
		
		//Extract feature vector for image
		VectorExtractor ve = new VectorExtractor();
        DoubleFV fv = ve.extractFeature(image);
        fv.normaliseFV();
		double[] featureVector = fv.values;

		//Find k nearest neighbours
		List<IntDoublePair> neighbours = knn.searchKNN(featureVector, K);
		
		//Map classes to the number of neighbours that have that class
		Map<String,Integer> classCount = new HashMap<String,Integer>();
		
		//For all neighbours
		for(IntDoublePair result: neighbours){
			//Get neighbour class
			String resultClass = classes.get(result.first);
			
			int newCount = 1;
			//Retrieve existing count for the class
			if(classCount.containsKey(resultClass)){
				newCount += classCount.get(resultClass);
			}
			
			//Add 1 to class count
			classCount.put(resultClass, newCount);
			
		}
		
		//List of class and their count
		List<Map.Entry<String, Integer>> guessList = new ArrayList<Map.Entry<String, Integer>>(classCount.entrySet());
		
		//Sort list with greatest count first
		Collections.sort(guessList, new Comparator<Map.Entry<String, Integer>>(){
			public int compare(Entry<String, Integer> o1, Entry<String, Integer> o2){
				return o2.getValue().compareTo(o1.getValue());
			}
		});
		
		//Confidence in result
		double confidence = guessList.get(0).getValue().doubleValue() / (double) K;
		
		//Guessed class is first in list
		String guessedClass = guessList.get(0).getKey();
	
        BasicClassificationResult<String> result = new BasicClassificationResult<String>();
        result.put(guessedClass, confidence);

		return result;
	}
	
	/** 
	 * Extract TinyImage feature vector from image
	 */
	class VectorExtractor implements FeatureExtractor<DoubleFV,FImage>{

		public DoubleFV extractFeature(FImage image) {
			//Smallest dimension of image is the biggest the square can be
			int size = Math.min(image.width, image.height);
			//Extract the square from centre
	
			FImage center = image.extractCenter(size, size);

			//Resize image to tiny image
			FImage small = center.process(new ResizeProcessor(SQUARE_SIZE, SQUARE_SIZE));
			
			//2D array to 1D vector
			return new DoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(small.pixels)));
		}
		
	}
}
