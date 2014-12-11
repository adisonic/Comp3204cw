package uk.ac.soton.ecs.run3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;
import uk.ac.soton.ecs.Main;
import uk.ac.soton.ecs.Run;

public class Run3 implements Run {
	
	public static void main(String[] args) throws Exception{
		Run3 r3 = new Run3();
		Main.run(r3, "zip:/Users/Tom/Desktop/training.zip");
	}
	
	//Bayes classifier
	private NaiveBayesAnnotator<FImage, String> ann;

	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
		
		//Dense sift pyramid
		DenseSIFT dsift = new DenseSIFT(5, 5);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(
				dsift, 6f, 7);
		
		//Assigner assigning sift features to visual word
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(trainingSet, pdsift);
		
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		//Feature extractor based on bag of visual words
		FeatureExtractor<DoubleFV, FImage> extractor = hkm.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));
		
		
		ann = new NaiveBayesAnnotator<FImage, String>(extractor,NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);
		System.out.println("Start training");
		ann.train(trainingSet);
		System.out.println("Train done");
		
		
	}
	

	public ClassificationResult<String> classify(FImage image) {
		System.out.println("Classify");
		return ann.classify(image);
	}
	
	
	//Generate visual words
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift) {
		//List of sift features from training set
		List<LocalFeatureList<ByteDSIFTKeypoint>> siftFeatures = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

		System.out.println("Sift train start");
		//For each image
		for (FImage img : sample) {
			System.out.println("image");
			//Get sift features
			pdsift.analyseImage(img);
			siftFeatures.add(pdsift.getByteKeypoints(0.005f));
		}
		
		System.out.println("Sift train done");

		//Reduce set of sift features for time
		if (siftFeatures.size() > 10000)
			siftFeatures = siftFeatures.subList(0, 10000);

		//Create a kmeans classifier with 600 categories (600 visual words)
		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(600);
		
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(siftFeatures);
		System.out.println("Start cluster");
		//Generate clusters (Visual words) from sift features.
		ByteCentroidsResult result = km.cluster(datasource);
		System.out.println("Cluster done");

		return result.defaultHardAssigner();
	}
	
	
	//Extract bag of visual words feature vector
	class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
	    PyramidDenseSIFT<FImage> pdsift;
	    HardAssigner<byte[], float[], IntFloatPair> assigner;

	    public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
	    {
	        this.pdsift = pdsift;
	        this.assigner = assigner;
	    }

	    public DoubleFV extractFeature(FImage image) {
	    	
	    	//Get sift features of input image
	        pdsift.analyseImage(image);

	        //Gag of visual words histogram representation 
	        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

	        //Bag of visual words for blocks and combine
	        BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
	                bovw, 2, 2);

	        //Return normalised feature vector
	        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
	    }
	}


}
