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
import org.openimaj.image.feature.global.Gist;
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

public class Run3Combination implements Run {

    public static int IMAGES_FOR_VOCABULARY = 4;    
	public static void main(String[] args) throws Exception{
		
		VFSGroupDataset<FImage> images = new VFSGroupDataset<FImage>("/Users/vlad/Projects/uni/cv/groupf/training", ImageUtilities.FIMAGE_READER);

		GroupedDataset<String, ListDataset<FImage>, FImage> data = GroupSampler.sample(images, 15, false);

		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(
				data, 4, 0, 4);
		
		Run3Combination r3 = new Run3Combination();
		r3.train(splits.getTrainingDataset());
		
		ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
				new ClassificationEvaluator<CMResult<String>, String, FImage>(
					r3, splits.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
					
        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result.getDetailReport());
		//Run3 r3 = new Run3();
		//Main.run(r3, "zip:/Users/Tom/Desktop/training.zip");
	}
	
	private LiblinearAnnotator<FImage, String> ann;

	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {

			
		DenseSIFT dsift = new DenseSIFT(5, 5);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(
				dsift, 6f, 2,5);
		
		
        GroupedRandomSplitter<String, FImage> rndspl = new GroupedRandomSplitter<String, FImage>(trainingSet, IMAGES_FOR_VOCABULARY, 0, 0);
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(rndspl.getTrainingDataset(), pdsift);
		
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<DoubleFV, FImage> extractor = hkm.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));
		
		ann = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC_DUAL, 1.0, 0.00001);	
		
		System.out.println("Start training");
		ann.train(trainingSet);
		System.out.println("Train done");
		
		
	}
	

	public ClassificationResult<String> classify(FImage image) {
		System.out.println("Classify");
		return ann.classify(image);
	}
	
	
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift) {
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
        int  i =0;
		System.out.println("Sift train start");
		for (FImage img : sample) {
			pdsift.analyseImage(img);
			allkeys.add(pdsift.getByteKeypoints(0.005f));
			System.err.print("\r " + (i++));
		}
	System.err.println();	
//		Collections.shuffle(allkeys);
		System.err.println("Sift train done");

		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(600);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		System.out.println("Start cluster");
		ByteCentroidsResult result = km.cluster(datasource);
		System.out.println("Cluster done");

		return result.defaultHardAssigner();
	}
	
	
	class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
	    PyramidDenseSIFT<FImage> pdsift;
	    HardAssigner<byte[], float[], IntFloatPair> assigner;
	    Gist<FImage> gist = new Gist<FImage>(256, 256);

	    public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
	    {
	        this.pdsift = pdsift;
	        this.assigner = assigner;
	    }

	    public DoubleFV extractFeature(FImage object) {
	        FImage image = object.getImage();
	        pdsift.analyseImage(image);

	        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

	        BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
	                bovw, 4, 4);
	        
	        DoubleFV fv = spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
	        gist.analyseImage(object);
	        DoubleFV g = gist.getResponse().normaliseFV();

	        return fv.concatenate(g);
	    }
	}


}
