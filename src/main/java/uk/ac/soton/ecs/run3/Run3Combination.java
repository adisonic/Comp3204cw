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
	
	private NaiveBayesAnnotator<FImage, String> ann;

	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
		DenseSIFT dsift = new DenseSIFT(5, 5);
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(trainingSet, dsift);
		
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<DoubleFV, FImage> extractor = hkm.createWrappedExtractor(new PHOWExtractor(dsift, assigner));
		
		ann = new NaiveBayesAnnotator<FImage, String>(extractor,NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);
		System.out.println("Start training");
		ann.train(trainingSet);
		System.out.println("Train done");
		
		
	}
	

	public ClassificationResult<String> classify(FImage image) {
		System.out.println("Classify");
		return ann.classify(image);
	}
	
	
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample, DenseSIFT pdsift) {
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
        int prg = 0;
        int total = sample.numInstances();
		System.out.println("Sift train start");
		for (FImage img : sample) {
			pdsift.analyseImage(img);
			allkeys.add(pdsift.getByteKeypoints(0.005f));
            System.err.print("\r " + (prg++) + " / " + total + "    ");
		}
	    System.err.println("\r Sift train done 100%.");

		Collections.shuffle(allkeys);
		System.out.println("Sift train done");

		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(100);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		System.out.println("Start cluster");
		ByteCentroidsResult result = km.cluster(datasource);
		System.out.println("Cluster done");

		return result.defaultHardAssigner();
	}
	
	
	class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
	    DenseSIFT pdsift;
	    HardAssigner<byte[], float[], IntFloatPair> assigner;
	    Gist<FImage> gist = new Gist<FImage>(256, 256);

	    public PHOWExtractor(DenseSIFT pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
	    {
	        this.pdsift = pdsift;
	        this.assigner = assigner;
	    }

	    public DoubleFV extractFeature(FImage object) {
	        FImage image = object.getImage();

	        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

	        BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
	                bovw, 4, 4);
	        
	        DoubleFV fv = spatial.aggregate(pdsift.getByteKeypoints(0.005f), image.getBounds()).normaliseFV();
            gist.analyseImage(object);
            fv = fv.concatenate(gist.getResponse().asDoubleFV());
	        return fv;
	    }
	}


}
