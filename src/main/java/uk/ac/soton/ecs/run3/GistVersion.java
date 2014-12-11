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
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
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
import org.openimaj.ml.annotation.basic.KNNAnnotator;
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
import org.openimaj.util.comparator.DistanceComparator;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;
import uk.ac.soton.ecs.Main;
import uk.ac.soton.ecs.Run;

public class GistVersion implements Run {
	
	public static void main(String[] args) throws Exception{
		
		VFSGroupDataset<FImage> images = new VFSGroupDataset<FImage>("/Users/Tom/Desktop/training/", ImageUtilities.FIMAGE_READER);

		GroupedDataset<String, ListDataset<FImage>, FImage> data = GroupSampler.sample(images, 15, false);

		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(
				data, 50, 0, 50);
		
		GistVersion r3 = new GistVersion();
		r3.train(splits.getTrainingDataset());
		
		ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
				new ClassificationEvaluator<CMResult<String>, String, FImage>(
					r3, splits.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
					
	Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
	CMResult<String> result = eval.analyse(guesses);
	System.out.println(result.getDetailReport());
		
	//	GistVersion gv = new GistVersion();
	//	Main.run(gv, "zip:/Users/Tom/Desktop/training.zip");
	}
	
	LiblinearAnnotator<FImage, String>  ann;
	
	

	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
		
		//Gist descriptor creator
		Gist<FImage> gist = new Gist<FImage>(256, 256);
		
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<DoubleFV, FImage> extractor = hkm.createWrappedExtractor(new GistExtractor(gist));
		
		ann = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC_DUAL, 1.0, 0.00001);	
		
		System.out.println("Start training");
		ann.train(trainingSet);
		System.out.println("Train done");
		
		
	}
	

	public ClassificationResult<String> classify(FImage image) {
		return ann.classify(image);
	}
		
	
	//Extract a feature vector from an image using gist
	class GistExtractor implements FeatureExtractor<FloatFV, FImage> {
		
		private Gist<FImage> gist;
		
		public GistExtractor(Gist<FImage> gist){
			this.gist = gist;
		}
		
	    public FloatFV extractFeature(FImage object) {
	    	gist.analyseImage(object);
	        return gist.getResponse();
	    }
	}


}
