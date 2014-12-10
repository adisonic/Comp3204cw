package uk.ac.soton.ecs.run2;

import org.apache.commons.vfs2.FileSystemException;
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
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;


import de.bwaldvogel.liblinear.SolverType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class Main {

	public static void main(String[] args){

		
	
		try {

//			try {

			VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<FImage>("zip:D:/training.zip", ImageUtilities.FIMAGE_READER);
			
			GroupedRandomSplitter<String, FImage> splits =  new GroupedRandomSplitter<String, FImage>(trainingSet, 4, 0, 4);
			
			
			
			GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
			
			
			GroupedDataset<String, ListDataset<FImage>, FImage> testing = splits.getTestDataset();
			
			for(String group : training.getGroups()){
				
				
			}
			
			
				DenseSIFT dsift = new DenseSIFT(4, 8);
				PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);
				
				
				HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(training, pdsift);
				
				FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdsift, assigner);
				LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
						extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
				ann.train(training);
//
//				//Evaluation
//				ClassificationEvaluator<CMResult<String>, String, Record<FImage>> eval = 
//						new ClassificationEvaluator<CMResult<String>, String, Record<FImage>>(
//								ann, splits.getTestDataset(), new CMAnalyser<Record<FImage>, String>(CMAnalyser.Strategy.SINGLE));
//
//				Map<Record<FImage>, ClassificationResult<String>> guesses = eval.evaluate();
//				CMResult<String> result = eval.analyse(guesses);
//				System.out.println("wassssuuuppp");
//				System.out.println(result.toString());
//			} catch (IOException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
		
			//System.out.println(trainingSet.numInstances());
			//VFSGroupDataset<FImage> sampleTrainingSet = GroupSampler
	/*	
	 *Just a loop to access the images 	
	 * 
	 * 	for (final Entry<String, VFSListDataset<FImage>> entry : trainingSet.entrySet()) {
			DisplayUtilities.display(entry.getKey(), entry.getValue().getRandomInstance());

			//Prints random pic of each person
				} */
			
			
		} catch (//FileSystem
				Exception e) {

			e.printStackTrace();
		}
	}

	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
			Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift)
			{
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

		for (FImage rec : sample) {
			FImage img = rec.getImage();

			pdsift.analyseImage(img);
			allkeys.add(pdsift.getByteKeypoints(0.005f));
		}

		if (allkeys.size() > 10000) 
			allkeys = allkeys.subList(0, 10000);

		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(500); //trying out 500 to start with
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		ByteCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();
			}


	static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
		PyramidDenseSIFT<FImage> pdsift;
		HardAssigner<byte[], float[], IntFloatPair> assigner;

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
					bovw, 2, 2);

			return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
		}
	}

}
