package uk.ac.soton.ecs.run2;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
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
import uk.ac.soton.ecs.Run;
import uk.ac.soton.ecs.run2.Main.PHOWExtractor;

public class LinearClassifier implements Run{
	
	private LiblinearAnnotator<FImage, String> ann;

	public static void main(String[] args) throws FileSystemException{
		VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<FImage>("zip:D:/training.zip", ImageUtilities.FIMAGE_READER);
		GroupedRandomSplitter<String, FImage> splits =  new GroupedRandomSplitter<String, FImage>(trainingSet, 4, 0, 4);

		
		GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
		
		LinearClassifier hello = new LinearClassifier();
		hello.train(training);
		FImage tempimage = training.getRandomInstance();
		
		DisplayUtilities.display(tempimage);
		System.out.println(hello.classify(tempimage).toString());
	}
	@Override
	public void train(
			GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
		
		DenseSIFT dsift = new DenseSIFT(4, 8);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);
		
		System.out.println("about to assign");
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(trainingSet, pdsift);
		System.out.println("about to extract");
		FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdsift, assigner);

		System.out.println("liblinerannotatotr");
		ann = new LiblinearAnnotator<FImage, String>(
					extractor, Mode.MULTILABEL, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		System.out.println("about to train");
		ann.train(trainingSet);
			
		
	}

	@Override
	public ClassificationResult<String> classify(FImage image) {
		return ann.classify(image);
	}
	
	
	//Hard Assigner
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
			Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift)
			{
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

		for (FImage rec : sample) {
			FImage img = rec.getImage();

			pdsift.analyseImage(img);
			allkeys.add(pdsift.getByteKeypoints(0.005f));
		}

		if (allkeys.size() > 1000) 
			allkeys = allkeys.subList(0, 1000);

		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(500); //trying out 500 to start with
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		ByteCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();
			}
	
	//Extractor class
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
