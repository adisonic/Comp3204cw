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
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
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
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;
import uk.ac.soton.ecs.Run;
//import uk.ac.soton.ecs.run2.Main.PHOWExtractor;
import uk.ac.soton.ecs.Main;

public class experimentallc implements Run{
	
	private LiblinearAnnotator<FImage, String> ann;

	public static void main(String[] args) throws Exception{
		
		
		VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<FImage>("zip:D:/training.zip", ImageUtilities.FIMAGE_READER);
		GroupedRandomSplitter<String, FImage> splits =  new GroupedRandomSplitter<String, FImage>(trainingSet, 4, 0, 4);
//
				GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
		
		LinearClassifier hello = new LinearClassifier();

		FImage tempimage = training.getRandomInstance();
		PatchExtractor pe = new PatchExtractor();
		pe.extract(tempimage);

	}
	@Override
	public void train(
			GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
	
		
	
		//HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(trainingSet);
	
		FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(assigner);

		System.out.println("liblinerannotatotr");
		ann = new LiblinearAnnotator<FImage, String>(
					extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		System.out.println("about to train");
		ann.train(trainingSet); //not working
			
	
	}

	@Override
	public ClassificationResult<String> classify(FImage image) {
		return ann.classify(image);
	}
	
	

	
	static class PatchExtractor{
		
		private static final float STEP = 40;
		private static final float PATCH_SIZE = 60;
		
		static public List<LocalFeature<SpatialLocation, FloatFV>> extract(FImage image){
			RectangleSampler rect = new RectangleSampler(image, STEP, STEP, PATCH_SIZE, PATCH_SIZE);
			List<LocalFeature<SpatialLocation, FloatFV>> areaList = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();
			for(Rectangle r: rect){
				FImage area = image.extractROI(r);
				
				float[] vector = ArrayUtils.reshape(area.pixels);
				FloatFV featureV = new FloatFV(vector);
				SpatialLocation sl = new SpatialLocation(r.x, r.y);
				LocalFeature<SpatialLocation, FloatFV> lf = new LocalFeatureImpl<SpatialLocation, FloatFV>(sl,featureV);
				areaList.add(lf);
			
			}
			
			
			return areaList;
			
		}
		
		
	}
	
	//Extractor class
	static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
	
		HardAssigner<float[], float[], IntFloatPair> assigner;

		public PHOWExtractor(HardAssigner<float[], float[], IntFloatPair> assigner)
		{
			
			this.assigner = assigner;
		}

		public DoubleFV extractFeature(FImage image) {

			BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);

			BlockSpatialAggregator<float[], SparseFloatFV> spatial = new BlockSpatialAggregator<float[], SparseFloatFV>(
					bovw, 2, 2);
			spatial.a
			return spatial.aggregate(PatchExtractor.extract(image), image.getBounds()).normaliseFV();
		}
	}

}
