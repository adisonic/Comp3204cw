package uk.ac.soton.ecs.run2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.AbstractMultiListDataSource;
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
import uk.ac.soton.ecs.Main;

public class LinearClassifier implements Run {

    // Clustering parameters
    public static int CLUSTERS = 600;
    public static int IMAGES_FOR_VOCABULARY = 3;

    // Patch parameters
	public static float STEP = 8;
	public static float PATCH_SIZE = 16;

	private LiblinearAnnotator<FImage, String> ann;

	public static void main(String[] args) throws Exception{
		LinearClassifier hello = new LinearClassifier();
		Main.run(hello, "zip:/Users/Tom/Desktop/training.zip");
	}

	@Override
	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
        // build vocabulary using images from all classes.
        GroupedRandomSplitter<String, FImage> rndspl = new GroupedRandomSplitter<String, FImage>(trainingSet, IMAGES_FOR_VOCABULARY, 0, 0);
		HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(rndspl.getTrainingDataset());
	
		FeatureExtractor<DoubleFV, FImage> extractor = new PatchClusterFeatureExtractor(assigner);

		System.out.println("liblinerannotatotr");
		ann = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTILABEL, SolverType.L2R_LR, 1.0, 0.00001);
		System.out.println("about to train");
		ann.train(trainingSet); //not working

        System.out.println("Training finished.");
	}

	@Override
	public ClassificationResult<String> classify(FImage image) {
		return ann.classify(image);
	}
	
    /**
     * Build a HardAssigner based on k-means ran on randomly picked patches from images.
     */
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample) {
		List<float[]> allkeys = new ArrayList<float[]>();
	
		for (FImage image : sample) {
			List<LocalFeature<SpatialLocation, FloatFV>> sampleList = extract(image, STEP, PATCH_SIZE);
			System.out.println(sampleList.size());

			for(LocalFeature<SpatialLocation, FloatFV> lf : sampleList){
				allkeys.add(lf.getFeatureVector().values);
			}
		}
	
		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(CLUSTERS);
		System.out.println("cluster");
		
        float[][] data = allkeys.toArray(new float[][]{});
		
		FloatCentroidsResult result = km.cluster(data);
		return result.defaultHardAssigner();
	}

    /**
     * Extract patches regarding to STEP and PATCH_SIZE.
     * @param image The image to extract features from.
     * @param step The step size.
     * @param patch_size The size of the patches.
     */
	public static List<LocalFeature<SpatialLocation, FloatFV>> extract(FImage image, float step, float patch_size){
        RectangleSampler rect = new RectangleSampler(image, step, step, patch_size, patch_size);
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
	
	//Extractor class
	static class PatchClusterFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {
		HardAssigner<float[], float[], IntFloatPair> assigner;

		public PatchClusterFeatureExtractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
			this.assigner = assigner;
		}

		public DoubleFV extractFeature(FImage image) {
			BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
			BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<float[], SparseIntFV>(bovw, 2, 2);
			return spatial.aggregate(extract(image, STEP, PATCH_SIZE), image.getBounds()).normaliseFV();
		}
	}

}
