package uk.ac.soton.ecs.run2;

import java.util.ArrayList;
import java.util.List;

import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import uk.ac.soton.ecs.Run;
import de.bwaldvogel.liblinear.SolverType;

/**
 * The linear classifier used for Run#2.
 */
public class LinearClassifier implements Run {

    // Clustering parameters
    public static int CLUSTERS = 500;
    public static int IMAGES_FOR_VOCABULARY = 10;

    // Patch parameters
	public static float STEP = 8;
	public static float PATCH_SIZE = 12;

	private LiblinearAnnotator<FImage, String> ann;

    /**
     * Train a classifier, create a feature extractor using the Quantiser
     * and then train a linear classifier using the feature extractor.
     *
     * For training the Quantiser:
     *
     * - it gets IMAGES_FOR_VOCABULARY images
     *   from each class, samples dense patches of size PATCH_SIZE by PATCH_SIZE
     *   with step STEP.
     * - for each patch, create a feature vector, and then a double[][] with
     *   all the feature vectors.
     * - use k-means on the double[][] to obtain CLUSTERS clusters.
     * - obtain a HardAssigner from k-means and obtain a FeatureExtractor.
     * - train a linear model for the FeatureExtractor.
     *
     * This method prints debug messages on stderr.
     *
     * @param trainingSet The training set to use.
     */
	@Override
	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {

        // build vocabulary using images from all classes.
        GroupedRandomSplitter<String, FImage> rndspl = new GroupedRandomSplitter<String, FImage>(trainingSet, IMAGES_FOR_VOCABULARY, 0, 0);
		HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(rndspl.getTrainingDataset());

        // create FeatureExtractor. 
		FeatureExtractor<DoubleFV, FImage> extractor = new PatchClusterFeatureExtractor(assigner);

        // Create and train a linear classifier.
		System.err.println("Start training...");
		ann = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(trainingSet); 

        System.err.println("Training finished.");
	}

	@Override
	public ClassificationResult<String> classify(FImage image) {
		return ann.classify(image);
	}
	
    /**
     * Build a HardAssigner based on k-means ran on randomly picked patches from images.
     * @param sample The dataset to use for creating the HardAssigner.
     */
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample) {
		List<float[]> allkeys = new ArrayList<float[]>();

        // extract patches    
		for (FImage image : sample) {
			List<LocalFeature<SpatialLocation, FloatFV>> sampleList = extract(image, STEP, PATCH_SIZE);
			System.err.println(sampleList.size());

			for(LocalFeature<SpatialLocation, FloatFV> lf : sampleList){
				allkeys.add(lf.getFeatureVector().values);
			}
		}

        // Instantiate CLUSTERS-Means.     
		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(CLUSTERS);
        float[][] data = allkeys.toArray(new float[][]{});
	
        // Clustering using K-means.    
		System.err.println("Start clustering.");
		FloatCentroidsResult result = km.cluster(data);
		System.err.println("Clustering finished.");

		return result.defaultHardAssigner();
	}

    /**
     * Extract patches regarding to STEP and PATCH_SIZE.
     * @param image The image to extract features from.
     * @param step The step size.
     * @param patch_size The size of the patches.
     */
	public static List<LocalFeature<SpatialLocation, FloatFV>> extract(FImage image, float step, float patch_size){
        List<LocalFeature<SpatialLocation, FloatFV>> areaList = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();

        // Create patch positions
        RectangleSampler rect = new RectangleSampler(image, step, step, patch_size, patch_size);

        // Extract feature from position r.
        for(Rectangle r : rect){
            FImage area = image.extractROI(r);

            float[] vector = ArrayUtils.reshape(area.pixels);
            FloatFV featureV = new FloatFV(vector);
            SpatialLocation sl = new SpatialLocation(r.x, r.y);
            LocalFeature<SpatialLocation, FloatFV> lf = new LocalFeatureImpl<SpatialLocation, FloatFV>(sl,featureV);

            areaList.add(lf);
        }

        return areaList;	
	}

    /**
     * Patch FeatureExtractor based on a HardAssigner.
     */    
	static class PatchClusterFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {
		HardAssigner<float[], float[], IntFloatPair> assigner;

		public PatchClusterFeatureExtractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
			this.assigner = assigner;
		}

        /**
         * Extract features of image, in respect to the HardAssigner.
         * @param image The FImage to use.
         * @return A feature vector.
         */
		public DoubleFV extractFeature(FImage image) {
			BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
			BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<float[], SparseIntFV>(bovw, 2, 2);
			return spatial.aggregate(extract(image, STEP, PATCH_SIZE), image.getBounds()).normaliseFV();
		}
	}

}
