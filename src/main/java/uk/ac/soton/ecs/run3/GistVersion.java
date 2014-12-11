package uk.ac.soton.ecs.run3;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.global.Gist;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;

import uk.ac.soton.ecs.Main;
import uk.ac.soton.ecs.Run;
import de.bwaldvogel.liblinear.SolverType;

public class GistVersion implements Run {
	
	public static void main(String[] args) throws Exception{
		GistVersion gv = new GistVersion();
		Main.run(gv, "zip:/Users/Tom/Desktop/training.zip");
	}
	
	//Linear annotator
	private LiblinearAnnotator<FImage, String> ann;

	//Train classifier
	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
		//Gist descriptor generator
		Gist<FImage> gist = new Gist<FImage>(256,256);
		
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Uniform);
		//Gist geature extractor
		FeatureExtractor<DoubleFV, FImage> extractor = hkm.createWrappedExtractor(new GistExtractor(gist));
		
		//Classifier for gist features
		ann = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		
		System.out.println("Start training");
		ann.train(trainingSet);
		System.out.println("Train done");
		
		
	}
	

	public ClassificationResult<String> classify(FImage image) {
		System.out.println("Classify");
		return ann.classify(image);
	}
		
	
	//Generates feature vector for image based on gist
	class GistExtractor implements FeatureExtractor<FloatFV, FImage> {
		
		private Gist<FImage> gist;
		
		public GistExtractor(Gist<FImage> gist){
			this.gist = gist;
		}
		
	    public FloatFV extractFeature(FImage object) {
	    	//Extract and return gist feature
	    	gist.analyseImage(object);
	        return gist.getResponse();
	    }
	}


}
