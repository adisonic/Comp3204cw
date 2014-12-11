package uk.ac.soton.ecs.run3;

import java.util.Map;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.global.Gist;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;

import uk.ac.soton.ecs.Run;
import de.bwaldvogel.liblinear.SolverType;

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
        System.err.println(result.getDetailReport());
		
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

		//67.1 = L2R_L2LOSS_SVC
		//76.1 = L2R_L2LOSS_SVC_DUAL
		ann = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC_DUAL, 1.0, 0.00001);

		/*
		ann = new KNNAnnotator<FImage, String,DoubleFV>(extractor, new DistanceComparator<DoubleFV>(){
			public double compare(DoubleFV o1, DoubleFV o2) {
				return DoubleFVComparison.SUM_SQUARE.compare(o1, o2);
			}

			@Override
			public boolean isDistance() {
				return true;
			}
			
		},20);
		*/
		
		System.err.println("Start training");
		ann.train(trainingSet);
		System.err.println("Train done");
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
