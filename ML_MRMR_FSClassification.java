package mulan.examples;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.lazy.BRkNN;
import mulan.classifier.lazy.IBLR_ML;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.EnsembleOfSubsetLearners;
import mulan.classifier.meta.HOMER;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.meta.HierarchyBuilder.Method;
import mulan.classifier.meta.RAkELd;
import mulan.classifier.neural.BPMLL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.EvaluateByMLRDT;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;
import mulan.inst.MultiInstances;
import mulan.feasel.driftInfor;
import mulan.feasel.mRMR;
import mulan.feasel.mRMR.SELECT_METHOD;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import mulan.feasel.SimilarityEvaluation;
import mulan.dice.data.Instance;
import mulan.dice.data.SimpleInstances;
import mulan.dice.data.io.ArffReader;
import mulan.dice.tree.builder.TreeBuilder;
import mulan.dice.tree.model.CBRRDTModel;
import mulan.dice.tree.model.CBRRDTModel.Prediction;
import mulan.dice.tree.structure.Node;
/**
 * ML_MRMR_FeatureSelection_DataStreamClassification
 * 
 * 
 * @date Jan. 2017
 * @project MultiLabel
 */
public class ML_MRMR_FSClassification {
	// the number of class values, added on  May 27, 2017
	int classValNum = 2;
	// the number of discretized attribute values, added on  May 22, 2017
	int discretValNum = 2;
	// the threshold used in the drifting detection
	double blta = 0.2;
	// the threshold used in the distance evaluation of label distribution based on hamloss and cosine  
	double gamma = 0.3;
	// the number of selected features
	public int nfea = 20;
	// the ratio of candidate features, added on  May 11, 2017
	public double alph = 0.2;
	// the selection type
	public SELECT_METHOD selectMethod = SELECT_METHOD.MID;
	// discreterization
	public double discretize = 0.5;
	// the size of a data chunk
	int dataBlock = 100;//200000;
	// the directory of file
	public String filePath;
	// training-arff
	public String trainArff;
	// testing-arff
	public String testArff;
	// xml file
	public String xmlFile;
	// the beggining index of labels
	public int startIndex;
	// the ending index of labels
	public int endIndex;
	// the number of models in the ensemble model 
	public int modelSize = 100;
	//store the  feature sets selected for all data chunks --- added on  May 15, 2017
	public HashMap<Integer,Double> featureSetOfChunks = new HashMap<Integer,Double>();
	//added on  May 22, 2017, for cosine computation
	//Map<Integer, Double[]> labelDistVerMap =  null;
	public static void main(String[] args) throws Exception {
		String[] comParms = {"-alph", "0.2", "-blta", "0.2", "-gamma", "0.2", "-dataBlock", "200", "-modelSize", "100"};
		/************Data Transformation*****************/
		/*ML_MRMR_FSClassification mcf = new ML_MRMR_FSClassification();
		String filePath= "F:\\workspace\\MultiLabel\\data\\available\\enron";
		String srcFile = "enron.arff";
		int attrSize = 1054;
		int labelNum = 53;
		mcf.DataTranformForMultiKeySort(filePath, srcFile, attrSize, labelNum);
		System.out.println("DataTranformForMultiKeySort!");
		*/
		/****************Classify by mulan after ML-MRMR-Feature selection**********************/
		/*
		// scene data set
//		String[] options = {"-path","F:/workspace/MultiLabel/data/available/scene","-arff","scene.arff","-xml","scene.xml",
//				"-attrSize","300", "-labelNum","6", "-simElvType", "Jaccard", "-algType", "IBLR_ML", "-bDiscretized", "true"};
		// enron data set
	   String[] options = {"-path","F:/workspace/MultiLabel/data/available/enron","-arff","enron.arff","-xml","enron.xml",
				"-attrSize","1054", "-labelNum","53", "-simElvType", "Jaccard", "-algType", "MLKNN", "-bDiscretized", "false"};
		ML_MRMR_FSClassification mcf = new ML_MRMR_FSClassification();
		System.out.println("ensembleClassification...");
		mcf.ML_MRMR_FS_ClassifyByMulan(options);
		*/
		/*********Classify by MLRDT**************/
		//cmd: -path F:\workspace\MultiLabel\data\available\enron -train enron-train.arff -test enron-test.arff -output RDTTest.txt -attrSize 1053 -labelNum 53
		//cmd: -path F:\workspace\MultiLabel\data\available\scene -train scene-train.arff -test scene-test.arff -output RDTTest.txt -attrSize 300 -labelNum 6
//		ML_MRMR_FSClassification mcf = new ML_MRMR_FSClassification();
//		mcf.MLRDTTest(args);
		//mcf.main1(args);
		/*********Classify by MLRDT after ML-MRMR-Feature selection**************/
//		String[] options = {"-path","F:/workspace/MultiLabel/data/available/scene","-train","scene-labelSorted.arff","-test","scene-labelSorted.arff","-xml","scene.xml",
//				"-attrSize","300", "-labelNum","6", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "true", "-bHoldOutTest", "true", "-bAvgVoting", "true", "-alph", "0.2"};
//		String[] options = {"-path","F:/workspace/MultiLabel/data/available/yeast","-train","yeast.arff","-test", "null", "-xml","yeast.xml",
//				"-attrSize","117", "-labelNum","14", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "true", "-bAvgVoting", "true", "-alph", "0.2"};
//		String[] options = {"-path","F:/workspace/MultiLabel/data/available/emotions","-train","emotions.arff","-test", "null","-xml","emotions.xml",
//				"-attrSize","79", "-labelNum","6", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "true", "-bAvgVoting", "true", "-alph", "0.2"};
//		String[] options = {"-path","F:/workspace/MultiLabel/data/available/mediamill","-train","mediamill.arff","-test", "null","-xml","mediamill.xml",
//				"-attrSize","221", "-labelNum","101", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "true", "-bAvgVoting", "true", "-alph", "0.2"};
//		String[] options = {"-path","F:/workspace/MultiLabel/data/available/rcv1subset1","-train","rcv1subset1.arff","-test", "null","-xml","rcv1subset1.xml",
//				"-attrSize","47337", "-labelNum","101", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "true", "-bAvgVoting", "true", "-alph", "0.2"};
//		String[] options = {"-path","F:/workspace/MultiLabel/data/available/rcv1All","-train","rcv1All.arff","-test", "null","-xml","rcv1All.xml",
//				"-attrSize","47337", "-labelNum","101", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "true", "-bAvgVoting", "true", "-alph", "0.2"};
/*		// cannot handle data sets of "cbmi09-bow" and "cbmi09-mpeg", require normalization.
		String[] options = {"-path","F:/workspace/MultiLabel/data/available/cbmi09-bow","-train","cbmi09-bow.arff","-test", "null","-xml","cbmi09-bow.xml",
				"-attrSize","201", "-labelNum","101", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "true", "-bAvgVoting", "true", "-alph", "0.2"};
*/
	//	String[] options = {"-path","F:/workspace/MultiLabel/data/available/enron","-train","enron-sorted.arff","-test", "enron-sorted.arff","-xml","enron.xml",
	//	"-attrSize","1054", "-labelNum","53", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
	//	String[] options = {"-path","F:/workspace/MultiLabel/data/available/enron","-train","enron-train.arff","-test", "enron-test.arff","-xml","enron.xml",
	//			"-attrSize","1054", "-labelNum","53", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
	//	String[] options = {"-path","F:/workspace/MultiLabel/data/available/Corel5k","-train","Corel5k-train.arff","-test", "Corel5k-test.arff","-xml","Corel5k.xml",
	//			"-attrSize","873", "-labelNum","374", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
		
		String[] options1 = {"-path","H:/workspace/MultiLabel/data/available/Corel16k010","-train","Corel16k010-train.arff-sort.arff","-test", "Corel16k010-test.arff","-xml","Corel16k010.xml",
			"-attrSize","644", "-labelNum","144", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
		String[] options2 = {"-path","H:/workspace/MultiLabel/data/available/tmc2007-500","-train","tmc2007-500-train.arff-sort.arff","-test", "tmc2007-500-test.arff","-xml","tmc2007-500.xml",
		"-attrSize","522", "-labelNum","22", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
		
		// use datachunk =200, alph 0.2, code has errors, wait to handle...
//		String[] options3 = {"-path","H:/workspace/MultiLabel/data/available/eurlex-directory-codes","-train","eurlex-dc-leaves-fold1-train.arff-sort.arff","-test", "eurlex-dc-leaves-fold1-test.arff","-xml","eurlex-dc-leaves.xml",
//				"-attrSize","5412", "-labelNum","412", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "true", "-bAvgVoting", "true", "-alph", "0.2"};
        // now use the discretized data
		String[] options3 = {"-path","H:/workspace/MultiLabel/data/available/eurlex-directory-codes","-train","eurlex-dc-leaves-fold1-train.arff-sort.arff.Discretized.arff","-test", "eurlex-dc-leaves-fold1-test.arff","-xml","eurlex-dc-leaves.xml",
				"-attrSize","5412", "-labelNum","412", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
		
		String[] options4 = {"-path","H:/workspace/MultiLabel/data/available/IMDB-ECC-F","-train","IMDB-ECC-F-4-5-train.transform.arff-sort.arff","-test", "IMDB-ECC-F-1-5-test.transform.arff","-xml","IMDB-ECC-F.xml",
				"-attrSize","1029", "-labelNum","28", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
		String[] options5 = {"-path","H:/workspace/MultiLabel/data/available/nuswide-bow","-train","nus-wide-full_BoW_l2-removeId-train.arff-sort-Discretized.arff","-test", "nus-wide-full_BoW_l2-removeId-test.arff","-xml","nus-wide-full_BoW_l2.xml",
				"-attrSize","581", "-labelNum","81", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
	//	String[] options = {"-path","F:/workspace/MultiLabel/data/available/delicious","-train","delicious-train.arff","-test", "delicious-test.arff","-xml","delicious.xml",
	//			"-attrSize","1383", "-labelNum","983", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
//		String[] options = {"-path","F:/workspace/MultiLabel/data/available/genbase","-train","genbase-miss-first-attr.arff","-test", "null","-xml","genbase.xml",
//				"-attrSize","1212", "-labelNum","27", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
//		String[] options = {"-path","F:/workspace/MultiLabel/data/available/tmc2007-500","-train","tmc2007-500-train.arff","-test", "tmc2007-500-test.arff","-xml","tmc2007-500.xml",
//				"-attrSize","522", "-labelNum","22", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
	   String[] options6 = {"-path","H:/workspace/MultiLabel/data/available/bookmarks","-train","bookmarks-4-5-train.arff-sort.arff","-test", "bookmarks-1-5-test.arff","-xml","bookmarks.xml",
				"-attrSize","2358", "-labelNum","208", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
		String[] options7 = {"-path","H:/workspace/MultiLabel/data/available/mediamill","-train","mediamill-train.arff-sort-Discretized.arff","-test", "mediamill-test-Discretized.arff","-xml","mediamill.xml",
		"-attrSize","221", "-labelNum","101", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true", "-alph", "0.2"};
		ML_MRMR_FSClassification mcf = new ML_MRMR_FSClassification();
	
		//featureSetOfChunks = new HashMap<Integer,>
	//	mcf.ML_MRMR_FS_ClassifyByMLRDT(options2);
//		mcf.ML_MRMR_FS_ClassifyByMLRDT(options1);
//
		mcf.InitComParms(comParms);
		mcf.ML_MRMR_FS_ClassifyByMLRDT(options6);
//		mcf.ML_MRMR_FS_ClassifyByMLRDT(options4);
//		mcf.ML_MRMR_FS_ClassifyByMLRDT(options3);
//		mcf.ML_MRMR_FS_ClassifyByMLRDT(options5);
//		mcf.ML_MRMR_FS_ClassifyByMLRDT(options7);
		
		/*************Sort training data by label sequences*******************/
		//mcf.MultiLabelSort(args);
		/*******************Statistics on Label distributions***************************/
//		String dir = "H:/workspace/MultiLabel/data/available/mediamill";
//		String scrFile = "mediamill-train.arff-labelSorting";
//		int chunkSize = 200;
//		mcf.DoLabelStatistics(dir, scrFile, chunkSize);
	}
	/**************Initial Common Parameters
	 * @throws Exception 
	 * @throws  **********************/
	public void InitComParms(String[] comParms) throws Exception {
		alph = Double.valueOf(Utils.getOption("alph", comParms)); 
		blta = Double.valueOf(Utils.getOption("blta", comParms));
		gamma = Double.valueOf(Utils.getOption("gamma", comParms));
		modelSize = Integer.valueOf(Utils.getOption("modelSize", comParms));
		dataBlock = Integer.valueOf(Utils.getOption("dataBlock", comParms));
	}
	/**************创建目录**********************/
	public boolean createDir(String destDirName) {
		File dir = new File(destDirName);
		if (dir.exists()) {// 判断目录是否存在
			System.out.println("创建目录失败，目标目录已存在！");
			return false;
		}
		if (!destDirName.endsWith(File.separator)) {// 结尾是否以"/"结束
			destDirName = destDirName + File.separator;
		}
		if (dir.mkdirs()) {// 创建目标目录
			System.out.println("创建目录成功！" + destDirName);
			return true;
		} else {
			System.out.println("创建目录失败！");
			return false;
		}
	}
	/*****
	 *nonZeroAttrMap 
		store the non-zero values of attributes, which will be used in the drifting detection
	*****/
	/*************Do statistics for Non-zero-values of attributes**************/
    private  Map<Integer,Integer> nonZeroAttrValueStats(Map<Integer,Integer> nonZeroAttrMap, weka.core.Instance tmpInst, int startIndex){
    	//for ( int i = 0; i < )
    	//double[] values = tmpInst.toDoubleArray();
    	int attrIndex = 0;
    	int tmpCount = 0;
    	for ( int j = 0; j < tmpInst.numValues(); j++ ){
    		attrIndex = tmpInst.attributeSparse(j).index();
    		if ( attrIndex < startIndex ){ // not include the class index
	    		if ( nonZeroAttrMap.containsKey(attrIndex) ){
	    			tmpCount = nonZeroAttrMap.get(attrIndex);
	    			tmpCount++;
					nonZeroAttrMap.remove(attrIndex);
					nonZeroAttrMap.put(attrIndex, tmpCount);
				}
				else{
					nonZeroAttrMap.put(attrIndex, 1);
				}
    		}
//    		else
//    			System.out.println(attrIndex);
    	
    	}
    	
    	return nonZeroAttrMap;
    }
    /**
	 * Do statistics for label distributions  used in the concept drifting vertically
	 * 
	 * @param chunkInsts 
				 the given instances
	 * @return the distribution of class labels
	 * @throws Exception
	 */
    private Map<Integer, Double[]> DoLabelDistrStatisticsByVertical(Instances chunkInsts){
    	double[] tmpValueArr = null;
		Map<Integer, Double[]> labelDistMap = new HashMap<Integer, Double[]>();
		
		for ( int k = startIndex; k < endIndex+1; k++ ){//rang of label index
			Double[] aLabelDist = new Double [dataBlock];// store the number of classes
			for ( int j = 0; j < chunkInsts.size(); j++ ){
				tmpValueArr = chunkInsts.get(j).toDoubleArray();
				aLabelDist[j] = (double)tmpValueArr[k];
			}
//			for ( int j = chunkInsts.size(); j< dataBlock; j++ )// deleted on  May 22, 2017
//				aLabelDist[j] = 0.0;
			
			Arrays.sort(aLabelDist);// in an ascending sort
			if ( aLabelDist.length == 1 && aLabelDist[0] < 0.0000000000000000000001)//if the current label has no 1 values, do not count 
				continue;
			labelDistMap.put(k-startIndex, aLabelDist);
		}
		return labelDistMap;
    }
    /**
   	 * Do statistics of label distributions for cosine computation used in the concept drifting vertically, modified on  May 27, 2017
   	 * 
   	 * @param labelDistr 
   				 the given label distribution of instances
   	 * @param maxAttrVCount 
   				 the maximum number of attribute values
   	* @param chunkSize 
   				 the size of data chunk
   	 * @return the distribution of class labels 
   	 * @throws Exception
   	 */
       private void DoLabelDistrStatisticsForCosine(Map<Integer, Double[]>  labelDistMap, ArrayList<HashMap<Integer, Integer>> labelDistr,int maxAttrVCount, int chunkSize){
       	double[] tmpValueArr = null;
   		//Map<Integer, Double[]> labelDistMap = new HashMap<Integer, Double[]>();
   		HashMap<Integer, Integer> aLabelDistr = null;
   		int index = 0;
   		boolean bAllZero = false;
   		for ( int i = 0; i < labelDistr.size(); i++ ){
   			aLabelDistr = labelDistr.get(i);
   			Double [] labelDistDoubleArr = new Double [maxAttrVCount];
   			bAllZero = false;
	   		 Iterator iterator = aLabelDistr.keySet().iterator();                
	         while (iterator.hasNext()) {    
	          Object key = iterator.next();    
	          index = Integer.valueOf(key.toString());
	          labelDistDoubleArr[index] = (double)aLabelDistr.get(key);
	         // System.out.println("||||||||labelDistDoubleArr[index] = "+index+":"+labelDistDoubleArr[index]);
	          if ( index == 0 && Math.abs(labelDistDoubleArr[index]-chunkSize) < 0.000000001 ){
	        	  bAllZero = true;
	        	  break;
	          }
	         // System.out.println("index:"+index+"-value:"+aLabelDistr.get(key));    
	         } 
	         if ( !bAllZero )
	        	 labelDistMap.put((Integer)i, labelDistDoubleArr);
   		}
   		
   		System.out.println("||||||||labelDistVerMap = "+labelDistMap.size());
   		//return labelDistMap;
       }
    /**
  	 * Do statistics for label distributions used in the concept drifting Horizontally
  	 * added on  May 04, 2017, modified on  May 22, 2017
  	 * @param chunkInsts 
  				 the given instances
  	 * @return the distribution of class labels for each instance
  	 * @throws Exception
  	 */
      private ArrayList<int[]> DoLabelDistrStatisticsByHorizontal(Instances chunkInsts){
      	double[] tmpValueArr = null;
      	ArrayList<int[]> labelDistHorMap = new ArrayList<int[]>();
  		
  		for ( int j = 0; j < chunkInsts.size(); j++ ){//for each instance
  			tmpValueArr = chunkInsts.get(j).toDoubleArray();
  			int[] aLabelDist = new int [endIndex-startIndex+1];//[tmpValueArr.length];// store the number of classes for each instance
  			for ( int k = startIndex; k < endIndex+1; k++ ){
  				aLabelDist[k-startIndex] = (int)tmpValueArr[k];// modified 
  			}
  			labelDistHorMap.add(aLabelDist);
  		}
  		return labelDistHorMap;
      }
     
    /**
  	 * Do statistics for label distributions used in the concept drifting
  	 * 
  	 * @param chunkInsts 
  				 the given instances
  	 * @return the distribution of class labels
  	 * @throws Exception
  	 */
      private ArrayList<int[]> DoLabelDistrStatisticsOverridden(Instances chunkInsts){
      	double[] tmpValueArr = null;
  		ArrayList<int[]> labelDistrList = new ArrayList<int[]>();
  		
  		for ( int j = 0; j < chunkInsts.size(); j++ ){
			tmpValueArr = chunkInsts.get(j).toDoubleArray();
			int[] aLabelDist = new int [endIndex-startIndex+1];
			for ( int k = startIndex; k < endIndex+1; k++ ){//rang of label index
				aLabelDist[k] = (int)tmpValueArr[k];
			}
			labelDistrList.add(aLabelDist);
		}
  		return labelDistrList;
      }
      /**************Statistics of true label distributions******************/
      public void GroundtruthsOfLabelDistrStatistics(){
    	  
      }
     /******Sort training data by label sequences******/
      //-path H:\workspace\MultiLabel\data\available\bookmarks -arff bookmarks-4-5-train.arff -xml bookmarks.xml -attrSize 2150 -labelNum 208 -sparse true
      //-path H:\workspace\MultiLabel\data\available\mediamill -arff mediamill-train.arff -xml mediamill.xml -attrSize 120 -labelNum 101 -sparse false
      //-path H:\workspace\MultiLabel\data\available\Corel16k010 -arff Corel16k010-train.arff -xml Corel16k010.xml -attrSize 500 -labelNum 144 -sparse true 
      //-path H:\workspace\MultiLabel\data\available\IMDB-ECC-F -arff IMDB-ECC-F-4-5-train.transform.arff -xml IMDB-ECC-F.xml -attrSize 1001 -labelNum 28 -sparse true
      //-path H:\workspace\MultiLabel\data\available\eurlex-directory-codes -arff eurlex-dc-leaves-fold1-train.arff -xml eurlex-dc-leaves.xml -attrSize 5000 -labelNum 412 -sparse true  
      //-path H:\workspace\MultiLabel\data\available\tmc2007-500 -arff tmc2007-500-train.arff -xml tmc2007-500.xml -attrSize 500 -labelNum 22 -sparse true
      //-path H:\workspace\MultiLabel\data\available\nuswide-bow -arff nus-wide-full_BoW_l2-removeId-train.arff -xml nus-wide-full_BoW_l2.xml -attrSize 500 -labelNum 81 -sparse false

    public void MultiLabelSort(String[] options) throws Exception{
    	// the file directory
    	filePath = Utils.getOption("path", options);
    	// the source file
    	trainArff = Utils.getOption("arff", options);
    	// the file of class's information
    	xmlFile = Utils.getOption("xml", options);
    	// the number of attribute dimensions
    	int attrSize = Integer.valueOf(Utils.getOption("attrSize", options));
    	// the number of class labels
    	int labelNum = Integer.valueOf(Utils.getOption("labelNum", options));
    	boolean bSparse = Boolean.valueOf(Utils.getOption("sparse", options));
    	 MultiLabelInstances trainInsts = new MultiLabelInstances(filePath+"/"+trainArff, filePath+"/"+xmlFile);
		int insCount = trainInsts.getNumInstances();
		Instances instsTrain = trainInsts.getDataSet();
		//Set<Attribute> labelSet = trainInsts.getLabelAttributes();
		int[] labelArr = trainInsts.getLabelIndices();
		int[] feaArr = trainInsts.getFeatureIndices();
		
		ArrayList<AttrLabelSort> labelSeqList = new ArrayList<AttrLabelSort>();
		//
		int tmpLabel = 0;
		String aFeaSeq = "";
		weka.core.Instance inst = null;
		String aLableSeq = "";
		double tmpFeaVal = 0;
		//int feaSize = attrSize - labelNum;
		   for(int i = 0; i < insCount; i++){
	        	inst = instsTrain.get(i);
	        	aLableSeq = "";
	        	//System.out.println(labelArr.length);
				for (int j = 0; j < labelArr.length; j++){
					tmpLabel = (int)(inst.value(labelArr[j]));
					if ( bSparse ){
						//System.out.println(j+"-labelArr[j] = "+labelArr[j]);
						//if ( tmpLabel > 0 )
							aLableSeq += ","+labelArr[j]+" "+tmpLabel;//inst.valueSparse(labelArr[j]);
					}
					else{
						//tmpLabel = (int)inst.value(labelArr[j]);
						tmpLabel = (int)inst.value(labelArr[j]);
						aLableSeq += ","+tmpLabel;
					}
				}
				aFeaSeq = "";
				for ( int k = 0; k < feaArr.length; k++ ){
					tmpFeaVal = inst.value(feaArr[k]);
					if ( bSparse ){
						if ( tmpFeaVal > 0.00000000001 )
							aFeaSeq += ","+feaArr[k]+" "+(int)tmpFeaVal;//inst.valueSparse(feaArr[k]);// use (int) for bookmark, corel16k101,IMDB-ECC-F, tmc, for others please delete (int) 
					}
					else
						aFeaSeq += ","+tmpFeaVal;//inst.value(feaArr[k]);
				}
				AttrLabelSort aPair = new AttrLabelSort(aFeaSeq,aLableSeq);
				labelSeqList.add(aPair);
		   }
		   //Collections.sort(labelSeqList);
		   Collections.sort(labelSeqList, new LabelSeqComparator());

		   BufferedWriter bwFS = new BufferedWriter(new FileWriter(filePath+"/"+trainArff+"-sort.arff"));
		   BufferedWriter bwLabel = new BufferedWriter(new FileWriter(filePath+"/"+trainArff+"-labelSorting"));
		   AttrLabelSort aPair = null;
		   String finalLabelSeq = "";
		   String completeLabelSeq = "";
		   String[] parArr = null;
		   String[] tmParArr = null;
		   for ( int i = 0; i < labelSeqList.size(); i++ ){
			   aPair = labelSeqList.get(i);
			   if ( bSparse ){
				   finalLabelSeq = aPair.getLabelValue();
				 //  System.out.println(finalLabelSeq);
				   parArr = finalLabelSeq.split(",");
				  
				   finalLabelSeq = "";
				   completeLabelSeq = "";
				   for (int j = 1; j < parArr.length; j++){
					   tmParArr = parArr[j].split(" ");
					   tmpLabel = Integer.valueOf(tmParArr[1]);
					   completeLabelSeq += ","+tmpLabel;
					   if ( tmpLabel > 0  ){
						   finalLabelSeq += ","+tmParArr[0]+" "+tmpLabel;
					   }
				   }
				   bwFS.write("{"+aPair.getFeaValue().replaceFirst(",", "")+finalLabelSeq+"}\n");
			   }
			   else{
				   bwFS.write(aPair.getFeaValue().replaceFirst(",", "")+aPair.getLabelValue()+"\n");
				   completeLabelSeq = aPair.getLabelValue();
			   }
			   bwLabel.write(completeLabelSeq+"\n");
			   if ( i % 1000 == 0 ){
				   bwFS.flush();
				   System.out.println(i+" instances are handled.");
			   }
		   }
		   bwFS.close();
		   bwLabel.close();
    }
    /**
	 * Do statistics of label distribution given data
	 * 
	 * @param filePath
	 * 			the path of the source data
	 * @param srcFile
	 * 			 the file of the original data 
	 * @param chunkSize
	 * 			 the size of data chunk 
	 * @return 
	 * 
     * @throws Exception 
	 */
	 private void DoLabelStatistics(String filePath, String srcFile, int chunkSize) throws Exception{
		 FileWriter writer = new FileWriter(filePath+"\\"+srcFile+".labelStats");
		 BufferedReader reader = null;
		 String[] parArr = null;
		 String aKey = "";
		 char[] splitSymbol = {','};
		// ArrayList<HashMap<String, Integer>> arrList = new ArrayList<HashMap<String, Integer>>();
		 HashMap<String, Integer> hashMap = new HashMap<String, Integer>();
		 HashMap<String, Integer> globalHashMap = new HashMap<String, Integer>();
	        try {
	            reader = new BufferedReader(new FileReader(filePath+"\\"+srcFile));
	            String tempString = null;
	            int line = 1;
	            // 一次读入一行，直到读入null为文件结束
	            while ((tempString = reader.readLine()) != null) {
	               // System.out.println("line " + line + ": " + tempString);
	                line++;
//	                if ( tempString.contains("1"))
//	                	System.out.println(tempString);
	                aKey = "";
	                parArr = tempString.replace(",", ", ").split(", ");
	                for ( int i = 0; i < parArr.length; i++ ){
	                	/*if ( parArr[i] == "" || parArr[i] == "0")
	                		continue;*/
	                	if ( parArr[i].contains("1") ){
	                		aKey += i+"-";
	                	}
	                }
	                if ( aKey == "" )
	                	aKey = "AllZero\t";
	                if ( !hashMap.containsKey(aKey) )
	                	hashMap.put(aKey, 1);
	                else{
	                	Integer value = hashMap.get(aKey);
	                	//hashMap[tempString] = value++;
	                	//hashMap.remove(aKey);
	                	hashMap.put(aKey,++value);
	                }
	                
	                if ( !globalHashMap.containsKey(aKey) )
	                	globalHashMap.put(aKey, 1);
	                else{
	                	Integer value = globalHashMap.get(aKey);
	                	//hashMap[tempString] = value++;
	                	//globalHashMap.remove(aKey);
	                	globalHashMap.put(aKey,++value);
	                }
	                
	                
	                if ( line % chunkSize == 0 ){
	                	 Iterator iterator = hashMap.keySet().iterator();                
	                     while (iterator.hasNext()) {    
	                    	 Object key = iterator.next();    
	                    	 System.out.println("map.get(key) is :"+hashMap.get(key));  
	                    	 writer.write("\t"+key.toString()+"\t"+hashMap.get(key)+"\n");
	                    }
	                    writer.write("\n");
	                	hashMap = new HashMap<String, Integer>();
	                }
	            }
	            reader.close();
	        } catch (IOException e) {
	            e.printStackTrace();
	        } finally {
	            if (reader != null) {
	                try {
	                    reader.close();
	                } catch (IOException e1) {
	                }
	            }
	        }
	        Iterator iterator = globalHashMap.keySet().iterator();                
            while (iterator.hasNext()) {    
           	 Object key = iterator.next();    
           	 System.out.println("map.get(key) is :"+globalHashMap.get(key));  
           	 writer.write("\t"+key.toString()+"\t"+globalHashMap.get(key));
           }
           writer.write("\n");
           
		 reader.close();
		 writer.close();
		 
	}
	 
	 /**
	     * 以行为单位读取文件，常用于读面向行的格式化文件
	     */
	    public static void readFileByLines(String fileName) {
	        File file = new File(fileName);
	        BufferedReader reader = null;
	        try {
	            System.out.println("以行为单位读取文件内容，一次读一整行：");
	            reader = new BufferedReader(new FileReader(file));
	            String tempString = null;
	            int line = 1;
	            // 一次读入一行，直到读入null为文件结束
	            while ((tempString = reader.readLine()) != null) {
	                // 显示行号
	                System.out.println("line " + line + ": " + tempString);
	                line++;
	            }
	            reader.close();
	        } catch (IOException e) {
	            e.printStackTrace();
	        } finally {
	            if (reader != null) {
	                try {
	                    reader.close();
	                } catch (IOException e1) {
	                }
	            }
	        }
	    }
	/**********Classify by MLRDT after ML-MRMR-Feature selection************/
    public void ML_MRMR_FS_ClassifyByMLRDT(String[] options) throws Exception{
    	// the type of similarity evaluation
		String simElvType = "";
		// the type of algorithms
		String algType = "";
		// whether the given data set requires discretization
		boolean bDiscretized;
    	// parsing the parameters
    	// the file directory
    	filePath = Utils.getOption("path", options);
    	// the source file
    	trainArff = Utils.getOption("train", options);
    	// the testing file
    	testArff = Utils.getOption("test", options);
    	// the file of class's information
    	xmlFile = Utils.getOption("xml", options);
    	// the number of attribute dimensions
    	int attrSize = Integer.valueOf(Utils.getOption("attrSize", options));
    	// the number of class labels
    	int labelNum = Integer.valueOf(Utils.getOption("labelNum", options));
    	// the minimum number of instances at a leaf 
    	int minS = Integer.valueOf(Utils.getOption("minS", options));//4;
    	// the number of decision trees
    	int treeNum = Integer.valueOf(Utils.getOption("treeNum", options));//10;
    	// the similarity evaluation method
    	simElvType = Utils.getOption("simElvType", options);
    	// the flag of whether implementing the discretization
    	bDiscretized = Boolean.valueOf(Utils.getOption("bDiscretized", options)).booleanValue();
    	System.out.println("bDiscretized = "+bDiscretized);
    	//boolean bHoldOutTest = Boolean.valueOf(Utils.getOption("bHoldOutTest", options)).booleanValue();
    	// whether using the average voting prediction results
    	boolean bAvgVoting = Boolean.valueOf(Utils.getOption("bAvgVoting", options)).booleanValue();
    	System.out.println("bAvgVoting = "+bAvgVoting);
    	startIndex = attrSize-labelNum;//- 1;--modified on  Feb. 04, 2017 
    	endIndex = attrSize - 1;
    	System.out.println("classIndex-beg-end:"+startIndex+"-"+endIndex);
    	//alph = Double.valueOf(Utils.getOption("alph", options)); // added on  May 11, 2017
    	//System.out.println("alph = "+alph);
    	
    	// specify the number of selected features, added on  May 11, 2017
    	nfea = (int) (alph*(attrSize-labelNum));
    	System.out.println("selected features: "+nfea+":"+alph);
    	// Ensemble model in the list structure
		LinkedList<CBRRDTModel> ensembleRDTModelList = new LinkedList<CBRRDTModel>();
		//Evaluation result;
		//store the data chunk for testing
		Queue<SimpleInstances> ensembleBlock = new LinkedList<SimpleInstances>();
		// the selected feature sets for the adjoining data chunks 
		int[] curFeaInd = new int [nfea];
		double[] curScoreArr = new double[nfea];
		// store all selected feature sets and corresponding score sets used in drifting detection
		ArrayList<int[]> feaIndList = new ArrayList<int[]>();
		ArrayList<double[]> feaScoreList = new ArrayList<double[]>();
	
		EvaluateByMLRDT evluate = new EvaluateByMLRDT();
		
		// store the results relevant to eight evaluation measures
		FileWriter testWriter = new FileWriter(filePath+"\\predictionResult-f"+alph+".txt");
		// store the non-zero values of attributes for multiple data chunks
		ArrayList<Map<Integer,Integer>> nonZeroAttrMapList = new ArrayList<Map<Integer,Integer>>();
		 /********************Read the training data********************/
		long beginTime = System.currentTimeMillis();
		ArffReader rd = new ArffReader();
		System.out.println(filePath+"\\"+trainArff);
		rd.setFilePath(filePath+"\\"+trainArff);
		// set the attribute size (including the class labels)
		rd.setAttrSize(attrSize);
		Map<Integer, Map<String, Integer>> nominalAttrs = new HashMap<Integer, Map<String, Integer>>();
		// the number of instances in the source file
		int instSize = rd.getInstNumAndAttrInforByReader(nominalAttrs);
		// the number of chunks and the size of each data chunk is "dataBlock"
		if ( dataBlock > instSize )
			dataBlock = instSize;
		int nChunk;
		if (instSize % dataBlock == 0) {
			nChunk = instSize / dataBlock;
		} else {
			nChunk = instSize / dataBlock + 1;
		}
        System.out.println("dataChunkNum = "+nChunk);
		BufferedReader br = new BufferedReader(new FileReader(rd.getFilePath()));
		// 
		DataSource ds = new DataSource(filePath + "/" + trainArff);
		Instances trainDataCpy = ds.getDataSet();
	
		long endTime = System.currentTimeMillis();
		System.out.println("The time of reading data:"+(endTime-beginTime)+"ms");
		 // store the result of feature selection
		 BufferedWriter bwFS = new BufferedWriter(new FileWriter(filePath+"/featureSelection-result.txt"));
		 createDir(filePath+"/ensemblePredict/");
		 createDir(filePath+"/SelectedArff/");
		 // store the data distributions of class labels
		 ArrayList<Map<Integer, Double[]>> labelDistVerMapArr = new ArrayList<Map<Integer, Double[]>>();
		 ArrayList<ArrayList<int[]>> labelDistHorMapArr = new ArrayList<ArrayList<int[]>>(); 
		 System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");//added on  Feb. 10, 2017
		/***********Train the model by data chunks****************/
		long newbeginTime = System.currentTimeMillis();
		long trainTime = 0;
		for (int i = 0; i < nChunk; i++) {
//			tmpChunkInsts = new SimpleInstances(trainInstances.getAttributes(), trainInstances.getMat(), 
//					trainInstances.getIds(),trainInstances.getRelation());
			
			Instances chunkInstsCpy = new Instances(trainDataCpy,0);
			weka.core.Instance tmpInst = null;
			beginTime = System.currentTimeMillis();
			//store the non-zero values of attributes, which will be used in the drifting detection
			Map<Integer, Integer> nonZeroAttrMap = new HashMap<Integer, Integer>();
			for (int j = 0; j < dataBlock && (j + i * dataBlock < trainDataCpy.size()) ; j++) {
				tmpInst = trainDataCpy.get(j + i * dataBlock);
				nonZeroAttrMap = nonZeroAttrValueStats(nonZeroAttrMap, tmpInst, startIndex);
				chunkInstsCpy.add((weka.core.Instance)tmpInst);//.setData(anInst.getIndexs(),trainInstances.getMat());//add(trainInstances.get(j + i * dataBlock));
			}
			endTime = System.currentTimeMillis();
			System.out.println("The time of copying a data chunk:"+(endTime-beginTime)+"ms");
			/**********do Statistics of label distributions**********/
			beginTime = System.currentTimeMillis();
			//Map<Integer, Double[]> labelDistVerMap = DoLabelDistrStatisticsByVertical(chunkInstsCpy);
			Map<Integer, Double[]>  labelDistVerMap = new HashMap<Integer, Double[]>();//will get the values in the function of "featureSelectionForRDT" DoLabelDistrStatisticsForCosine(chunkInstsCpy,bDiscretized);
			ArrayList<int[]> labelDistHorMap = DoLabelDistrStatisticsByHorizontal(chunkInstsCpy);
			//ArrayList<int[]> curLabelDistrList = DoLabelDistrStatisticsOverridden(chunkInstsCpy);
			endTime = System.currentTimeMillis();
			System.out.println("The time of DoLabelDistrStatisticsByHorizontal:"+(endTime-beginTime)+"ms");
			trainTime += (endTime-beginTime);
			//SimpleInstances chunkInsts = rd.getInstancesByChunk(br, nominalAttrs, instSize, dataBlock);
			/*********concept drifting detection between the current data chunk and history ones*********/
			beginTime = System.currentTimeMillis();
			SimpleInstances fsBlockInsts = featureSelectionForRDT(labelDistVerMap,  bwFS, curFeaInd, curScoreArr, chunkInstsCpy,i+1, bDiscretized, attrSize, true);
			System.out.println("labelDistVerMap-size = "+labelDistVerMap.size());
			//MultiLabelInstances fsBlockInsts = featureSelection(curFeaInd, curScoreArr, chunkInstsCpy,i+1, bDiscretized);
			endTime = System.currentTimeMillis();
			System.out.println("The time of feature selection:"+(endTime-beginTime)+"ms");
			bwFS.write("||The time of feature selection:"+(endTime-beginTime)+"ms");
			trainTime += (endTime-beginTime);
			if ( i > 0 )
				ensembleBlock.add(fsBlockInsts);	
			else
				ensembleBlock.add(null);//--the first chunk cannot be predicted
			SimpleInstances testMLInst = ensembleBlock.poll();
			driftInfor drifInforObj = null;
			driftInfor oldestDrifInforObj = null;
			LinkedList<driftInfor> similarModels = null;
			// get the copy of the data distribution in the current data chunk
			int[]  feaIndCpy = Arrays.copyOf(curFeaInd, curFeaInd.length);
			double[] scoreArrCpy = Arrays.copyOf(curScoreArr, curScoreArr.length);
			beginTime = System.currentTimeMillis();
			if (feaIndList.size() > 0 ){
				if ( i == 6  ){
					System.out.println(i);
				}
				similarModels = driftDetectionByLabelDistr(testWriter,labelDistVerMapArr, labelDistVerMap,labelDistHorMapArr, labelDistHorMap, labelNum);// added on  May 04, 2017
				//similarModels = driftDetectionByLabelDistrByCosine(labelDistVerMapArr, labelDistVerMap, labelNum);
				//similarModels = driftDetectionByLabelDistrByHam(labelDistHorMapArr, labelDistHorMap, labelNum);// added on  May 04, 2017
				
				drifInforObj = similarModels.getLast();
				oldestDrifInforObj = similarModels.getFirst();
				System.out.println("driftDetectionByLabelDistr-similarModels.size = "+similarModels.size());
				for( int p = 0; p < similarModels.size();p++ ){
					System.out.println(p+":-label"+similarModels.get(p).modelIndex);
				}
				if (drifInforObj.driftType == "drift"){
					labelDistVerMapArr.add(labelDistVerMap);
					labelDistHorMapArr.add(labelDistHorMap);// added by May 04, 2017
					//labelDistListArr.add(curLabelDistrList);
					System.out.println("Drift in Label distribution");
					feaIndList.add(feaIndCpy);
					feaScoreList.add(scoreArrCpy);
					nonZeroAttrMapList.add(nonZeroAttrMap);
					testWriter.write("\n||drifInforObj.driftType=\t"+drifInforObj.driftType+"\t"+"labelDistChange\n");
				}else{
					// no drifting in the label distribution, then detect the feature distributions
					LinkedList<driftInfor> similarModelsByFea = driftDetectionByFeaDistr(similarModels, nonZeroAttrMapList, nonZeroAttrMap, feaIndList, feaScoreList, curFeaInd, curScoreArr,simElvType);
					drifInforObj = similarModelsByFea.getLast();
					oldestDrifInforObj = similarModelsByFea.getFirst();
					System.out.println("driftDetectionByFeaDistr-similarModels.size = "+similarModelsByFea.size());
					for( int p = 0; p < similarModelsByFea.size();p++ ){
						System.out.println(p+":fea-"+similarModelsByFea.get(p).modelIndex);
					}
					if (drifInforObj.driftType == "drift") {// find no similar model, and update the history set by adding the current data distribution
						feaIndList.add(feaIndCpy);
						feaScoreList.add(scoreArrCpy);
						nonZeroAttrMapList.add(nonZeroAttrMap);
						System.out.println("Drift in feature distribution");
						labelDistVerMapArr.add(labelDistVerMap);
						labelDistHorMapArr.add(labelDistHorMap);// added by May 04, 2017
						//labelDistListArr.add(curLabelDistrList);
					}
					if (drifInforObj.driftType == "nondrift"){// find similar model, and update the history set by replace the old model with the current data distribution
						
						feaIndList.remove(oldestDrifInforObj.modelIndex);//(drifInforObj.modelIndex); // only update the most similar model with the current model
						feaScoreList.remove(oldestDrifInforObj.modelIndex);//(drifInforObj.modelIndex);
						feaIndList.add(feaIndCpy);//(drifInforObj.modelIndex,feaIndCpy);
						feaScoreList.add(scoreArrCpy);//(drifInforObj.modelIndex,scoreArrCpy);
						nonZeroAttrMapList.remove(oldestDrifInforObj.modelIndex);//(drifInforObj.modelIndex);
						nonZeroAttrMapList.add(nonZeroAttrMap);//(drifInforObj.modelIndex,nonZeroAttrMap);
						
						labelDistVerMapArr.remove(oldestDrifInforObj.modelIndex);//(drifInforObj.modelIndex);
						labelDistVerMapArr.add(labelDistVerMap);//(drifInforObj.modelIndex, labelDistVerMap);
		
						labelDistHorMapArr.remove(oldestDrifInforObj.modelIndex);//(drifInforObj.modelIndex);// added by May 04, 2017
						labelDistHorMapArr.add(labelDistHorMap);//(drifInforObj.modelIndex,labelDistHorMap);
					}
					testWriter.write("\n||drifInforObj.driftType=\t"+drifInforObj.driftType+"\t"+"FeatureDistChange\n");
				}
			}
			else{
				feaIndList.add(feaIndCpy);
				feaScoreList.add(scoreArrCpy);
				nonZeroAttrMapList.add(nonZeroAttrMap);
				labelDistVerMapArr.add(labelDistVerMap);
				labelDistHorMapArr.add(labelDistHorMap);// added by May 04, 2017
				//labelDistListArr.add(curLabelDistrList);
			}
			endTime = System.currentTimeMillis();
			System.out.println("The time of concept drifting:"+(endTime-beginTime)+"ms");
			trainTime += (endTime-beginTime);
			//***********use the selected model from the ensemble mode to predict***********//*
			beginTime = System.currentTimeMillis();
			if (testMLInst != null) {
				FileWriter bw = new FileWriter(filePath+"/ensemblePredict/ensemblePredict"+i+".txt");
				
				if (drifInforObj.driftType == "drift") {//no similar model, use the model with the maximum similarity to predict
					System.out.println(i+"||predict using drifInforObj.modelIndex = "+drifInforObj.modelIndex);
					CBRRDTModel tmpModel = ensembleRDTModelList.get(drifInforObj.modelIndex);
					if ( tmpModel != null ){
						bw.write("||predicted by CBRRDTModel-"+(drifInforObj.modelIndex+1)+"\n");
						testWriter.write("||predicted by CBRRDTModel-"+(drifInforObj.modelIndex+1)+"\n");
						evluate.PredictBySingelModel(tmpModel, testMLInst, labelNum, attrSize,bw,testWriter);
					}
				}else{ //use the similar models to predict
					System.out.println(i+"||predict using ensembleMode with size = "+(similarModels.size()-1));
					evluate.PredictByEnsembleModels(similarModels, ensembleRDTModelList, testMLInst, labelNum, attrSize, bAvgVoting, bw, testWriter);
				}
				bw.close();
			}
			endTime = System.currentTimeMillis();
			System.out.println("prediction time using the selected model: "+(endTime-beginTime)+"ms");
			
			// generate a new model on the current data chunk
			/********************Generate the random decision trees********************/
	        beginTime = System.currentTimeMillis();
			TreeBuilder treeBuilder = new TreeBuilder(0, TreeBuilder.CBR_RDT);
			CBRRDTModel model = new CBRRDTModel();
			treeBuilder.setInstances(fsBlockInsts);
			treeBuilder.setMaxDeep(attrSize/2);
			treeBuilder.setMaxS(minS);
			treeBuilder.setClsSize(labelNum);
			treeBuilder.init();
			Node[] trees = treeBuilder.buildTrees(treeNum);
			treeBuilder.clear();
			model.init(trees, fsBlockInsts.getAttributes(), 1, minS);
			// not delete 
			if (drifInforObj != null && drifInforObj.driftType == "nondrift" ){// update the ensemble model by remove old concept and add the new one
				ensembleRDTModelList.remove(drifInforObj.modelIndex);
			}
			
			if (ensembleRDTModelList.size() >= modelSize){
				ensembleRDTModelList.removeFirst();
				feaIndList.remove(0);
				feaScoreList.remove(0);
				nonZeroAttrMapList.remove(0);
				labelDistVerMapArr.remove(0);
				labelDistHorMapArr.remove(0);// added by May 04, 2017
				//labelDistListArr.remove(0);
			}
			ensembleRDTModelList.add(model);
			endTime = System.currentTimeMillis();
			System.out.println("the time of generating a single RDT model: "+(endTime-beginTime)+"ms");
			System.out.println("ensemble-size: "+ensembleRDTModelList.size());
			trainTime += (endTime-beginTime);
		
		}
		endTime = System.currentTimeMillis();
		testWriter.write("\n||the time of training models: "+(endTime-newbeginTime)+"ms\n");
		testWriter.write("\n||the time of training models-concise: "+(trainTime)+"ms\n");
		System.out.println("\n||the time of training models-concise: "+(trainTime)+"ms");
		// using the test data to implement the holdout test
		boolean bHoldOutTest = true;
		if ( bHoldOutTest ){
//			HoldOutTestByChunks(filePath, testArff, labelDistMapArr, ensembleRDTModelList, null, nonZeroAttrMapList, feaIndList,
//					feaScoreList, attrSize, labelNum, bDiscretized, true, bAvgVoting, simElvType);
			HoldOutTestInBatch( filePath, testArff, ensembleRDTModelList, null, nonZeroAttrMapList, feaIndList,
					 feaScoreList, attrSize, labelNum, bDiscretized, true, bAvgVoting, simElvType);

		}
		//trainWriter.close();
		// added on  May 15, 2017
		 Iterator it = featureSetOfChunks.keySet().iterator();  
		 int key = 0;
		 double tmpVal = 0;
		 while(it.hasNext()) {  
			 key = (int)it.next();  
			 tmpVal = featureSetOfChunks.get(key);
			// System.out.println("key:" + key);  
			// System.out.println("value:" + tmpVal);  
			 bwFS.write(key+"\t"+tmpVal+'\n');
		 }  
		 bwFS.write("feature selection time = "+trainTime+"\n");
		 bwFS.close();
		 testWriter.close();
    }
    
    /**
	 * HoldOut Test using given testing data
	 * 
	 * @param filePath
	 * 			the path of the source data
	 * @param testFile
	 * 			 the file of testing data
	 * @param ensembleRDTModelList
	 * 			 the ensemble model of RDT in the training
	 * @param ensembleMLModelList
	 * 			 the ensemble model of non-RDT multi-label learner in the training
	 * @param nonZeroAttrMapList 
	 * 			 store the non-zero values of attributes for training data chunks
	 * @param feaIndList
	 * 			 store the selected feature sets of training data 
	 * @param feaScoreList
	 * 			 store the score sets corresponding to the selected feature sets of training data
	 * @param attrSize
	 * 			 the number of attribute dimensions
	 * @param labelNum
	 * 			 the number of class labels
	 * @param bDiscretized
	 * 			 the flag of whether implementing the discretization
	 * @param bUseRDT
	 * 			 whether the current prediction model is RDT or not
	 * @param bAvgVoting
	 * 			 whether using the average voting prediction results or the sum of the prediction results
	 * @param simElvType
	 * 			 the type of similarity evaluation used in the drifting detection
	 * @return 
	 * 
     * @throws Exception 
	 */
	 private void HoldOutTestInBatch(String filePath, String testFile, LinkedList<CBRRDTModel> ensembleRDTModelList, LinkedList<MultiLabelLearner> ensembleMLModelList, ArrayList<Map<Integer,Integer>> nonZeroAttrMapList, ArrayList<int[]> feaIndList,
				ArrayList<double[]> feaScoreList, int attrSize, int labelNum, boolean bDiscretized, boolean bUseRDT, boolean bAvgVoting, String simElvType) throws Exception{
		long beginTime = System.currentTimeMillis();
		long endTime = 0;
	//	BufferedWriter bwFS = new BufferedWriter(new FileWriter(filePath+"/featureSelection_onTestData.txt"));
		FileWriter testBW = new FileWriter(filePath+"/ensemblePredict_onTestData.txt");
		ArffReader rd = new ArffReader();
		System.out.println(filePath+"\\"+testFile);
		rd.setFilePath(filePath+"\\"+testFile);
		// set the attribute size (including the class labels)
		rd.setAttrSize(attrSize);
		SimpleInstances testInsts = rd.getInstances();
		
		System.out.println("||predict using ensembleMode with size = "+(ensembleRDTModelList.size()));
		EvaluateByMLRDT evluate = new EvaluateByMLRDT();
		evluate.PredictByEnsembleModels(null, ensembleRDTModelList, testInsts, labelNum, attrSize, bAvgVoting, testBW, testBW);
		endTime = System.currentTimeMillis();
		System.out.println("prediction time using the ensemble model of MLRDT: "+(endTime-beginTime)+"ms");
		testBW.close();
		
		
//		DataSource ds = new DataSource(filePath+"\\"+testFile);
//		Instances testInstances = ds.getDataSet();
//		//store the non-zero values of attributes, which will be used in the drifting detection
//		Map<Integer, Integer> nonZeroAttrMap = new HashMap<Integer, Integer>();
//		for ( int i = 0; i < testInstances.size(); i++ ){
//			nonZeroAttrMap = nonZeroAttrValueStats(nonZeroAttrMap, testInstances.get(i), startIndex);
//		}
//		int[] curFeaInd = new int [nfea];
//		double[] curScoreArr = new double [nfea];
//		BufferedWriter bwFS = new BufferedWriter(new FileWriter(filePath+"/featureSelection_onTestData.txt"));
//		FileWriter testBW = new FileWriter(filePath+"/ensemblePredict_onTestData.txt");
//		if ( bUseRDT ){
//			SimpleInstances fsTestInsts = featureSelectionForRDT(bwFS, curFeaInd, curScoreArr, testInstances,1, bDiscretized,attrSize);
//			// using all ensemble model to predict
//			LinkedList<driftInfor> similarModels = new LinkedList<driftInfor>();
//			for ( int i = 0; i < ensembleRDTModelList.size()+1; i++ ){// the last object is not usefull
//				driftInfor drifInforObj = new driftInfor();
//				drifInforObj.modelIndex = i;
//				similarModels.add(drifInforObj);
//			}
//			System.out.println("||predict using ensembleMode with size = "+(similarModels.size()-1));
//			EvaluateByMLRDT evluate = new EvaluateByMLRDT();
//			evluate.PredictByEnsembleModels(similarModels, ensembleRDTModelList, fsTestInsts, labelNum, attrSize, bAvgVoting, testBW, testBW);
//			endTime = System.currentTimeMillis();
//			System.out.println("prediction time using the ensemble model of MLRDT: "+(endTime-beginTime)+"ms");
//
//			/*LinkedList<driftInfor> similarModels = driftDetection(nonZeroAttrMapList, nonZeroAttrMap, feaIndList, feaScoreList, curFeaInd, curScoreArr,simElvType);
//			EvaluateByMLRDT evluate = new EvaluateByMLRDT();
//			driftInfor drifInforObj = similarModels.getLast();
//			if (fsTestInsts != null) {
//				FileWriter bw = new FileWriter(filePath+"/ensemblePredict_onTestData.txt");
//				if (drifInforObj.driftType == "drift") {//no similar model, use the model with the maximum similarity to predict
//					System.out.println("||predict using drifInforObj.modelIndex = "+drifInforObj.modelIndex);
//					CBRRDTModel tmpModel = ensembleRDTModelList.get(drifInforObj.modelIndex);
//					if ( tmpModel != null ){
//						bw.write("||predicted by CBRRDTModel-"+(drifInforObj.modelIndex+1)+"\n");
//						//testWriter.write("||predicted by CBRRDTModel-"+(drifInforObj.modelIndex+1)+"\n");
//						evluate.PredictBySingelModel(tmpModel, fsTestInsts, labelNum, attrSize,bw,bw);
//					}
//				}else{ //use the similar models to predict
//					System.out.println("||predict using ensembleMode with size = "+(similarModels.size()-1));
//					evluate.PredictByEnsembleModels(similarModels, ensembleRDTModelList, fsTestInsts, labelNum, attrSize, bAvgVoting, bw, bw);
//				}
//				
//				endTime = System.currentTimeMillis();
//				System.out.println("prediction time using the ensemble model of MLRDT: "+(endTime-beginTime)+"ms");
//			}//end if (fsTestInsts != null)
//			*/
//		}else{
//			MultiLabelInstances fsTestInsts = featureSelection(bwFS, curFeaInd, curScoreArr, testInstances,1, bDiscretized);
//			LinkedList<driftInfor> similarModels = driftDetectionByFeaDistr(null, nonZeroAttrMapList, nonZeroAttrMap, feaIndList, feaScoreList, curFeaInd, curScoreArr,simElvType);
//			if (fsTestInsts != null) {
//				Evaluator eval = new Evaluator();
//				List<Measure> measures = InitMeasures();
//				Evaluation result = eval.evaluateByEnsembleModels(similarModels, ensembleMLModelList, fsTestInsts, measures, bAvgVoting);
//				testBW.write("||predicted by ensemble non-RDT-ML-Learner\n");
//				testBW.write(result.toString());
//				
//			}
//			endTime = System.currentTimeMillis();
//			System.out.println("prediction time using the ensemble model of non-MLRDT: "+(endTime-beginTime)+"ms");
//		}
//		bwFS.close();
//		testBW.close();
	 }
	 /**
		 * HoldOut Test using given testing data
		 * 
		 * @param filePath
		 * 			the path of the source data
		 * @param testFile
		 * 			 the file of testing data
		 * @param labelDistMapArr
		 * 			 the data distribution of class labels
		 * @param ensembleRDTModelList
		 * 			 the ensemble model of RDT in the training
		 * @param ensembleMLModelList
		 * 			 the ensemble model of non-RDT multi-label learner in the training
		 * @param nonZeroAttrMapList 
		 * 			 store the non-zero values of attributes for training data chunks
		 * @param feaIndList
		 * 			 store the selected feature sets of training data 
		 * @param feaScoreList
		 * 			 store the score sets corresponding to the selected feature sets of training data
		 * @param attrSize
		 * 			 the number of attribute dimensions
		 * @param labelNum
		 * 			 the number of class labels
		 * @param bDiscretized
		 * 			 the flag of whether implementing the discretization
		 * @param bUseRDT
		 * 			 whether the current prediction model is RDT or not
		 * @param bAvgVoting
		 * 			 whether using the average voting prediction results or the sum of the prediction results
		 * @param simElvType
		 * 			 the type of similarity evaluation used in the drifting detection
		 * @return 
		 * 
	     * @throws Exception 
		 */
		 private void HoldOutTestByChunks(String filePath, String testFile, ArrayList<Map<Integer, Double[]>> labelDistMapArr, LinkedList<CBRRDTModel> ensembleRDTModelList, LinkedList<MultiLabelLearner> ensembleMLModelList, ArrayList<Map<Integer,Integer>> nonZeroAttrMapList, ArrayList<int[]> feaIndList,
					ArrayList<double[]> feaScoreList, int attrSize, int labelNum, boolean bDiscretized, boolean bUseRDT, boolean bAvgVoting, String simElvType) throws Exception{
			 long beginTime = System.currentTimeMillis();
			long endTime = 0;
			DataSource ds = new DataSource(filePath+"\\"+testFile);
			Instances testInstances = ds.getDataSet();
			EvaluateByMLRDT evluate = new EvaluateByMLRDT();
			
			int[] curFeaInd = new int [nfea];
			double[] curScoreArr = new double [nfea];
			BufferedWriter bwFS = new BufferedWriter(new FileWriter(filePath+"/featureSelection_onTestData.txt"));
			FileWriter testBW = new FileWriter(filePath+"/ensemblePredict_onTestData.txt");
			// the number of chunks and the size of each data chunk is "dataBlock"
			int nChunk;
			if (testInstances.size() % dataBlock == 0) {
				nChunk = testInstances.size() / dataBlock;
			} else {
				nChunk = testInstances.size() / dataBlock + 1;
			}
			if ( bUseRDT ){
				FileWriter bw = new FileWriter(filePath+"/ensemblePredict_onTestData.txt");
				for (int i = 0; i < nChunk; i++) {
					Instances chunkInstsCpy = new Instances(testInstances,0);
					weka.core.Instance tmpInst = null;
					beginTime = System.currentTimeMillis();
					//store the non-zero values of attributes, which will be used in the drifting detection
					Map<Integer, Integer> nonZeroAttrMap = new HashMap<Integer, Integer>();
					for (int j = 0; j < dataBlock && (j + i * dataBlock < testInstances.size()) ; j++) {
						tmpInst = testInstances.get(j + i * dataBlock);
						nonZeroAttrMap = nonZeroAttrValueStats(nonZeroAttrMap, tmpInst, startIndex);
						chunkInstsCpy.add((weka.core.Instance)tmpInst);//.setData(anInst.getIndexs(),trainInstances.getMat());//add(trainInstances.get(j + i * dataBlock));
					}
					endTime = System.currentTimeMillis();
					System.out.println("The time of copying a data chunk:"+(endTime-beginTime)+"ms");
//					/**********do Statistics of label distributions**********/
//					beginTime = System.currentTimeMillis();
//					Map<Integer, Double[]> labelDistMap = DoLabelDistrStatistics(chunkInstsCpy);
//					endTime = System.currentTimeMillis();
//					System.out.println("The time of label distribution's statistics:"+(endTime-beginTime)+"ms");
					/*********concept drifting detection between the current data chunk and history ones*********/
					SimpleInstances testMLInstChunk = featureSelectionForRDT(null, bwFS, curFeaInd, curScoreArr, chunkInstsCpy,i+1, bDiscretized, attrSize, false);
					driftInfor drifInforObj = null;
//					LinkedList<driftInfor> similarModels = driftDetectionByLabelDistr(labelDistMapArr, labelDistMap, labelNum);
//					drifInforObj = similarModels.getLast();
//					if (drifInforObj.driftType == "drift"){
//						System.out.println("||predict using drifInforObj.modelIndex = "+drifInforObj.modelIndex);
//						CBRRDTModel tmpModel = ensembleRDTModelList.get(drifInforObj.modelIndex);
//						if ( tmpModel != null ){
//							bw.write("||predicted by CBRRDTModel-"+(drifInforObj.modelIndex+1)+"\n");
//							//testWriter.write("||predicted by CBRRDTModel-"+(drifInforObj.modelIndex+1)+"\n");
//							evluate.PredictBySingelModel(tmpModel, testMLInstChunk, labelNum, attrSize,bw,bw);
//						}
//					}else{// no drifting in the label distribution, then detect the feature distributions
//						similarModels = driftDetectionByFeaDistr(nonZeroAttrMapList, nonZeroAttrMap, feaIndList, feaScoreList, curFeaInd, curScoreArr,simElvType);
//						drifInforObj = similarModels.getLast();
//						if (drifInforObj.driftType == "drift") {// find no similar model, and update the history set by adding the current data distribution
//							System.out.println("||predict using drifInforObj.modelIndex = "+drifInforObj.modelIndex);
//							CBRRDTModel tmpModel = ensembleRDTModelList.get(drifInforObj.modelIndex);
//							if ( tmpModel != null ){
//								bw.write("||predicted by CBRRDTModel-"+(drifInforObj.modelIndex+1)+"\n");
//								//testWriter.write("||predicted by CBRRDTModel-"+(drifInforObj.modelIndex+1)+"\n");
//								evluate.PredictBySingelModel(tmpModel, testMLInstChunk, labelNum, attrSize,bw,bw);
//							}
//						}
//						if (drifInforObj.driftType == "nondrift"){// find similar model, and update the history set by replace the old model with the current data distribution
//							System.out.println("||predict using ensembleMode with size = "+(similarModels.size()-1));
//							evluate.PredictByEnsembleModels(similarModels, ensembleRDTModelList, testMLInstChunk, labelNum, attrSize, bAvgVoting, bw, bw);
//						}
//					}
					LinkedList<driftInfor> similarModels = driftDetectionByFeaDistr(null, nonZeroAttrMapList, nonZeroAttrMap, feaIndList, feaScoreList, curFeaInd, curScoreArr,simElvType);
					drifInforObj = similarModels.getLast();
					
					if (drifInforObj.driftType == "drift") {//no similar model, use the model with the maximum similarity to predict
						System.out.println("||predict using drifInforObj.modelIndex = "+drifInforObj.modelIndex);
						CBRRDTModel tmpModel = ensembleRDTModelList.get(drifInforObj.modelIndex);
						if ( tmpModel != null ){
							bw.write("||predicted by CBRRDTModel-"+(drifInforObj.modelIndex+1)+"\n");
							//testWriter.write("||predicted by CBRRDTModel-"+(drifInforObj.modelIndex+1)+"\n");
							evluate.PredictBySingelModel(tmpModel, testMLInstChunk, labelNum, attrSize,bw,bw);
						}
					}else{ //use the similar models to predict
						System.out.println("||predict using ensembleMode with size = "+(similarModels.size()-1));
						evluate.PredictByEnsembleModels(similarModels, ensembleRDTModelList, testMLInstChunk, labelNum, attrSize, bAvgVoting, bw, bw);
					}
				}//end for (int i = 0; i < nChunk; i++)
				bw.close();
			}else{
//				MultiLabelInstances fsTestInsts = featureSelection(bwFS, curFeaInd, curScoreArr, testInstances,1, bDiscretized);
//				LinkedList<driftInfor> similarModels = driftDetection(nonZeroAttrMapList, nonZeroAttrMap, feaIndList, feaScoreList, curFeaInd, curScoreArr,simElvType);
//				if (fsTestInsts != null) {
//					Evaluator eval = new Evaluator();
//					List<Measure> measures = InitMeasures();
//					Evaluation result = eval.evaluateByEnsembleModels(similarModels, ensembleMLModelList, fsTestInsts, measures, bAvgVoting);
//					testBW.write("||predicted by ensemble non-RDT-ML-Learner\n");
//					testBW.write(result.toString());
//					
//				}
				endTime = System.currentTimeMillis();
				System.out.println("prediction time using the ensemble model of non-MLRDT: "+(endTime-beginTime)+"ms");
			}
			bwFS.close();
			testBW.close();
		 }
		
    /**
	 * Data Tranformation for Multi-Key Sorting
	 * 
	 * @param filePath
	 * 			the path of the source data
	 * @param srcFile
	 * 			 the file of the original data 
	 * @return 
     * @throws Exception 
	 */
	 private void DataTranformForMultiKeySort(String filePath, String srcFile, int attrSize, int labelNum) throws Exception{
		 FileWriter writer = new FileWriter(filePath+"\\"+srcFile+".transformed");
		// FileReader reader = new FileReader(filePath+"\\"+srcFile);
		 DataSource ds = new DataSource(filePath+"\\"+srcFile);
		 Instances data = ds.getDataSet();
		 int startIndex = attrSize - labelNum ;//- 1;--modified on  Feb. 04, 2017 
		 int endIndex = attrSize - 1;
		 String line = "";
		 for ( int i = 0; i < data.size(); i++ ){
			 line = "\t";
			 for ( int j = startIndex; j<= endIndex; j++ ){
				 line += (int)data.get(i).value(j)+"\t";
			 }
			 writer.write(line+"\n");
		 }
		 writer.close();
	}
    /**
	 * Initialize the nine evaluation measures 
	 * 
	 * @return List<Measure> measures
	 */
	private List<Measure> InitMeasures(){
		List<Measure> measures = new ArrayList<Measure>(9); // added on  Feb. 04， 2017
		measures.add(new HammingLoss());
		measures.add(new OneError());
		measures.add(new Coverage());
		measures.add(new RankingLoss());
		measures.add(new AveragePrecision());
		measures.add(new ExampleBasedFMeasure());
		measures.add(new ExampleBasedAccuracy());
		measures.add(new ExampleBasedPrecision());
		measures.add(new ExampleBasedRecall());
		return measures;
	}
    /**********Classify by mulan after ML-MRMR-Feature selection************/
	public void ML_MRMR_FS_ClassifyByMulan(String[] options) throws Exception {
		// Ensemble model in the list structure
		LinkedList<MultiLabelLearner> cmEnsembleModelList = new LinkedList<MultiLabelLearner>();
		// the writer
		BufferedWriter bw = null; 
		
		// 存放数据块
		Queue<MultiLabelInstances> ensembleBlock = new LinkedList<MultiLabelInstances>();
		// the selected feature sets for the adjoining data chunks 
		int[] curFeaInd = new int [nfea];
		double[] curScoreArr = new double[nfea];
		// store all selected feature sets and corresponding score sets used in drifting detection
		ArrayList<int[]> feaIndList = new ArrayList<int[]>();
		ArrayList<double[]> feaScoreList = new ArrayList<double[]>();
		// the drifting type
		String driftType = "";
		// the type of similarity evaluation
		String simElvType = "";
		// the type of algorithms
		String algType = "";
		// whether the given data set requires discretization
		boolean bDiscretized;
		// set the evaluation measures
		Evaluator eval = new Evaluator();
		List<Measure> measures = InitMeasures(); // added on  Feb. 04， 2017
		 // store the data distributions of class labels
		 ArrayList<Map<Integer, Double[]>> labelDistMapArr = new ArrayList<Map<Integer, Double[]>>();
		// store the non-zero values of attributes for multiple data chunks
		ArrayList<Map<Integer,Integer>> nonZeroAttrMapList = new ArrayList<Map<Integer,Integer>>();
		//parsing the parameters
		filePath = Utils.getOption("path", options);
		// the source file
		trainArff = Utils.getOption("train", options);
		// the testing file
		testArff = Utils.getOption("test", options);
		xmlFile = Utils.getOption("xml", options);
		int attrSize = Integer.valueOf(Utils.getOption("attrSize", options));
		int labelNum = Integer.valueOf(Utils.getOption("labelNum", options));
		simElvType = Utils.getOption("simElvType", options);
		algType = Utils.getOption("algType", options);
		System.out.println("algType: "+algType);
		bDiscretized = Boolean.valueOf(Utils.getOption("bDiscretized", options)).booleanValue();
		boolean bHoldOutTest = Boolean.valueOf(Utils.getOption("bHoldOutTest", options)).booleanValue();
		System.out.println("bDiscretized = "+bDiscretized);
		// whether using the average voting prediction results
		boolean bAvgVoting = Boolean.valueOf(Utils.getOption("bAvgVoting", options)).booleanValue();
		
		startIndex = attrSize - labelNum ;//- 1;--modified on  Feb. 04, 2017 
		endIndex = attrSize - 1;
		System.out.println("classIndex-beg-end:"+startIndex+"-"+endIndex);
		// 构建Instances块用来进行特征选择（后面会转成MultiInstances）
		DataSource ds = new DataSource(filePath + "/" + trainArff);
		Instances data = ds.getDataSet();
		//the number of chunks and the size of each data chunk is "dataBlock"
		int nChunk;
		if (data.numInstances() % dataBlock == 0) {
			nChunk = (data.numInstances()) / dataBlock;
		} else {
			nChunk = (data.numInstances()) / dataBlock + 1;
		}
        System.out.println("dataChunkNum = "+nChunk);
        // store the result of feature selection
        BufferedWriter bwFS = new BufferedWriter(new FileWriter(filePath+"/featureSelection-result.txt"));
		 // create the folders of ensemblePrediction" and "SelectedArff"
		 createDir(filePath+"/ensemblePredict/");
		 createDir(filePath+"/SelectedArff/");
		// store the results relevant to eight evaluation measures
		FileWriter testWriter = new FileWriter(filePath+"\\predictionResult.txt");
		
        // record the time consumption
        long beginTime = 0;
        long endTime = 0;
        Instance tmpInst = null;
		for (int i = 0; i < nChunk; i++) {
			Instances blockInsts = new Instances(data, 0);
			//store the non-zero values of attributes, which will be used in the drifting detection
			Map<Integer, Integer> nonZeroAttrMap = new HashMap<Integer, Integer>();
			for (int j = 0; j < dataBlock && (j + i * dataBlock < data.size()) ; j++) {
				tmpInst = (Instance) data.get(j + i * dataBlock);
				nonZeroAttrMap = nonZeroAttrValueStats(nonZeroAttrMap, (weka.core.Instance)tmpInst, startIndex);
				blockInsts.add((weka.core.Instance)tmpInst);
			}
			beginTime = System.currentTimeMillis();
			Map<Integer, Double[]> labelDistMap = DoLabelDistrStatisticsByVertical(blockInsts);
			endTime = System.currentTimeMillis();
			System.out.println("The time of label distribution's statistics:"+(endTime-beginTime)+"ms");
			/*********concept drifting detection between the current data chunk and history ones*********/
			MultiLabelInstances fsBlockInsts = featureSelection(bwFS, curFeaInd, curScoreArr, blockInsts,i+1, bDiscretized);
			if ( i > 0 )
				ensembleBlock.add(fsBlockInsts);	
			else
				ensembleBlock.add(null);//--the first chunk cannot be predicted
			MultiLabelInstances testMLInst = ensembleBlock.poll();
			driftInfor drifInforObj = null;
			// the lsit of multiple similar models
			LinkedList<driftInfor> similarModels = null;
			// get the copy of the data distribution in the current data chunk
			int[] feaIndCpy = new int [curFeaInd.length];
			double[] scoreArrCpy = new double [curScoreArr.length];
			for ( int p = 0; p < curFeaInd.length; p++ ){
				feaIndCpy[p] = curFeaInd[p];
			}
			for ( int p = 0; p < curScoreArr.length; p++ ){
				scoreArrCpy[p] = curScoreArr[p];
			}
//			if (feaIndList.size() > 0 ){
//				similarModels = driftDetectionByFeaDistr(nonZeroAttrMapList, nonZeroAttrMap, feaIndList, feaScoreList, curFeaInd, curScoreArr,simElvType);
//				drifInforObj = similarModels.getLast();
//				if (drifInforObj.driftType == "drift") {// find no similar model, and update the history set by adding the current data distribution
//					feaIndList.add(feaIndCpy);
//					feaScoreList.add(scoreArrCpy);
//					nonZeroAttrMapList.add(nonZeroAttrMap);
//				}
//				if (drifInforObj.driftType == "nondrift") {// find similar model, and update the history set by replace the old model with the current data distribution
//					feaIndList.remove(drifInforObj.modelIndex); // only update the most similar model with the current model
//					feaScoreList.remove(drifInforObj.modelIndex);
//					feaIndList.add(drifInforObj.modelIndex,feaIndCpy);
//					feaScoreList.add(drifInforObj.modelIndex,scoreArrCpy);
//					nonZeroAttrMapList.remove(drifInforObj.modelIndex);
//					nonZeroAttrMapList.add(drifInforObj.modelIndex,nonZeroAttrMap);
//				}
//				testWriter.write("\n||drifInforObj.driftType="+drifInforObj.driftType+"\n");
//			}
//			else{
//				feaIndList.add(feaIndCpy);
//				feaScoreList.add(scoreArrCpy);
//				nonZeroAttrMapList.add(nonZeroAttrMap);
//			}
			/***************Drifting detection**************/
			if (feaIndList.size() > 0 ){
				similarModels = driftDetectionByLabelDistrByCosine(labelDistMapArr, labelDistMap, labelNum);
				drifInforObj = similarModels.getLast();
				if (drifInforObj.driftType == "drift"){
					labelDistMapArr.add(labelDistMap);
					System.out.println("Drift in Label distribution");
					feaIndList.add(feaIndCpy);
					feaScoreList.add(scoreArrCpy);
					nonZeroAttrMapList.add(nonZeroAttrMap);
					
				}else{// no drifting in the label distribution, then detect the feature distributions
					similarModels = driftDetectionByFeaDistr(similarModels, nonZeroAttrMapList, nonZeroAttrMap, feaIndList, feaScoreList, curFeaInd, curScoreArr,simElvType);
					drifInforObj = similarModels.getLast();
					if (drifInforObj.driftType == "drift") {// find no similar model, and update the history set by adding the current data distribution
						feaIndList.add(feaIndCpy);
						feaScoreList.add(scoreArrCpy);
						nonZeroAttrMapList.add(nonZeroAttrMap);
						System.out.println("Drift in feature distribution");
						labelDistMapArr.add(labelDistMap);
					}
					if (drifInforObj.driftType == "nondrift"){// find similar model, and update the history set by replace the old model with the current data distribution
						feaIndList.remove(drifInforObj.modelIndex); // only update the most similar model with the current model
						feaScoreList.remove(drifInforObj.modelIndex);
						feaIndList.add(drifInforObj.modelIndex,feaIndCpy);
						feaScoreList.add(drifInforObj.modelIndex,scoreArrCpy);
						nonZeroAttrMapList.remove(drifInforObj.modelIndex);
						nonZeroAttrMapList.add(drifInforObj.modelIndex,nonZeroAttrMap);
						
						labelDistMapArr.remove(drifInforObj.modelIndex);
						labelDistMapArr.add(drifInforObj.modelIndex, labelDistMap);
					}
					testWriter.write("\n||drifInforObj.driftType="+drifInforObj.driftType+"\n");
				}
			}
			else{
				feaIndList.add(feaIndCpy);
				feaScoreList.add(scoreArrCpy);
				nonZeroAttrMapList.add(nonZeroAttrMap);
				
				labelDistMapArr.add(labelDistMap);
			}
			/***********use the selected model from the ensemble mode to predict***********/
			beginTime = System.currentTimeMillis();
			if (testMLInst != null) {
				Evaluation result = eval.evaluateByEnsembleModels(similarModels, cmEnsembleModelList, testMLInst, measures, bAvgVoting);
				System.out.println(result);
				testWriter.write("||predicted by ensemble non-RDT-ML-Learner\n");
				testWriter.write(result.toString());
			}
			endTime = System.currentTimeMillis();
			System.out.println("prediction time using the selected model: "+(endTime-beginTime)+"ms");
			
			/****************generate a new model on the current data chunk************/
			beginTime = System.currentTimeMillis();
			MultiLabelLearner model = generateModelByAlgType(algType, fsBlockInsts); 
			if (drifInforObj != null && drifInforObj.driftType == "nondrift" ){// update the ensemble model by remove old concept and add the new one
				cmEnsembleModelList.remove(drifInforObj.modelIndex);
			}
			if (cmEnsembleModelList.size() >= modelSize){
				cmEnsembleModelList.removeFirst();
				feaIndList.remove(0);
				feaScoreList.remove(0);
				nonZeroAttrMapList.remove(0);
				labelDistMapArr.remove(0);
			}
			cmEnsembleModelList.add(model);
			endTime = System.currentTimeMillis();
			System.out.println("the time of generating a single model: "+(endTime-beginTime)+"ms");
			System.out.println("ensemble-size: "+cmEnsembleModelList.size());
		}
		bwFS.close();
		testWriter.close();
}

	/**
	 * Generate the model according to the value of "algType"
	 * 
	 * @param algType
	 * 			the type of algorithm, such as "MLKNN" 
	 * @param fsBlockInsts 
	 * 			the current data chunk
	 * @throws Exception
	 */
	public MultiLabelLearner generateModelByAlgType(String algType, MultiLabelInstances fsBlockInsts) throws Exception{
		MultiLabelLearner learner = null;
		switch (algType){
		case "MLKNN":{
			int numOfNeighbors = 10;
			double smooth = 1.0;
			MLkNN model = new MLkNN(Math.min(numOfNeighbors,fsBlockInsts.getNumInstances()/2), smooth);
			//model.setDebug(true);
			model.build(fsBlockInsts);
			return model;
		}
//		case "MLRDT":
//			break;
		case "BRKNN":{
			int numOfNeighbors = 10;
			BRkNN model = new BRkNN(Math.min(numOfNeighbors,fsBlockInsts.getNumInstances()/2));
			model.build(fsBlockInsts);
			return model;
		}
		case "IBLR_ML":{
			int numOfNeighbors = 10;
			IBLR_ML model = new IBLR_ML(Math.min(numOfNeighbors,fsBlockInsts.getNumInstances()/2));
			model.build(fsBlockInsts);
			return model;
		}
		case "RAKEL":{
			 RAkEL model = new RAkEL();//(new LabelPowerset(new J48()));
			 model.build(fsBlockInsts);
			 return model;
		}
		case "RAkELD":{
			RAkELd model = new RAkELd();//(new LabelPowerset(new J48()));
			model.build(fsBlockInsts);
			return model;
		}
		case "HOMER":{
		    Classifier brClassifier = new NaiveBayes();
	        Method method = Method.BalancedClustering;
	        HOMER model = new HOMER(new BinaryRelevance(brClassifier),2,method);
	        model.build(fsBlockInsts);
		}
		case "BPMLL":{
			 BPMLL model = new BPMLL();
			 model.build(fsBlockInsts);
			 return model;
		}
		case "CLR":{
			 CalibratedLabelRanking model = new CalibratedLabelRanking(new SMO());
			 model.build(fsBlockInsts);
			 return model;
		}
		case "ESL":{
			EnsembleOfSubsetLearners model = new EnsembleOfSubsetLearners();
			model.build(fsBlockInsts);
			return model;
		}
		default: 
			break;
		}
		return learner;
	}
	/**
	 * 对Instances块进行特征选择，并返回一个属性减少的MultiLabelInstances
	 * note:MultiLabelInstances是mulan中的多标签的类。 MultilInstances是自己实现的多标签的类
	 * 
     * @param bwFS
	 *    		BufferedWriter, store the results of feature selection 
	 * @param feaInd
	 * 			the selected feature set
	 * @param scoreArr 
	 * 			the score sets for selected feature set
	 * @param blockInts 
	 * 			the data set for feature selection
	 * @param indexChunk 
	 * 			the index of the current data chunk
	 * @param bDiscretized 
	 * 			whether using the discretization
	 * @return
	 * @throws Exception
	 */
	public MultiLabelInstances featureSelection(BufferedWriter bwFS, int[] feaInd, double[] scoreArr, Instances blockInsts, int indexChunk, boolean bDiscretized)
			throws Exception {
		long beginTime = System.currentTimeMillis();
		mRMR mrmr = new mRMR(nfea, selectMethod, discretize);
		MultiInstances datas = new MultiInstances(blockInsts, startIndex,
				endIndex);
		mrmr.setDatas(datas);
		long endTime = 0;
		if (bDiscretized && discretize != 9999.0) { // modified on  Feb. 07, 2017
			// 调用MultiInstances中的离散化的方法
			mrmr.getDatas().discretize(discretize, startIndex, endIndex);
			endTime = System.currentTimeMillis();
		    System.out.println("discretization time: "+(endTime-beginTime)+"ms");
		}
	     beginTime = System.currentTimeMillis();
	    
	     int[] tmpFeaInd = mrmr.mRMRSelect(bwFS, scoreArr,indexChunk);
	     for ( int i = 0; i < tmpFeaInd.length; i++ )
	    	 feaInd[i] = tmpFeaInd[i];
		 endTime = System.currentTimeMillis();
	     System.out.println("feature select time on instances with size of "+blockInsts.numInstances()+":"+(endTime-beginTime)+"ms");
		// 经过特征选择之后的名字
		String fsFile = mrmr.toArffFile(blockInsts, feaInd, filePath, indexChunk);//---modified on  Feb. 4
	    // datas.getDataSet()-->is discretized data chunk
		//String fsFile = mrmr.toArffFile(mrmr.getDatas().getDataSet(), feaInd, filePath, nChunk);//---modified on  Feb. 4
		MultiLabelInstances fsBlockInsts = new MultiLabelInstances(filePath
				+ "/SelectedArff/" + fsFile, filePath + "/" + xmlFile);
		
		return fsBlockInsts;
	}
	
	/**
	 * 对Instances块进行特征选择，并返回一个属性减少的MultiLabelInstances
	 * note:MultiLabelInstances是mulan中的多标签的类。 MultilInstances是自己实现的多标签的类
	 * @param labelDistVerMap
	 * 			the label distribution for cosine computation
	 * @param bwFS
	 * 			BufferedWriter, store the results of feature selection 
	 * @param feaInd
	 * 			the selected feature set
	 * @param scoreArr 
	 * 			the score sets for selected feature set
	 * @param blockInts 
	 * 			the data set for feature selection
	 * @param indexChunk 
	 * 			the index of the current data chunk
	 * @param bDiscretized 
	 * 			whether using the discretization
	 * @param attrSize
	 * 			the dimension of attributes
	 * @param bInTraining
	 * 			whether the function is run in training or not
	 * @return
	 * @throws Exception
	 */
	public SimpleInstances featureSelectionForRDT(Map<Integer, Double[]>  labelDistVerMap, BufferedWriter bwFS, int[] feaInd, double[] scoreArr, Instances blockInsts, int indexChunk, boolean bDiscretized, int attrSize, boolean bInTraining)
			throws Exception {
		long beginTime = System.currentTimeMillis();
		mRMR mrmr = new mRMR(nfea, selectMethod, discretize);
		MultiInstances datas = new MultiInstances(blockInsts, startIndex,
				endIndex);
		
		mrmr.setDatas(datas);
		long endTime = 0;
		if (bDiscretized && discretize != 9999.0) { // modified on  Feb. 07, 2017
			// 调用MultiInstances中的离散化的方法
			mrmr.getDatas().discretize(discretize, startIndex, endIndex);
			endTime = System.currentTimeMillis();
		    System.out.println("discretization time: "+(endTime-beginTime)+"ms");
		   // added on  May 21, 2017----output the discretized data
		/*    int[] allFeaInd = new int [nfea];
		    mrmr.getDatas().OutputDiscretizedData(filePath, startIndex, endIndex, allFeaInd);
		    //mrmr.toArffFile(mrmr.getDatas().getDataSet(), allFeaInd, filePath, -1);
		    System.out.println("output all discretized data.");
		    return null;*/
		    
		    discretValNum = 3;// added on  May 22, 2017
		}
	
			
		//added on  May 22, 2017---label distributuon statitsitcs for cosine computation
		if ( bInTraining ){
			long tmpStartTime = System.currentTimeMillis();
			mrmr.getDatas().DoLabelDistrStats(startIndex, endIndex);
			//labelDistVerMap = 
			DoLabelDistrStatisticsForCosine(labelDistVerMap, mrmr.getDatas().getLabelDist(),classValNum,blockInsts.size());//discretValNum); modified on  May 27, 2017
			long tmpEndTime = System.currentTimeMillis();
			
			System.out.println(labelDistVerMap.size()+"--The time of DoLabelDistrStats for cosine computation:"+(tmpEndTime-tmpStartTime)+"ms");
		}
	     beginTime = System.currentTimeMillis();
	     //BufferedWriter bw = new BufferedWriter(new FileWriter(filePath+"/featureSelection"+indexChunk+".txt"));
	     int[] tmpFeaInd = mrmr.mRMRSelect(bwFS, scoreArr, indexChunk);
	     double tmpVal = 0;
		 //added on  May 15, 2017---deleted on  May 20, 2017
    	// Set<Integer> keySet = featureSetOfChunks.keySet();
	     for ( int i = 0; i < tmpFeaInd.length; i++ ){
	    	 feaInd[i] = tmpFeaInd[i];
	    	 //added on  May 15, 2017
//	    	 if (i < nfea)
//	    	 {
//		    	 if ( keySet.contains(feaInd[i]) ){
//		    		 tmpVal = featureSetOfChunks.get(feaInd[i]);
//		    		 tmpVal += scoreArr[feaInd[i]];
//		    	 }else{
//		    		 tmpVal = scoreArr[feaInd[i]];
//		    	 }
//		    	 featureSetOfChunks.put(feaInd[i], tmpVal);
//	    	 }
	     }
		 endTime = System.currentTimeMillis();
	     System.out.println("feature select time on instances with size of "+blockInsts.numInstances()+":"+(endTime-beginTime)+"ms");
		// 经过特征选择之后的名字
		String fsFile = mrmr.toArffFile(blockInsts, feaInd, filePath, indexChunk);//---modified on  Feb. 4
		ArffReader rd = new ArffReader();
		System.out.println(filePath+"\\SelectedArff\\"+fsFile);
		rd.setFilePath(filePath+"\\SelectedArff\\"+fsFile);
		// set the attribute size (including the class labels)
		rd.setAttrSize(attrSize);
		SimpleInstances fsBlockInsts = rd.getInstances();
		//bw.close();
		return fsBlockInsts;
	}
	public Map generateMap(int[] feaInd, double[] scoreArr, Map<Integer,Integer> prenonZeroAttrMap) throws Exception{
		Map<Integer, Double> mapA = new HashMap<Integer, Double>();
		//Map.Entry emp = null;
		for( int i = 0; i < feaInd.length; i++){
			//emp = new Map.Entry<feaInd[i], scoreArr[i]> ;
			if ( prenonZeroAttrMap.containsKey(feaInd[i]) )
				mapA.put(new Integer(feaInd[i]), new Double(scoreArr[i]));
//			else{
//				System.out.println("Null values!"+feaInd[i]);
//			}
		}
		return mapA;
	}
	/**
	 * drifting detection given two data chunks and return the drift type, such as "drift, no-drift, noisy impact"
	 * 
	 * @param preFeaIndScrMap
	 * 			 the selected feature set and corresponding score set in the previous data chunk
	 * @param curFeaIndScrMap
	 * 			 the selected feature set and corresponding score set in the current data chunk
	 * @return the drifting type in the evaluation
	 * @throws Exception
	 */
	public String driftDetectionBetTwoChunks(Map preFeaIndScrMap, Map curFeaIndScrMap, String simElvType) throws Exception{
		String driftype = "";
		SimilarityEvaluation simElvFun = new SimilarityEvaluation();
		double sim = 0;
		double intersectCount = 0;
		switch (simElvType){
		case "Jaccard":
			intersectCount = simElvFun.Jaccard(preFeaIndScrMap, curFeaIndScrMap);
			break;
		case "cosine":
			break;
			
			default: 
				break;
		}
		if ( intersectCount > blta*preFeaIndScrMap.size() )//
			driftype = "nondrift";
		else
			driftype = "drift";
		System.out.println("intersectCount:"+intersectCount+"--obj.driftType = "+driftype);
		//System.out.println(sim+":"+driftype);
		return driftype;
	}
	
	/**
	 * drifting detection between the current data chunk and each history data chunk, and return the drift type, such as "drift, no-drift, noisy impact"
	 * modified on  May 04, 2017
	 * @param labelDistMapArr 
				 the label distributions of all histroy data chunks in ensemble model
	 * @param curLabelDistrMap
	 * 			 the label distribution of the current data chunk;
	 * @param labelNum
	 * 			 the number of class labels; 
	 * @return the drift type and the model index in the string set to be used in the prediction
	 * @throws Exception
	 */
	public LinkedList<driftInfor> driftDetectionByLabelDistrByCosine(ArrayList<Map<Integer, Double[]>> labelDistMapArr, Map<Integer, Double[]> curLabelDistrMap, int labelNum) throws Exception{
		String driftype = "drift";
		LinkedList<driftInfor> similarModels = new LinkedList<driftInfor>();
		driftInfor obj = null;
		int nHistoryChunks = labelDistMapArr.size();
		if ( nHistoryChunks < 1 )
			return null;
		int maxSimModelIndex = nHistoryChunks-1;//
		Map<Integer, Double[]> preLabelDistrMap = null;
		SimilarityEvaluation simElvFun = new SimilarityEvaluation();
		double cosineDist = 0;
		double maxCosine = -1;
		for ( int i = nHistoryChunks-1; i >= 0; i-- ){ 
			preLabelDistrMap = labelDistMapArr.get(i);
			cosineDist = 0;
			//get the cosineDist values of all labels
			for ( int j = 0; j < labelNum; j++ ){
				cosineDist += 1-simElvFun.Cosine(preLabelDistrMap.get(j), curLabelDistrMap.get(j));
			}
			cosineDist /= labelNum;
			System.out.println("cosine = "+cosineDist);
			if ( maxCosine < cosineDist ){
				maxCosine = cosineDist;
				maxSimModelIndex = i;
			}
			
			if ( cosineDist < gamma ){// pcc value is lower, indicating the larger data distribution
				driftype = "drift";
			}else{
				driftype = "nondrift";
			}
			if (driftype == "nondrift"){
				//modelIndex = i;
				obj = new driftInfor();
				obj.driftType = driftype;
				obj.modelIndex = i;
				similarModels.add(obj);
				//break;
			}
		}// end for
		// add the most similar model at the last postition
		obj = new driftInfor();
		obj.modelIndex = maxSimModelIndex;
		if ( similarModels.size() == 0 )
			obj.driftType = "drift";
		else
			obj.driftType = "nondrift";
		similarModels.add(obj);
		return similarModels;
		
	}
	
	/**
	 * drifting detection between the current data chunk and each history data chunk, and return the drift type, such as "drift, no-drift, noisy impact"
	 * 
	 * @param labelDistListArr 
				 the label distributions of all histroy data chunks in ensemble model
	 * @param curLabelDistrList
	 * 			 the label distribution of the current data chunk;
	 * @param labelNum
	 * 			 the number of class labels; 
	 * @return the drift type and the model index in the string set to be used in the prediction
	 * @throws Exception
	 */
	public LinkedList<driftInfor> driftDetectionByLabelDistrByHam(ArrayList<ArrayList<int[]>> labelDistListArr, ArrayList<int[]> curLabelDistrList, int labelNum) throws Exception{
		String driftype = "drift";
		LinkedList<driftInfor> similarModels = new LinkedList<driftInfor>();
		driftInfor obj = null;
		int nHistoryChunks = labelDistListArr.size();
		if ( nHistoryChunks < 1 )
			return null;
		int maxSimModelIndex = nHistoryChunks-1;//
		ArrayList<int[]> preLabelDistrList = null;
		SimilarityEvaluation simElvFun = new SimilarityEvaluation();
		double sim = 0;
		double maxSim = -1;
		for ( int i = nHistoryChunks-1; i >= 0; i-- ){ 
			preLabelDistrList = labelDistListArr.get(i);
			
			// get the hamLoss based similarity 
			sim = simElvFun.HammingSim(preLabelDistrList,curLabelDistrList);
			
			System.out.println("sim = "+sim);
			if ( maxSim < sim ){
				maxSim = sim;
				maxSimModelIndex = i;
			}
			if ( sim < gamma ){// pcc value is lower, indicating the larger data distribution
				driftype = "drift";
			}else{
				driftype = "nondrift";
			}
			if (driftype == "nondrift"){
				//modelIndex = i;
				obj = new driftInfor();
				obj.driftType = driftype;
				obj.modelIndex = i;
				similarModels.add(obj);
				//break;
			}
		}// end for
		// add the most similar model at the last postition
		obj = new driftInfor();
		obj.modelIndex = maxSimModelIndex;
		if ( similarModels.size() == 0 )
			obj.driftType = "drift";
		else
			obj.driftType = "nondrift";
		similarModels.add(obj);
		return similarModels;
		
	}
	
	/**
	 * drifting detection between the current data chunk and each history data chunk horizontally and vertically, and return the drift type, such as "drift, no-drift, noisy impact"
	 * 
	 * @param labelDistListArr 
				 the label distributions of all histroy data chunks in ensemble model
	 * @param curLabelDistrList
	 * 			 the label distribution of the current data chunk;
	 * @param labelNum
	 * 			 the number of class labels; 
	 * @return the drift type and the model index in the string set to be used in the prediction
	 * @throws Exception
	 */
	public LinkedList<driftInfor> driftDetectionByLabelDistr(FileWriter testWriter, ArrayList<Map<Integer, Double[]>> labelDistMapArr, Map<Integer, Double[]> curLabelDistrMap,ArrayList<ArrayList<int[]>> labelDistListArr, ArrayList<int[]> curLabelDistrList, int labelNum) throws Exception{
		String driftype = "drift";
		LinkedList<driftInfor> similarModels = new LinkedList<driftInfor>();
		driftInfor obj = null;
		int nHistoryChunks = labelDistListArr.size();
		if ( nHistoryChunks < 1 )
			return null;
		int maxSimModelIndex = nHistoryChunks-1;//
		ArrayList<int[]> preLabelDistrList = null;
		SimilarityEvaluation simElvFun = new SimilarityEvaluation();
		double sim = 0;
		double maxSim = -1;
		double cosineDist = 0;
		Map<Integer, Double[]> preLabelDistrMap = null;
		double ham = 0;
		Integer key = 0;
		//for ( int i = nHistoryChunks-1; i >= 0; i-- ){ 
		for ( int i = 0; i < nHistoryChunks; i++ ){ // modified on  May 23, 2017
			preLabelDistrList = labelDistListArr.get(i);
			preLabelDistrMap = labelDistMapArr.get(i);
			cosineDist = 0;
			
			Iterator<Integer> iter = curLabelDistrMap.keySet().iterator();

			while (iter.hasNext()) { // modified on  May 27, 2017
			    key = iter.next();
			    cosineDist += 1-simElvFun.Cosine(preLabelDistrMap.get(key), curLabelDistrMap.get(key));
			}
//			for ( int j = 0; j < curLabelDistrMap.size(); j++ ){ // modified on  May 11, 2017, deleted May 27, 2017
//				cosineDist += 1-simElvFun.Cosine(preLabelDistrMap.get(j), curLabelDistrMap.get(j));
//			}
			if ( curLabelDistrMap.size() > 0 ){
				cosineDist /= (curLabelDistrMap.size()+0.00000000000000000001);//labelNum;
				// get the hamLoss based similarity 
				ham = simElvFun.HammingLossOverrid(preLabelDistrList,curLabelDistrList);
			}else{
				cosineDist = 0;
				ham = 0;
			}
				
//			System.out.println("cosineDist = "+cosineDist);
//			
//			System.out.println("ham = "+ham);
			
			sim =  1- 2*cosineDist*ham/(cosineDist+ham+0.000000000000000000001);
			// output the similarity between recent two data chunks
			if ( i == nHistoryChunks-1 ){
				if ( ham > 1-gamma ){//added on  May 23, 2017
					testWriter.write("||ham = \t"+ham+"\tdrift"+"\tcosineDist = \t"+cosineDist+"\tsim = \t"+sim+"\t");
					System.out.println("||ham = \t"+ham+"\tdrift"+"\tcosineDist = \t"+cosineDist+"\tsim = \t"+sim+"\t");
				}
				else{
					testWriter.write("||ham = \t"+ham+"\tnondrift"+"\tcosineDist = \t"+cosineDist+"\tsim = \t"+sim+"\t");
					System.out.println("||ham = \t"+ham+"\tnondrift"+"\tcosineDist = \t"+cosineDist+"\tsim = \t"+sim+"\t");
				}
			}
			if ( maxSim < sim ){ 
				maxSim = sim;
				maxSimModelIndex = i;
			}
			
			if ( sim < gamma ){// hybrid distance value is lower, indicating the larger data distribution
				driftype = "drift";
			}else{
				driftype = "nondrift";
			}
			if (driftype == "nondrift"){
				//modelIndex = i;
				obj = new driftInfor();
				obj.driftType = driftype;
				obj.modelIndex = i;
				similarModels.add(obj);
				//break;
			}
		}// end for
		// add the most similar model at the last postition
		obj = new driftInfor();
		obj.modelIndex = maxSimModelIndex;
		
		
		
		if ( similarModels.size() == 0 )
			obj.driftType = "drift";
		else
			obj.driftType = "nondrift";
		similarModels.add(obj);
		return similarModels;
		
	}
	/**
	 * drifting detection between the current data chunk and each history data chunk, and return the drift type, such as "drift, no-drift, noisy impact"
	 * 
	 * @param similarModelsByLabelDistr
	 * 			the selected similar models by label distribution
	 * @param nonZeroAttrMapList 
				 store the non-zero values of attributes, which will be used in the drifting detection
	 * @param feaIndList
	 * 			 the list of history selected feature sets;
	 * @param feaScoreList
	 * 			 the score list corresponding to history selected feature sets; 
	 * @param curFeaInd
	 * 			 the selected feature set in the current data chunk
	 * @param curScoreArr
	 * 			 the score set of the selected feature set in the current data chunk
	 * @return the drift type and the model index in the string set to be used in the prediction
	 * @throws Exception
	 */
	public LinkedList<driftInfor> driftDetectionByFeaDistr(LinkedList<driftInfor> similarModelsByLabelDistr, ArrayList<Map<Integer,Integer>> nonZeroAttrMapList, Map<Integer,Integer> curnonZeroAttrMap, 
			ArrayList<int[]> feaIndList, ArrayList<double[]> feaScoreList, int[] curFeaInd, double[] curScoreArr,String simElvType) throws Exception{
		String driftype = "drift";
		driftInfor obj = null;
		int nHistoryChunks = 0;
		int maxSimModelIndex = 0;
		double maxSim = -1; // modified on  May 22, 2017
		LinkedList<driftInfor> similarModels = new LinkedList<driftInfor>();
		Map curMap = generateMap(curFeaInd, curScoreArr, curnonZeroAttrMap);
		Map preMap = null;
		if ( similarModelsByLabelDistr != null ){
			nHistoryChunks = similarModelsByLabelDistr.size();
			if ( nHistoryChunks < 1 )
				return null;
			maxSimModelIndex = nHistoryChunks-1;//
			Map<Integer,Integer> prenonZeroAttrMap = null;
			driftInfor driftInforObj = null;
			for ( int j = 0; j < nHistoryChunks-1; j++ ){ // according the time stamp, find a model/chunk with similar data distribution
				driftInforObj = similarModelsByLabelDistr.get(j);
				prenonZeroAttrMap = nonZeroAttrMapList.get(driftInforObj.modelIndex);
				preMap = generateMap(feaIndList.get(driftInforObj.modelIndex), feaScoreList.get(driftInforObj.modelIndex), prenonZeroAttrMap);
				SimilarityEvaluation simElvFun = new SimilarityEvaluation();
				double intersectCount = 0;
				switch (simElvType){
				case "Jaccard":
					intersectCount = simElvFun.Jaccard(preMap, curMap);
					break;
				case "cosine":
					break;
				default: 
					break;
				}
				if ( maxSim < intersectCount ){
					maxSim = intersectCount;
					maxSimModelIndex = driftInforObj.modelIndex;
				}
				if ( intersectCount < blta*preMap.size() )//
					driftype = "drift";
				else
					driftype = "nondrift";
				//driftype = driftDetectionBetTwoChunks(preMap, curMap, simElvType);
				if (driftype == "nondrift"){
					//modelIndex = i;
					obj = new driftInfor();
					obj.driftType = driftype;
					obj.modelIndex = driftInforObj.modelIndex;
					similarModels.add(obj);
					//break;
				}
			}// end for
		}
		else{
			
			nHistoryChunks = feaIndList.size();
			if ( nHistoryChunks < 1 )
				return null;
			maxSimModelIndex = nHistoryChunks-1;//
			Map<Integer,Integer> prenonZeroAttrMap = null;
			//for ( int i = nHistoryChunks-1; i >= 0; i-- ){ // according the time stamp, find a model/chunk with similar data distribution
			for ( int i = 0; i < nHistoryChunks; i++ ){ // modified on  May 23, 2017， 
				prenonZeroAttrMap = nonZeroAttrMapList.get(i);
				preMap = generateMap(feaIndList.get(i), feaScoreList.get(i),prenonZeroAttrMap);
				SimilarityEvaluation simElvFun = new SimilarityEvaluation();
				double intersectCount = 0;
				switch (simElvType){
				case "Jaccard":
					intersectCount = simElvFun.Jaccard(preMap, curMap);
					break;
				case "cosine":
					break;
				default: 
					break;
				}
				if ( maxSim < intersectCount ){
					maxSim = intersectCount;
					maxSimModelIndex = i;
				}
				if ( intersectCount < blta*preMap.size() )//
					driftype = "drift";
				else
					driftype = "nondrift";
				
				//driftype = driftDetectionBetTwoChunks(preMap, curMap, simElvType);
				if (driftype == "nondrift"){
					//modelIndex = i;
					obj = new driftInfor();
					obj.driftType = driftype;
					obj.modelIndex = i;
					similarModels.add(obj);
					//break;
				}
			}// end for
		}
//		//
//		if ( similarModels.size() == 0 ){// find no similar model, use the newest model to predict
//			driftype = "drift";
//			obj = new driftInfor();
//			obj.driftType = driftype;
//			obj.modelIndex = maxSimModelIndex;
//			similarModels.add(obj);
//		}
		// add the most similar model at the last postition
		obj = new driftInfor();
		obj.modelIndex = maxSimModelIndex;
		if ( similarModels.size() == 0 )
			obj.driftType = "drift";
		else
			obj.driftType = "nondrift";
		similarModels.add(obj);
		return similarModels;
	}
	
	// cmd: -path F:\E盘\Jesse-Read-Multi-labels\IMDB-ECC-F -train IMDB-ECC-F-train.arff -test IMDB-ECC-F-test.arff -output RDTTest.txt -attrSize 1029 -labelNum 28
	//-path F:\workspace\DiceMultiLabelSystem\data -train scene-train.arff -test scene-test.arff -output RDTTest.txt -attrSize 300 -labelNum 6
	//-path F:\workspace\MultiLabel\data\available\enron -train enron-train.arff -test enron-test.arff -output RDTTest.txt -attrSize 1053 -labelNum 53
	/**
	 * ML-RDT Test using the CBRRDTModel 
	 * 
	 * @param path
	 * 			the path of the source data
	 * @param train
	 * 			 the file of the training data 
	 * @param test
	 * 			 the file of the testing data
	 * @param attrSize
	 * 			the number of attribute dimensions
	 * @param labelNum
	 *   		the number of the class labels
	 * @return 
	 * @throws Exception
	 */
	public void MLRDTTest(String[] args)throws Exception{
			String pathStr = Utils.getOption("path", args); 
			String trainFiles = Utils.getOption("train", args); // e.g. -arff emotions.arff
	        String testFiles = Utils.getOption("test", args); // e.g. -xml emotions.xml
	     
	     // store the experimental results in the training
	     	FileWriter trainWriter = new FileWriter(filePath+"\\traingResult.txt");
			String attrSizeStr = Utils.getOption("attrSize", args);
			int attrSize = Integer.parseInt(attrSizeStr);
			String labelNumStr = Utils.getOption("labelNum", args);
			int labelNum = Integer.parseInt(labelNumStr);
			int maxS = 4;
			int treeNum = 10;
			long beginTime = System.currentTimeMillis();
			ArffReader rd = new ArffReader();
			System.out.println(pathStr+"\\"+trainFiles);
			rd.setFilePath(pathStr+"\\"+trainFiles);
			rd.setAttrSize(attrSize);
			SimpleInstances trainInstances = rd.getInstances();
			TreeBuilder treeBuilder = new TreeBuilder(0, TreeBuilder.CBR_RDT);
			CBRRDTModel model = new CBRRDTModel();
			treeBuilder.setInstances(trainInstances);
			treeBuilder.setMaxDeep(attrSize/2);
			treeBuilder.setMaxS(maxS);
			treeBuilder.setClsSize(labelNum);
			treeBuilder.init();
			Node[] trees = treeBuilder.buildTrees(treeNum);
			// added by peipeili 11-06-13
			System.out.println("training-over");
		    long endTime = System.currentTimeMillis();
		    System.out.println("train-time:"+(endTime-beginTime));
		    trainWriter.write("train-time = "+Integer.toString((int)(endTime-beginTime))+"\n");
			beginTime = System.currentTimeMillis();
			treeBuilder.clear();
			model.init(trees, trainInstances.getAttributes(), 1, maxS);
			rd.setFilePath(pathStr+"\\"+testFiles);
			rd.setAttrSize(attrSize);
			SimpleInstances testInstances =  rd.getInstances();
			EvaluateByMLRDT evluate = new EvaluateByMLRDT();
			
			// store the results relevant to eight evaluation measures
			FileWriter testWriter = new FileWriter(filePath+"\\predictionResult.txt");
			evluate.PredictBySingelModel(model, testInstances, labelNum, attrSize,trainWriter,testWriter);
			model.clear();
			endTime = System.currentTimeMillis();
			System.out.println("test-time:"+(endTime-beginTime));
			trainWriter.write("test-time = "+Integer.toString((int)(endTime-beginTime))+"\n");
	       // System.out.println(result2);
			trainWriter.close();
	}
	// cmd: -path F:\E盘\Jesse-Read-Multi-labels\IMDB-ECC-F -train IMDB-ECC-F-train.arff -test IMDB-ECC-F-test.arff -output RDTTest.txt -attrSize 1029 -labelNum 28
		//-path F:\workspace\DiceMultiLabelSystem\data -train scene-train.arff -test scene-test.arff -output RDTTest.txt -attrSize 300 -labelNum 6
		public static void main1(String[] args) throws Exception{
			
			String pathStr = Utils.getOption("path", args); 
			String trainFiles = Utils.getOption("train", args); // e.g. -arff emotions.arff
	        String testFiles = Utils.getOption("test", args); // e.g. -xml emotions.xml
			//String outputFile = Utils.getOption("output", args);
			//System.out.printf(outputFile);
	     // store the experimental results in the training and detailed prediction results
			FileWriter trainWriter = new FileWriter(pathStr+"\\traingResult.txt");
			// store the results relevant to eight evaluation measures
			FileWriter testWriter = new FileWriter(pathStr+"\\predictionResult.txt");
			String attrSizeStr = Utils.getOption("attrSize", args);
			int attrSize = Integer.parseInt(attrSizeStr);
			String labelNumStr = Utils.getOption("labelNum", args);
			int labelNum = Integer.parseInt(labelNumStr);
			int minS = 4;
			int treeNum = 10;
			long beginTime = System.currentTimeMillis();
			ArffReader rd = new ArffReader();
			System.out.println(pathStr+"\\"+trainFiles);
			rd.setFilePath(pathStr+"\\"+trainFiles);
			rd.setAttrSize(attrSize);
			SimpleInstances trainInstances = rd.getInstances();
			TreeBuilder treeBuilder = new TreeBuilder(0, TreeBuilder.CBR_RDT);
			CBRRDTModel model = new CBRRDTModel();
			treeBuilder.setInstances(trainInstances);
			treeBuilder.setMaxDeep(attrSize/2);
			treeBuilder.setMaxS(minS);
			treeBuilder.setClsSize(labelNum);
			treeBuilder.init();
			Node[] trees = treeBuilder.buildTrees(treeNum);
			// added on  11-06-13
			System.out.println("training-over");
		    long endTime = System.currentTimeMillis();
		    System.out.println("train-time:"+(endTime-beginTime));
		    trainWriter.write("train-time = "+Integer.toString((int)(endTime-beginTime))+"\n");
			beginTime = System.currentTimeMillis();
			treeBuilder.clear();
			model.init(trees, trainInstances.getAttributes(), 1, minS);
			rd.setFilePath(pathStr+"\\"+testFiles);
			rd.setAttrSize(attrSize);
			SimpleInstances testInstances =  rd.getInstances();
			EvaluateByMLRDT evluate = new EvaluateByMLRDT();
			evluate.PredictBySingelModel(model, testInstances, labelNum, attrSize,trainWriter,testWriter);
			model.clear();
			endTime = System.currentTimeMillis();
			System.out.println("test-time:"+(endTime-beginTime));
			trainWriter.write("test-time = "+Integer.toString((int)(endTime-beginTime))+"\n");
	       // System.out.println(result2);
			trainWriter.close();
			testWriter.close();
		}
		
		protected class LabelSeqComparator implements Comparator<AttrLabelSort>{

			@Override
			public int compare(AttrLabelSort o1, AttrLabelSort o2) {
				return o1.getLabelValue().compareTo(o2.getLabelValue());
				/*String datetimeStr1 = o1.date + " "+ o1.time;
				String datetimeStr2 = o2.date + " "+ o2.time;
				SimpleDateFormat sdfd = new SimpleDateFormat("yyyy-MM-dd HH:mm");
				
			    try {
					Date date1 = sdfd.parse(datetimeStr1);
					Date date2 = sdfd.parse(datetimeStr2);
					return date1.compareTo(date2);
				} catch (ParseException e) {
					e.printStackTrace();
				}  */
				
				
				//return 0;
			}

}
}
