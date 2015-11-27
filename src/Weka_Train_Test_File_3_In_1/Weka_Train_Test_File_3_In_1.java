package Weka_Train_Test_File_3_In_1;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Weka_Train_Test_File_3_In_1 {

    public static void main(String[] args) throws Exception{
        // Chia file theo ty le 70,30                
        // Lay file can chia
        BufferedReader reader = new BufferedReader( new FileReader("data/credit-g.arff"));
        Instances dataorg = new Instances(reader);
        reader.close();

        // Chia 70, 30 lan dau
        int percent=70;
        int trainSize = (int) Math.round(dataorg.numInstances() * percent/100);
        int testSize = dataorg.numInstances() - trainSize;
        Instances train = new Instances(dataorg, 0, trainSize);
        Instances test = new Instances(dataorg, trainSize, testSize);

        // Chia 70, 30 lan hai tren tap train
        percent=70;
        int trainSize70 = (int) Math.round(train.numInstances() * percent/100);
        int testSize30 = train.numInstances() - trainSize70;
        Instances train70 = new Instances(train, 0, trainSize70);
        Instances test30 = new Instances(train, trainSize70, testSize30);

        // classifier
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);
        train70.setClassIndex(train.numAttributes() - 1);
        test30.setClassIndex(test.numAttributes() - 1);

        // Bai 2 Tree 2 Train , Test
        System.out.println("-------- Bai 2: Tree J48 --------");
        // TODO Auto-generated method stub
        
        int folds=3;
        String[] options = new String[1];
        options[0] = "-U";            // unpruned tree
        Classifier tree = new J48();         // new instance of tree
        tree.setOptions(options);     // set the options
        
        Evaluation evaltree = new Evaluation(train);
        evaltree.crossValidateModel(tree, train, folds, new Random(1), args);

        tree.buildClassifier(train);  // build classifier
        Evaluation evalb2 = new Evaluation(train);
        evalb2.evaluateModel(tree, test);
        
        double maxUAC_tree=0;
        int bestk_tree=0;
        for (int i = 0; i < test.numInstances(); i++) {
            if(evalb2.weightedAreaUnderROC()>maxUAC_tree){ 
               maxUAC_tree=evalb2.weightedAreaUnderROC(); 
               bestk_tree=i;
               }
        }                        
// Bai Tap 3 KNN		
// k là số láng giềng 
            System.out.println("-------- Bai 3: KNN --------");
            double maxUAC_KNN=0;
            int bestk_KNN=0;
            int d= (int) Math.round(test30.numInstances()/20);
            
            for(int k=1; k<=test.numInstances();k=k+d){
            Classifier ibktrain = new IBk(k);         // new instance of tree
            Evaluation evalibktrain = new Evaluation(train);
            evalibktrain.crossValidateModel(ibktrain, train, folds, new Random(1), args);            
            
            if(evalibktrain.weightedAreaUnderROC()>maxUAC_KNN){ 
                maxUAC_KNN=evalibktrain.weightedAreaUnderROC(); 
                bestk_KNN=k;
                }
            }   
            
            IBk KNN= new IBk(bestk_KNN);
            KNN.buildClassifier(train);  // build classifier
            
            Evaluation evalibktest = new Evaluation(dataorg);
            evalibktest.evaluateModel(KNN, test);

// Bai Tap 4 SVM		
            System.out.println("-------- Bai 4: SVM --------");
            int typeOfKernelFunction = 2;
            double cNumber = 1;
            int bestTypeOfKernelFunction = 2;
            double bestCNumber = 1;
            double maxUAC_SVM=0;
            int seed=1;  
	// cnumber from 1->6
            for (int i = 0; i < 10; i++) {
		cNumber += i;
            // kernel type from 0-3
            for (int j = 2; j <= 3; j++) {
		typeOfKernelFunction = j;
		seed = i * j;
		Evaluation eval = new Evaluation(train);
                // option for SVM
                String[] svmoptions = new String[6];
                svmoptions[0] = "-K";
                svmoptions[1] = String.valueOf(typeOfKernelFunction);
                svmoptions[2] = "-C";
                svmoptions[3] = String.valueOf(cNumber);
                svmoptions[4] = "-H";
                svmoptions[5] = "0";
		LibSVM svm = new LibSVM();
        	svm.setOptions(svmoptions);
		eval.crossValidateModel(svm, train, folds, new Random(seed));
		System.out.println(eval.toClassDetailsString("=== Class detail ==="));
		if (maxUAC_SVM < eval.weightedAreaUnderROC()) {
                    maxUAC_SVM = eval.weightedAreaUnderROC();
                    bestTypeOfKernelFunction = typeOfKernelFunction;
                    bestCNumber = cNumber;
		}
            }
            }
		// Test with best value for SVM
		// option for SVM
		maxUAC_SVM=0;
		String[] optionssvm = new String[6];
		optionssvm[0] = "-K";
		optionssvm[1] = String.valueOf(bestTypeOfKernelFunction);
		optionssvm[2] = "-C";
		optionssvm[3] = String.valueOf(bestCNumber);
		optionssvm[4] = "-H";
		optionssvm[5] = "0";

		LibSVM bestSVM = new LibSVM();
		bestSVM.setOptions(options);
		bestSVM.buildClassifier(train);
		Evaluation evalSVM = new Evaluation(train);
		evalSVM.evaluateModel(bestSVM, train);

		System.out.println("==============SVM===============");
		System.out.println("Found kernel: " + bestTypeOfKernelFunction + " C: "
				+ bestCNumber);
		System.out
				.println(evalSVM.toClassDetailsString("=== Class detail ==="));
		System.out.println(evalSVM.toSummaryString("=== Summary ===", false));
		System.out.println("==============KNN================");
		System.out.println("Found best k: " + bestk);
		System.out
				.println(evalKNN.toClassDetailsString("=== Class detail ==="));
		System.out.println(evalKNN.toSummaryString("=== Summary ===", false));
		System.out.println("==============DT================");
		System.out.println(evalDT.toClassDetailsString("=== Class detail ==="));
		System.out.println(evalDT.toSummaryString("=== Summary ===", false));
            }                      
  }


   
    

