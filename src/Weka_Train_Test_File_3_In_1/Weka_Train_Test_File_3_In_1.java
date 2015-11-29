package Weka_Train_Test_File_3_In_1;
import java.io.BufferedReader;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IBk; // KNN
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;

public class Weka_Train_Test_File_3_In_1 {

    public static void main(String[] args) throws Exception{
        // Chia file theo ty le 70,30                
        // Lay file can chia
        BufferedReader reader = new BufferedReader( new FileReader("data/credit-g.arff"));
        Instances dataorg = new Instances(reader);
        reader.close();

        // Giu bien trung binh
        double sumUAC_DTree=0.0;
        double sumUAC_KNN=0.0;
        double sumUAC_libSVM=0.0;
        // Chia lam 3 doan
        int percent=3;
        for (int dem=0; dem<3; dem++){
        Instances train = dataorg.trainCV(percent, dem);
        Instances test = dataorg.testCV(percent, dem);

        // classifier
        dataorg.setClassIndex(dataorg.numAttributes() - 1);
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);

        // Bai 2 Tree 2 Train , Test
        System.out.println("-------- Bai 2: Tree J48 --------");
        // TODO Auto-generated method stub
        
        int folds=3;
        String[] optionstree = new String[2];
        optionstree[0] = "-U";            // unpruned tree
        optionstree[1] = "-i";
        Classifier tree = new J48();         // new instance of tree
        tree.setOptions(optionstree);     // set the options
        
        Evaluation evaltreetemp = new Evaluation(train);
        evaltreetemp.crossValidateModel(tree, train, folds, new Random(1), args);

        tree.buildClassifier(train);  // build classifier
        Evaluation evaltreeJ48 = new Evaluation(train);
        evaltreeJ48.evaluateModel(tree, test);
        
        System.out.println("==============DT================");
	System.out.println(evaltreeJ48.toClassDetailsString("=== Class detail ==="));
	System.out.println(evaltreeJ48.toSummaryString("=== Summary ===", false));        
        sumUAC_DTree=evaltreeJ48.weightedAreaUnderROC()+sumUAC_DTree;
// Bai Tap 3 KNN		
// k là số láng giềng 
        System.out.println("-------- Bai 3: KNN --------");
        double maxUAC_KNN=0;
        int bestk_KNN=0;
        int d= (int) Math.round(test.numInstances()/20);

        for(int k=1; k<=d;k++){
        Classifier ibktrain = new IBk(k);         // new instance of tree
        Evaluation evalibktrain = new Evaluation(train);
        evalibktrain.crossValidateModel(ibktrain, train, folds, new Random(1), args);            
            
        if(evalibktrain.weightedAreaUnderROC()>maxUAC_KNN){ 
            maxUAC_KNN=evalibktrain.weightedAreaUnderROC(); 
            bestk_KNN=k;
            }
        }
        
        System.out.println(bestk_KNN);
        Evaluation evalibktest = new Evaluation(dataorg);
        IBk KNN= new IBk(bestk_KNN);
        KNN.buildClassifier(train);  // build classifier                      
        evalibktest.evaluateModel(KNN, test);
        sumUAC_KNN=sumUAC_KNN + evalibktest.weightedAreaUnderROC();
        
        System.out.println("Found best k: " + bestk_KNN);
        System.out.println(evalibktest.toClassDetailsString("=== Class detail ==="));
        System.out.println(evalibktest.toSummaryString("=== Summary ===", false));

// Bai Tap 4 SVM		
        System.out.println("-------- Bai 4: SVM --------");
        int typeOfKernelFunction = 2;
        double cNumber = 0.2;
        int bestTypeOfKernelFunction = 2;
        double bestCNumber = 0.2;
        double maxUAC_SVM=0;
        int seed=1;  
    // cnumber from 1->6
        for (int i = 0; i < 10; i++) {
            cNumber = cNumber++;
        // kernel type from 0-3
        for (int j = 2; j <= 3; j++) {
            // Khoa lenh in cua libSVM
            System.setOut(new PrintStream(new OutputStream() {
                @Override public void write(int b) throws IOException {}
            }));

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
            if (maxUAC_SVM < eval.weightedAreaUnderROC()) {
                maxUAC_SVM = eval.weightedAreaUnderROC();
                bestTypeOfKernelFunction = typeOfKernelFunction;
                bestCNumber = cNumber;
            }              
        }
        }
		// Test with best value for SVM
		// option for SVM  
//            System.setout(new PrintStream());
            maxUAC_SVM=0;
            String[] optionssvm = new String[6];
            optionssvm[0] = "-K";
            optionssvm[1] = String.valueOf(bestTypeOfKernelFunction);
            optionssvm[2] = "-C";
            optionssvm[3] = String.valueOf(bestCNumber);
            optionssvm[4] = "-H";
            optionssvm[5] = "0";

            LibSVM bestSVM = new LibSVM();
            bestSVM.setOptions(optionssvm);
            bestSVM.buildClassifier(train);
            Evaluation evalSVM = new Evaluation(dataorg);                
            evalSVM.evaluateModel(bestSVM, test);
            sumUAC_libSVM=sumUAC_libSVM + evalSVM.weightedAreaUnderROC();
            
            // Cho phep in cac ket qua nguoi dung can    
            System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out)));
            
            System.out.println("Found kernel: " + bestTypeOfKernelFunction + " C: "	+ bestCNumber);
            System.out.println(evalSVM.toClassDetailsString("=== Class detail ==="));
            System.out.println(evalSVM.toSummaryString("=== Summary ===", false));
            }   
        
            System.out.println("UAC_DTree Trung binh " + String.valueOf((double)Math.round((sumUAC_DTree/3)*1000)/1000));
            System.out.println("UAC_KNN Trung binh " + String.valueOf((double)Math.round((sumUAC_KNN/3)*1000)/1000));
            System.out.println("UAC_libSVM Trung binh " + String.valueOf((double)Math.round((sumUAC_libSVM/3)*1000)/1000));
  }
}

   
    

