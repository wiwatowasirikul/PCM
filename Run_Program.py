###############################################################################
## PyPCM Version 1.0.0, October 2014.                                        ##
##                                                                           ## 
## Wiwat Owasirikul                                                          ##
##     Department of Radiological Technology, MedTech, Mahidol               ##
##Chanin Nantasenamat                                                        ##
##      Center of Data mining and bioimformatic, MedTech, Mahidol            ##
##Likit                                                                      ##
###############################################################################

def ModuleMain():
    import sys
    if len(sys.argv) > 1:
        Rawfile = sys.argv[1]       #Determined your original file name
        Indicator = sys.argv[2]     #User defined name for saving
        Ligand_index = sys.argv[3]  #Ligand index [0-13] (see Descriptors_Extraction.py for more informs)
        Protein_index = sys.argv[4]  #Protein index [0-11] (see Descriptors_Extraction.py for more informs)
        Model_index = sys.argv[5]   #User defined PCM modeling 
        Predictor = sys.argv[6]        #PLS, RF(random forest), SVM
        CV_Method = sys.argv[7]     #k-folds, LOO
        FeatureSelectionMode = sys.argv[8] #No, Yes
        SpiltCriteria = float(sys.argv[9]) # 0.15 by default
        Iteration = int(sys.argv[10])       #20  by default
        NumPermute = int(sys.argv[11])      # 100  by default
    
    
    import Descriptors_Extraction as Ex
    import PCM_workflow as pcm

    userdefined = Ex.UserDefined(Rawfile,Indicator,Ligand_index,Protein_index, Model_index, 
                                 Predictor, SpiltCriteria,CV_Method,FeatureSelectionMode, Iteration, 
                                 NumPermute)
                            
    Ex.AnalysisInputfile(userdefined)
    X, Y, H, harray, NumDes = pcm.Model_Selection(userdefined)
    ind_ext = pcm.Index_Train_Ext(X, userdefined)
    Mean, SD, Ykeep = pcm.Prediction(X,Y,ind_ext, userdefined)
    Q2_intercept, Scamb = pcm.Yscrambling(X,Y, userdefined)
    pcm.Combine_array(NumDes, harray, Mean, SD, Ykeep, Q2_intercept, Scamb, userdefined)
    
if __name__ == '__main__':
    import time 
    start_time = time.time()
    ModuleMain()
    elaped_time = time.time()-start_time
    print elaped_time

   ##python Run_Program.py ExampleData Result [0] [0] [0] PLS 10-folds No 0.15 20 100     