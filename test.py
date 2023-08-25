""""
The testing generates a 

inference.csv delimited with ','  and a '\n' after reach row with headers  

run_id,source_dataset,source_domain,target_domain,best/epoch,mode,*TRAINING_HYPERPARAMETERS,*METRICS 

Where:

-run_id: WANDB model name
-source_dataset: the name of the context (e.g, NEXUS,BOOKSTORE)
-source_domain: video used for training
-target_domain: video used for testing 
-best/epoch: epoch of the model used for testing
-mode: one of the following modality {transformers-pos,
                                      transfomers-imagePos,
                                      transformers-image,
                                      kalmanFilter} 

-*TRAINING_HYPERPARAMETERS: all the relevant HYPERP_PARAMETERS (e.g. {epochs,training_size,early_stopping})
-*METRICS: all the relevant metrics (e.g., {mse,ade,precison,recall,fitness})


Output example, 

run_id,source_dataset,source_domain,target_domain,best/epoch,mode,epochs,training_size,early_stopping,mse,ade,precison,recall,fitness
2fbrladm,NEXUS,cam1,cam2,98,transformers-pos,100,2000000,TRUE,0.5,0.7,0.2,0.1,0.4
if388dge,NEXUS,cam2,cam2,99,transformers-imagePos,100,2400000,TRUE,0.4,0.6,0.15,0.12,0.38

"""


