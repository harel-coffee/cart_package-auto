# CARTESIAN

CARTESIAN (aCceptance And Response classificaTion-based requESt IdentifcAtioN) is a multi-class pull requests recommendation system that recommends the right actions (accept, respond, and reject) on pull requests which it to be taken by the integrator. CARTESIAN is based on XGBoost classifier trained on the 268k pull requests of 19 popular projects hosted on GitHub. CARTESIAN has produced promising results and we expect that it can help the integrators in the selection of the most likely to be accepted/rejected pull requests.

## Replication Instructions
In the following sub-section, we describe the steps required to replicate our study.

###### How to use the CARTESIAN and run the script
The usage of the underline used python script is the following
```
1. The dataset used in this study is available in directory Dataset
2. To trained the classifiers run the script train_models.py. The script will train the four classifiers: XGBoost, Support Vector Machines, Logistic Regression, and Random Forest, and save the trained models in the current directory. 
3. To evaluate the performance of CARTESIAN, run the script cartesian.py. Results will be saved into the Results directory.
```



