# CARTESIAN

CARTESIAN (aCceptance And Response classificaTion-based requESt IdentifcAtioN) is a multi-class pull requests recommendation system that recommends the right actions (accept, respond, and reject) on pull requests which it to be taken by the integrator. CARTESIAN is based on XGBoost classifier trained on the 268k pull requests of 19 popular projects hosted on GitHub. CARTESIAN has produced promising results and we expect that it can help the integrators in the selection of the most likely to be accepted/rejected pull requests.

## Replication Instructions
In the following sub-section, we describe the steps required to replcate our study.

###### How to use the CARTESIAN and run the script
The usage of the underline used python script is the following
```
1. First run the script create_list_for_db.py
2. Then create a database in MySql Community server workbench 
3. Run the script create_tables.py to create all the required tables
4. At the end run the script Import_data_to_db.py to import the data to the database
```
