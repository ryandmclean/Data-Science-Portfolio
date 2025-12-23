Welcome to an analytics engineering project showcasing the data pipeline workflow. This project uses Python to generate gameplay for a variety of users, then uses DuckDB and DBT to transform those events into a data warehouse. 

- The product is a game that has 100 levels. The user will move linearly through the 100 levels, only moving on if they complete and pass the level. The simulation is set up to start all students on January 1, 2026 and simulate all 100 levels. 
    - Where attrition analyses are desired, a certain cutoff could be created such as 1 month or 6 months. 
- There are configurations that can be used to create different samples. Generally, the control group has a 80% probability of completing a level and an 80% probability of passing a level if they complete it.  

### Understanding the structure

 - src/simulate_events.py is a script that is used to generate gamplay for a certain number of students based on some tuning parameters.
 - src/run_pipeline.py is a script that imports the events.csv created by the simulate_events.py and then creates a variety of tables needed for analyses
 - learning_app_warehouse: The folder containing the DBT models, split into dl, dw, and dm. 
    - dl: Data Lake, containing raw data like the event stream
    - dw: Data Warehouse, containing dimension tables and intermediate fact tables
    - dm: Data Mart, containing report/analysis ready models. 
- data: Contains the CSVs used/created
    - raw: contains the events.csv
    - exports: contains CSV files for each table created via DBT

Setup and Use:
- Create a virtual environment and install the packages based on the requirements.txt
    - `python -m venv .venv`
    - `source .venv/bin/activate`
    - `pip install -r requirements.txt`
- Run `python src/simulate_events.py` to create whatever additional events you are looking for
- Run `python src/run_pipeline.py` to create the data warehouse and export it into the CSV files
