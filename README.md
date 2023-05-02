
In this lab, you’ll be learning the end-to-end machine learning process that involves data preparation, training a model locally and in the cloud, and debugging it to perform responsibly.  We’ll be using the [UCI hospital diabetes dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00296/) to train a classification model using the Scikit-Learn framework.  The model will predict whether or not a diabetic patient will be readmitted back to a hospital within 30 days of being discharged.

# Exercise 1:  Training a model locally

## Prerequisites
1. Clone the lab repository into your local machine.
```bash
git clone https://github.com/ruyakubu/BUILD-AzureML-workshop.git
```
2. Change directory into the lab folder.
```bash
cd BUILD-AzureML-workshop
```
3. Open the lab folder in Visual Studio Code.
4. Then open the *1-dataprep-and-training-local.ipynb* notebook.
5. Click on the **Run All** button on the top of the notebook to run the notebook.


We'll start by cleaning our dataset: we'll check for special characters, remove missing values, delete irrelevant columns, and reformat our data. After the data has been cleansed, we’ll train the model. 

## Task 1: Data Preparation

For this exercise, you’ll need to load the **1-dataprep-and-training-local.ipynb** jupyter notebook.

Let's first read the hospital diabetes data into a DataFrame, using the "read_csv" function from the “pandas” library.

```python
df = pd.read_csv('data/diabetic_data.csv')
```

*Understanding the dataset*

Next we'll check the size of the data to verify that it will be sufficient for our training. 

```python
df.shape
```

101,766 is a very large dataset, which is good.

Let’s display the DataFrame to see what kind of features are in the diabetes patients’ hospital data. 

```python
df.head()
```

As you can see it contains the patient’s race, gender, age, weight, their prior hospital visits, lab results, and so on - these will be our model's inputs.  It also contains information on whether or not they were readmitted back to the hospital - this will be our model's output. 

*Checking for special characters*

If you look at the *weight* column in the portion of the DataFrame you printed, you'll notice that some of its values contain "?". Let's change that character to null.

```python
df = df.replace('?', np.NaN) 
df.head() 
```

*Removing null values*

Running the *count* function on the dataframe tells us how many values in each column are not null. 

```python
df.count()
```

As you can see, we have null values in most columns, and *weight* in particular has a lot of null values. Even thought the patient's weight is likely to impact the readmission status, we've decided to drop it because it contains such little information.

```python
df = df.drop(['weight'], axis=1)
```

We'll also drop all rows that contain some null values:

```python
df = df.dropna()
```

*Deleting irrelevant columns*

There are 20+ columns that are not relevant to whether the patient is readmitted to the hospital, so we'll drop those.

```python
df.drop(['encounter_id', 'patient_nbr', 'payer_code', 'medical_specialty', 'admission_type_id', 'repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton', 'metformin','glyburide-metformin','glipizide-metformin' 'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone', 'diag_2', 'diag_3', 'change'], axis=1, inplace=True)
```

*Formatting data*

To make it easier to work with this data later in this workshop, we'll rename some columns to make their names more readable, and we'll consolidate some categories.

For example, in the code snippet below, we replace the age groups with better descriptions. And because some age groups have very few patients, we'll also combine them into larger age groups:

```python
df.loc[:, "age"] = df["age"].replace( ["[0-10)", "[10-20)", "[20-30)"], "30 years or younger")
df.loc[:, "age"] = df["age"].replace(["[30-40)", "[40-50)", "[50-60)"], "30-60 years")
df.loc[:, "age"] = df["age"].replace(["[60-70)", "[70-80)", "[80-90)", "[90-100)"], "Over 60 years")
```

We'll do similar renaming and aggregating operations for other columns. It's not critical to understand the code in this section in order to understand the rest of the workshop, so let's move on to training the dataset.


## Task 2:  Training data

Now that the data has been cleansed, we can use it to train our diabetes hospital readmission classification model. We'll start by reducing the number of rows in the DataFrame to make training a bit faster for the workshop.

```python
df = df.sample(frac=0.20)
```

Then we'll split the training and testing datasets with scikit-learn’s *train_test_spilt* function. 

```python
train, test = train_test_split(df, train_size=0.80, random_state=1)
```

Next we'll format both datasets to a parquet file.  The data files will be store on the local machine to be used later.

```python
train.to_parquet('data/training_data.parquet')
test.to_parquet('data/testing_data.parquet')
```

Next we'll split the train and test data into features X (our inputs), and targets Y (our labels). 

```python
# Split train and test data into features X and targets Y.
target_column_name = 'readmit_status'
Y_train = train[target_column_name]
X_train = train.drop([target_column_name], axis = 1)  
Y_test = test[target_column_name]
X_test = test.drop([target_column_name], axis = 1)  
```

Then we'll transform string data to numeric values using scikit-learn’s *OneHotEncoder*, and we'll standardize numeric data using scikit-learn’s *StandardScalar*.  After that, we'll create a pipeline with these two processing steps, and the LogisticRegression classification model.  And finally, we'll train the model using the *fit* function, and we'll score it.

```python
# Transform string data to numeric one-hot vectors
categorical_selector = selector(dtype_exclude=np.number)
categorical_columns = categorical_selector(X_train)
categorial_encoder = OneHotEncoder(handle_unknown="ignore")

# Standardize numeric data by removing the mean and scaling to unit variance
numerical_selector = selector(dtype_include=np.number)
numerical_columns = numerical_selector(X_train)
numerical_encoder = StandardScaler()

# Create a preprocessor that will preprocess both numeric and categorical data
preprocessor = ColumnTransformer([
('categorical-encoder', categorial_encoder, categorical_columns),
('standard_scaler', numerical_encoder, numerical_columns)])

clf = make_pipeline(preprocessor, LogisticRegression())

print("Training model...") 
model = clf.fit(X_train, Y_train)
print("Accuracy score: ", clf.score(X_test,Y_test))
```

Well done...a score of around 0.85 looks good!


# Exercise 2:  Training a model in the cloud

In the previous exercise you trained your model locally, and in this exercise you'll train it in the cloud.  Training in the cloud brings many advantages: you can easily track model versions, you can scale your training to use more compute power, and you can deploy it for others to use.

## Prerequisites
1. Open the Azure Machine Learning studio at https://ml.azure.com
2. Then open the *2-compute-training-job-cloud.ipynb* notebook.
3. Click on the **Run All** button on the top of the notebook to run the notebook.

This notebook takes about 20 minutes to run, so it may not be done running by the time you finish going through the material. If that's the case, move on to the third notebook, and come back at the end to see the results.

## Task 1: Create a cloud client

Before you can use the Azure Machine Learning studio, you need to create a cloud client session to authenticate and connect to the workspace.  The authorization needs the subscription id, resource group, and name of the Azure ML workspace, which it gets from the "config.json" file in this repo.

```json
{
    "subscription_id": "...",
    "resource_group": "...",
    "workspace_name": "..."
}
```

TODO: Need to ask Skillable to create a config.json file with the authentication details.

With that file in place, we can authenticate with the following code:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)
```

## Task 2: Register the training data

Next we'll register the training data we saved ealier with Azure Machine Learning. 

```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

DATA_NAME = 'hospital_train_parquet'

training_data = Data(
    name=DATA_NAME,
    path='data/training_data.parquet',
    type=AssetTypes.URI_FILE,
    description='RAI hospital train data'
)
tr_data = ml_client.data.create_or_update(training_data)
```

This command refers to the parquet training data we saved to disk in notebook 1, copies the data to the cloud, and registers it under the name "hospital_train_parquet." We'll use that name to refer to our data later.

You can verify the data is registered by opening the Azure ML studio at https://ml.azure.com, clicking on "Data," and finding the entry with the name we specified.

## Task 3: Create a compute cluster

Next we'll create a compute cluster that contains the details of the virtual machines we'll use to train our model. We'll specify a machine size, a minimum and maximum number of instances in the cluster, and the maximum number of seconds that a machine can be idle before we release for others to use.

```python
from azure.ai.ml.entities import AmlCompute

my_compute = AmlCompute(
    name="trainingcompute",
    size="Standard_DS2_v2",
    min_instances=0,
    max_instances=4,
    idle_time_before_scale_down=3600
)
ml_client.compute.begin_create_or_update(my_compute)
```

You can verify the compute cluster was created in the Studio, by going to "Compute," and then "Compute clusters."

## Task 4: Create the job

The next step is to run the code that trains our model using our hospital data, in the cloud. We'll create a job for that purpose.

We need to specify the following information to create this job:
* Description: This will show up in the Studio later, together with all the detailed information for this job.
* Experiment name: This is the name you'll see listed in the Studio, as soon as the job execution is under way.
* Compute: The compute cluster you created earlier.
* Inputs: Specifies the inputs we want to pass to our training script in the "Command" section. In our scenario, we want to pass in the training data, and the name of the target column.
* Outputs: Specifies the output returned by the training script, which in our scenario is just the trained model.
* Code: The directory where the training script is located.
* Environment: An environment that specifies all the software dependencies the job needs to run. In our case, we're using a curated environment from Responsible AI, since we’ll need it in the next lab.
* Command: The actual command used to run our training script. 

```python
from azure.ai.ml import command, Input, Output
from azure.ai.ml.entities import Model

TARGET_COLUMN_NAME = 'readmit_status'

# Create the job.
job = command(
    description='Trains hospital readmission model',
    experiment_name='hospital_readmission',
    compute=COMPUTE_NAME,
    inputs=dict(training_data=Input(type='uri_file', path=f'{DATA_NAME}@latest'), 
                target_column_name=TARGET_COLUMN_NAME),
    outputs=dict(model_output=Output(type=AssetTypes.MLFLOW_MODEL)),
    code='src/',
    environment='azureml://registries/azureml/environments/AzureML-responsibleai-0.20-ubuntu20.04-py38-cpu/versions/4',
    command='python train.py ' + 
            '--training_data ${{inputs.training_data}} ' +
            '--target_column_name ${{inputs.target_column_name}} ' +
            '--model_output ${{outputs.model_output}}'
)
job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(job.name)
```

You can take a look at the "src/train.py" file specified in the command, if you'd like. It contains the training code you're already familiar with, a bit of code to deal with the arguments, and a couple of lines of code to save the model usingn the MLFlow package. 

The job will take several minutes to run. You can follow the progress in the Studio by clicking on "Jobs," and then looking for the experiment name specified in the code.


## Task 5: Register the model

When the job finishes running, it outputs a trained model. We want to register that model, so that we can invoke it later to make predictions. Here's the code we need to register the model:

```python
from azure.ai.ml.entities import Model

MODEL_NAME = 'hospital_readmission_model'

# Register the model.
model_path = f"azureml://jobs/{job.name}/outputs/model_output"
model = Model(name=MODEL_NAME,
                path=model_path,
                type=AssetTypes.MLFLOW_MODEL)
registered_model = ml_client.models.create_or_update(model)
```

You can check that the model is registered by looking for the model name in the Studio, under "Models."


## Task 6: Deploy the model

Next we're going to create an endpoint that we can use to make predictions using our trained model. Endpoints can have multiple deployments, and direct a percentage of their traffic to each deployment. We're going to keep it simple in this scenario, by creating a single deployment that takes all the traffic.

```python
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint 

ENDPOINT_NAME = 'hospital-readmission-endpoint'
DEPLOYMENT_NAME = 'blue'

# Create the managed online endpoint.
endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    auth_mode='key',
)
registered_endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint)

# Get the latest version of the registered model.
registered_model = ml_client.models.get(name=MODEL_NAME, label='latest')

# Create the managed online deployment.
deployment = ManagedOnlineDeployment(name=DEPLOYMENT_NAME,
                                        endpoint_name=ENDPOINT_NAME,
                                        model=registered_model,
                                        instance_type='Standard_DS4_v2',
                                        instance_count=1)
ml_client.online_deployments.begin_create_or_update(deployment)

# Set deployment traffic to 100%.
registered_endpoint.traffic = {DEPLOYMENT_NAME: 100}
ml_client.online_endpoints.begin_create_or_update(
    registered_endpoint)
```

This takes several minutes to run. You can verify that your endpoint was created by going to the Studio, clicking on "Endpoints", and looking for the endpoint name on tht list. 


## Task 7: Invoke the endpoint

Once the endpoint is created, you can invoke it. In this case, we're going to invoke it using the input data in the file "test_data.json." You should get a prediction of "not readmitted" for this data.

```python
TEST_DATA_PATH="test_data.json"

# Invoke the endpoint.
result = ml_client.online_endpoints.invoke(endpoint_name=ENDPOINT_NAME, request_file=TEST_DATA_PATH)
print(result)
```

Well done! You trained a model in the cloud, and you created an endpoint that you (or anyone!) can use to make predictions! :)


# Exercise 3:  Add a Responsible AI dashboard

In this exercise, you’ll be learning how to create the Responsible AI dashboard for the trained model. This will help us get a holistic insight from our diabetes hospital readmission model. In addition, it will expose any errors or areas where there could be undesirable responsible AI issues. The dashboard is comprised of different components to enable you to debug and analyze the model performance, identify errors, conduct fairness assessment, evaluate performance model interpretability and more. The dashboard can be created using the Azure Portal, CLI or SDK. In this exercise, we will be using the SDK to create a dashboard.

## Prerequisites
1. Open the Azure Machine Learning studio at https://ml.azure.com
2. Click on *Notebooks* on the left navigation menu.
3. Under your username, click on *Upload folder* option and upload the *BUILD-AzureML-workshop* directory that your clone in the last exercise. (**ONLY**:  If you have not already uploaded the directory)
4. Then open the *3-create-responsibleai-dashboard.ipynb* notebook.

## Task 1: Define the dashboard components

The Responsible AI dashboard components are already pre-defined in the Azure Machine Learning studio. To use the components, you need to submit the component name and version to the Azure Machine Learning client’s session created in the previous exercise. The user has the option to add as many components they want on the Responsible AI dashboard. The components you’ll be using are:

* Error Analysis
* Explanation
* Insight Gather

``` python
label = "latest"

rai_constructor_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_insight_constructor", label=label
)

# We get latest version and use the same version for all components
version = rai_constructor_component.version

rai_explanation_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_explanation", version=version
)

rai_erroranalysis_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_erroranalysis", version=version
)

rai_gather_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_insight_gather", version=version
)
```

## Task 2: Create the job to create the dashboard

When you have specified the RAI components you need, it is time to define an [Azure pipeline](https://aka.ms/MBAzureMLPipeline) and config each RAI component. For the list of settings needed to configure each of your RAI components, refer to [RAI component parameters](https://aka.ms/MBARAIComponentSettings).

To define the pipeline for the RAI Dashboard, declare the experiment name, description and the name of the compute server that will be running the pipeline job. We use the [dsl.pipeline](https://docs.microsoft.com/python/api/azure-ai-ml/azure.ai.ml.dsl?view=azure-python-preview%22%20%5Ct%20%22_blank) annotation above the pipeline function to specify these fields.

The RAI constructor component is what initializes the global data needed for the different components for the RAI dashboard. The constructor component takes the following parameters:

* *title* - The title of the dashboard.
* *task_type* - Our Diabetes Hospital Readmission is a classification use-case, so we’ll set value to classification.
* *model_info* -  the model output path 
* *model_input* -  the MLFlow model input path
* *train_dataset* - the registered training dataset location 
* *test_dataset* - the registered testing dataset location 
* *target_column_name* - the target column our model is trying to predict
* *categorical_column_names* - the columns in our dataset that have non-numeric values

``` python
        # Initiate the RAIInsights
        create_rai_job = rai_constructor_component(
            title="RAI Dashboard",
            task_type="classification",
            model_info=expected_model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),            
            train_dataset=training_data,
            test_dataset=testing_data,
            target_column_name=target_column_name,
            categorical_column_names=json.dumps(categorical),
        )
        create_rai_job.set_limits(timeout=120)
``` 

The Explanation component is responsible for the dashboard providing a better understanding of what features influence the model’s predictions. It takes a comment that is a description pertaining to your use case. Then sets the *rai_insights_dashboard* to be the output insights generated from the RAI pipeline job for Explanations.

``` python
        # Explanation
        explanation_job = rai_explanation_component(
            comment="Explain the model",
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        )
        explanation_job.set_limits(timeout=120)
```

The Error Analysis component is responsible for the dashboard providing an error distribution of the feature groups contributing to the error rate of the model. Its only configuration is to set the *rai_insights_dashboard* to be the output insights generated from the RAI pipeline job for the overall and feature error rates.

``` python
        # Error Analysis
        erroranalysis_job = rai_erroranalysis_component(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        )
        erroranalysis_job.set_limits(timeout=120)
```

Once all the RAI components are configured with the parameters needed for the use case, the next thing to do is add all of them into the list of insights to include on the RAI dashboard. Then upload the dashboard and UX settings for the RAI Dashboard.

``` python
        rai_gather_job = rai_gather_component(
            constructor=create_rai_job.outputs.rai_insights_dashboard,
            insight_1=explain_job.outputs.explanation,
            insight_4=erroranalysis_job.outputs.error_analysis,
        )
        rai_gather_job.set_limits(timeout=120)
```

The pipeline job outputs are the dashboard and UX config to be displayed.

## Task 2: Run job to create the dashboard

After the pipeline is defined, we'll initialize it by specifying the input parameters and the path to the outputs. Lastly, we use the submit_and_wait function to run the pipeline and register it to the Azure Machine Learning studio.

``` python
# Pipeline Inputs
insights_pipeline_job = rai_classification_pipeline(
    target_column_name=target_column,
    training_data=hospital_train_parquet,
    testing_data=hospital_test_parquet,
)

# Output workaround to enable the download
rand_path = str(uuid.uuid4())
insights_pipeline_job.outputs.dashboard = Output(
    path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/dashboard/",
    mode="upload",
    type="uri_folder",
)
insights_pipeline_job.outputs.ux_json = Output(
    path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/ux_json/",
    mode="upload",
    type="uri_folder",
)

# submit and run pipeline
insights_job = submit_and_wait(ml_client, insights_pipeline_job)
```

To monitor the progress of the pipeline job, click on the Jobs icon from the [Azure ML studio](https://aka.ms/MBAzureMLStudio). By clicking on the pipeline job, you can get the status.

![Azure ML jobs](/img/azureml_jobs_page.png)

To visualize the individual progression of each of the components in the pipeline, click on the pipeline name. This gives you a better view of which components are completed, pending, or failed.

![Azure ML jobs progress](/img/rai_dashboard_pipeline.png)

After the RAI dashboard pipeline job has successfully completed, click on the “Models” tab of the Azure ML studio to find your registered model. Then, select the name of the model you trained from the previous exercise.

![Azure ML Models](/img/model-list.png)

From the Model details page, click on the “Responsible AI” tab. Then select the name of the dashboard name.

![Azure ML Model details](/img/model-details.png)

Terrific…you now have an Responsible AI dashboard.

![Azure ML dasboard gif](/img/rai-dashboard.gif)

# Exercise 4:  Debugging your model with Responsible AI (RAI) dashboard

In this exercise, you will use the RAI dashboard to debug the diabetes hospital readmission classification model. Traditional machine learning performance metrics provide aggregated calculations which are insufficient to understand model errors distribution or if the model is behaving responsibly. The RAI dashboard provides a way to analysis where there are disparities in the model’s predictions and understand the features that influence the model’s predictions.  It enables the assessment of disproportional representation for sensitive features such as race, gender, political views, or religion. This exercise will explore areas where there could be fairness, inclusiveness, or reliability & safety issues.

## Task 1: Error Analysis

While the overall performance metrics such as classification accuracy, precision, recall or MAE scores are good proxies to help you build trust with your model, they are insufficient in locating where in the data there are model inaccuracies. Machine Learning model often has errors that are not distributed uniformly in your underlying dataset. Although the overall model may have high accuracy, there may be a subset of data that the model is not performing well on. This subset of data may be a crucial demographic you do not want the model to be erroneous. This exercise, you will use the Error Analysis component of the RAI dashboard to identify where the model has a high error rate.

The Tree Map from the Error Analysis provides visual indicators to help in locating the problem areas quicker. For instance, the darker shade of red color a tree node has, the higher the error rate
 

![Tree Map](/img/1-ea-treemap.png)

In the above diagram, the first things we observe from the root node is that out of the 697 total test records, the component found 98 incorrect predictions while evaluating the model.  To investigate what's causing the high error rate with this group of patients, we will create a cohort for these groups of patients.

1. Find the leaf node with the darkest shade of red color.  Then, double-click on the leaf node to select all nodes in the path leading up to the node. This will highlight the path and display the feature condition for each node in the path.

![Tree map High error rate](/img/2-ea-tree-highest-error.png)

2. Click on the **Save as a new cohort** button on the upper right-hand side of the error analysis component. Note:  The dashboard displays the “Filters” with the feature conditions of the nodes in the path selection

3. Enter a name for the cohort: For example: ***Err: Prior_inpatient > 1 && < 4; Num_lab_procedure > 46***.  **Tip** The "Err:" prefix helps indicate that this cohort has the highest error rate. 

![Tree map High error rate](/img/3-ea-save-higherror-cohort.png)

4. Click on the **Save** button to save the cohort.

As much as it’s advantageous in finding out why the model is performing poorly, it is equally important to figure out what’s causing our model to perform well for some data cohorts.  So, we’ll need to find the tree path with the least number of errors to gain insights as to why the model is performing better in this cohort vs others. 

1. Find the leaf node with the lightest shade of gray color and the lowest error rate when you hover the mouse over the node.  Then, double-click on the leaf node to select all nodes in the path leading up to the node. This will highlight the path and display the feature condition for each node in the path.

![Tree map Low error rate](/img/4-ea-tree-lowest-error.png)

2. Click on the **Save as a new cohort** button on the upper right-hand side of the error analysis component. Note:  The dashboard displays the “Filters” with the feature conditions of the nodes in the path selection

3. Enter a name for the cohort: For example: ***Prior_inpatient = 0 && < 3; Num_lab_procedure < 27***.  

![Tree map High error rate](/img/5-ea-save-lowerror-cohort.png)

4. Click on the **Save** button to save the cohort.

This is adequate for you start investigating model inaccuracies, comparing the different features between top and bottom performing cohorts will be useful for improving our overall model quality.  

In the next exercise, you will use the Model Overview component of the RAI dashboard to start our analysis.

## Task 2: Model Overview

Evaluating the performance of a machine learning model requires getting a holistic understanding of its behavior. This can be achieved by reviewing more than one metric such as error rate, accuracy, recall, precision or MAE to find disparities among performance metrics.  In this exercise, we will explore how to use the Model Overview component of the Responsible AI (RAI) dashboard to find model performance disparities across cohorts. 

To start evaluating the performance of the model, you show be able to see the cohort you created in the previous exercise.  The "All data" is created by default by the dashboard.

![Model Overview](/img/1-mo-model-overview.png)

1. We can see that the "All data" has good overall accuracy score of 0.859 with sample size of 697 results in the test dataset.  
- However, the accuracy score for the "Err: Prior_inpatient > 1 && < 4; Num_lab_procedure > 46" cohort is 0.455.  This is a very poor accuracy score.  This confirms that the model is not performing well for diabetic patients this cohort that have prior hopitalization (prior_inpatient) between 1 and 4 times, as well as over 45 number of lab procedures (num_lab_procedures).  
- The cohort with the lowest error rate has an accuracy score of 0.962.  This is a very great overall accuracy score.  This confirms that the model is performing well for diabetic patients in this cohort that have prior hopitalization (prior_inpatient) less than 3 times, as well as less than 27 number of lab procedures(num_lab_procedures)
- The rate of False Positive rates are close to 0 for all cohorts; meaning there's a low number of cases where the model is inaccurately predicting patients that are going to be readmitted back to the hospital in 30 days. - Contrarily, the False Negative rate are close to 1, which is high. This indicates that there's a high number of cases where the model is falsely predicting that patients will not be readmitted, however the actual outcome is they will be readmitted in 30 days back to the hospital. 

This means the model is correctly predicting that patients will not be readmitted back to the hospital in 30 days a majority of the time.  However, the model is less confident in correctly predicting patients who will be readmitted within 30 days back to the hospital.

Probability Distribution

Similar to the performance metrics, we have the ability to view the “Probability Distribution.” 

1. Under the **Probability Distribution**, you can see that the cohort with the lower error rate has a higher probability at 0.972 of patients being readmitted in 30 days.  The cohort with the higher error rate has a lower probability at 0.844 of patients not being readmitted in 30 days. 

![Probability Distribution](/img/1-mo-dataset-prob.png)

## Task 3: Data Analysis

Data can be overrepresented in some cases and underrepresented in others. This may lead to data biases, causing the model to have fairness, inclusiveness, safety, and/or reliability issues.  In this exercise, we will explore how to use the Data Analysis component of the Azure Responsible AI (RAI) dashboard to discover the root-cause of the model’s poor performance. 

The Table view pane under Data Analysis helps visualize all the features and individual results of the data in a selected cohort.  For our analysis, we will use the Chart view pane to visualize the data.  First, we’ll use the chart to compare the data distribution of the number of patients not readmitted vs. readmitted in our test dataset using True Y and Predicted Y.  Then, we’ll examine if there are disparities for sensitive features or patients with prior hospitalization.

*Data imbalance issues with test dataset*

1. Select the “All data” option from the **“Select a dataset cohort to explore”** drop-down menu.
2. On the y-axis, we’ll click on the current selected “race” value, which will launch a pop-up menu.
3. Under “Select your axis value,” we’ll choose “Count.”

![Chart view](/img/1-da-chart-view.png)

4. On the x-axis, we’ll click on the current selected “Index” value, then choose “True Y” under the **“Select your axis value”** menu.

![predicted Y count](/img/2-da-count-truey.png)

5. We can see that, out of the 994 diabetes patients represented in our test data, 587 patients are not readmitted and 110 are readmitted back to a hospital within 30 days. These are the actual values or “TrueY.”

![predicted Y count](/img/3-da-count-predictedy.png)

For contrast, let’s compare those values with what our model actually predicts. To do that, let’s change the “True Y” value on the x-axis by selecting the “Predicted Y”.  Now, we see that the model’s number of patients readmitted back to the hospital is 24, while the number of patients not readmitted is 673. So, this exposes an extreme data imbalance issue where the model does not perform well for cases where patients are readmitted.

*Sensitive data representation*

1. Click on the x-axis label.
2. In the pop-up window pane, select the “Dataset” radio button.
3. Then under “select feature”, select “race” on the drop-down menu.
4. On the x-axis keep the “count” selected.

Now we'll compare race distribution, we find if there’s disparities in representation. Caucasians represent 73% of patients in the test data. African-Americans make up 21% of the patients. Hispanics represent 3% of the data. There’s obviously data gaps between the different ethnicities, which can lead to fairness issues. This is an area where ML professionals can intervene and help mitigate data disparities to make sure the model does not provide any racial biases.  If this model is to be use in a hospital when there is a predominately minority population, then the model will not be reliable in making accurate predictions for minority patients.

![race count](/img/6-da-age-count.png)

The gender representation among the patients are fairly balanced. So, this is not an area of concern.

![gender count](/img/5-da-gender-count.png)

Age is not proportionately distributed across our data, as seen in our three age groups. Diabetes tends to be more common among older age groups, so this may be an acceptable and expected disparity. However, this is another area for ML professionals to validate with medical specialists to understand if this is a normal representation of individuals with diabetes across age groups.

![age count](/img/6-da-age-count.png)


*Hospital readmission disparities*

1. Click on the y-axis label. Then, in the pop-up window pane, select the “Dataset” radio button.
3. Under “select feature”, select “race” on the drop-down menu.
4. Click on the x-axis label. Under the **“Select your axis value”** choose “Predicted Y” option.

![race readmission](/img/6-da-race-readmission.png)

For race, the chart shows that, due to the data imbalance, the model will not be able to accurately predict if a patient will be readmitted back to the hospital for some ethnicities. As we saw above, the Caucasian patients are overrepresented in this data set. 

![race readmission](/img/7-da-age-readmission.png)

There’s an overrepresentation of data for patients “over 60 years” and data underrepresentation for patients “30 years or younger.” Here, the effects of data imbalance were evident between the model’s classification of “Not readmitted” vs. “Readmitted.”

As you can see from all the data analysis we performed, data is a significant blind spot that is often missed when evaluating model performance. After tuning a model, you can increase accuracy scores, but that does not mean you have a model that is fair and inclusive.

## Task 4: Feature Importance

Examining a model is not just about understanding how accurately it can make a prediction, but also why it made the prediction. In some case a model has adverse behavior or makes a mistake that can be harmful to individuals or society. The ability to explain a model’s outcome provides shared understanding for data scientists, decision-makers, end-users, and auditors. In Some industries have compliance regulations that require organizations to provide an explanation for how and why a model made the prediction it did. In this exercise, we will use the Feature Importance component of the Azure Responsible AI (RAI) dashboard to understand what features have most influence on a model’s prediction.

*Global explanation*

The Feature Importance component of the RAI dashboard enables users to get a comprehensive understanding of why and how a model made a prediction. It displays the top data features that drove a model’s overall predictions in the Feature Important section of the dashboard. This is also known as the global explanation.

![global explanation](/img/1-fi-global-explanation.png)

Users can toggle the slider back-and-forth on top of the chart to display all the features, which are ordered in descending order of importance on the x-axis. The y-axis shows how much weight a feature has in driving a model’s prediction in comparison to the rest of the other features. The color of bar(s) on the chart corresponds to the cohorts created on the dashboard. In the diagram, it looks like *prior_inpatient*, *discharge_destination*, *diabetes_Med_prescribe*, *race*, and *prior_emergency* are the top 5 features driving our diabetic hospital readmission classification model predictions.

Having a sensitive feature such as *race* that is one of the top 5 features driving our model’s predictions is a red flag for potential fairness issues. This is an area where ML professionals can intervene to make sure the model does not making any racial biases leading to racial bias issues

*Feature Influence on a Model Prediction*

Let's explore how a feature's values positively or negatively influence a model’s outcome.

1. Select the *“Class: Not Readmitted”* option under the **Class importance weights** drop-down menu on the right-hand side of the dashboard.
2. The dashboard gives you the ability to double-click on any bar or box on the chart to get details. In our case, we’ll double-click on the *“number_diagnoses”* bar from the “All data” cohort (in blue color).
3. This generates another chart below the Aggregate feature importance chart.
4. The x-axis displays the number of diagnoses that were entered into the system for the diabetic patient.
5. The y-axis displays the level of contribution to the model making a prediction of Not Readmitted.
  * Numbers above 0 show the level of positive contribution to the model’s prediction of a patient Not Readmitted
  * Numbers below 0 show the level of negative contribution against the model’s prediction from Not Readmitted to Readmitted.

![Feature influence ](/img/2-fi-feature-influence.png)

As you can see from the graph, as we progress from 1 to 9 in “number_diagnoses” the model’s confidence decreases in a patient’s outcome to be not readmitted within 30 days back to the hospital. When the number of diagnoses is 7 or greater, the datapoints fall below the 0 axis.  Meaning the feature start to negatively impact the model to make a prediction of “Not Readmitted” and the model starts to predict a “Readmitted” classification. This makes intuitive sense because a diabetic patient with additional medical conditions is more likely to get sick and return to the hospital again.

This is the level of detail you can use the Feature Importance component of the RAI dashboard to understand why a model made a prediction and make informed decisions on how to improve the model.

*Local Explanation*

The component has a table view that enables users to see which records the model made a correct vs. incorrect prediction. You can use each individual patient’s record to see which features positively or negatively drove that individual outcome. This is especially useful when debugging to see where the model is performing erroneously for a specific patient, and which features are positive or negative contributors.

To explore, we’re going to:

1. Click on the "Individual Feature Importance" tab.

![Individual Feature influence](/img/3-fi-individual-influence.png)

2. Next, under the "Incorrect predictions" we’ll select record index #679. 

This will generate a Feature Important plot chart under the Table view. Here we see that “age”, “diabetes_Med_prescribe” and “insulin” are the top 3 features contributing to positively drive our model incorrectly predicting that the selected patient will not be readmitted within 30 days (the outcome should be Readmitted).

This exercise shows how Feature Importance removes the black box way of not knowing how the model went about making a prediction. It provides a global understanding of what key features the model uses to make a prediction. In addition, for each feature, the user has the ability to explore and visualize how it is positively or negatively influencing the model’s prediction for one class vs. another for each value in the feature. This also exposes the thresholds the model has to produce a certain outcome. We saw this in the “Number_Diagnoses” feature. 


# Conclusion

The RAI dashboard components provide practical tools to help data scientists and AI developere build machine learning models that are less harmful and more trustworthy to society. To improve the prevention of treats to human rights; discriminating or excluding certain group to life opportunities; and the risk of physical or psychological injury. It also helps to build trust in your model’s decisions by generating local explanations to illustrate their outcomes. Traditional model evaluation metrics are not enough to reveal responsible AI issues such as fairness, inclusiveness, reliability & safety or transparency. You need practical tools like the RAI dashboard to help you understand the impact of your model will on society and how to improve it.
