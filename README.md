
In this lab you’ll be learning the end-to-end machine learning process that involves data preparation, training a model, registering the model and debug them.  First you will go through the process of training the model locally.   Then you will learn how to make your machine learning model training process more dynamic, manageable and scalable in the cloud using Azure Machine Learning service. For the workshop we’ll be using the [UCI hospital diabetes dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00296/) to train a classification model using the Scikit-Learn framework.  The model will predict whether or not a diabetic patient will be readmitted back to a hospital with 30 days of being discharged.  To prep the data to be used for training our model we are going to review our dataset to cleanse any redundant data, missing values, reformat data or delete any irrelevant data that will be used for our use case.  After the data has been clean, we’ll train.  

# Exercise 1:  Training a model locally

## Task #1: Data Prep

For this exercise, you’ll load the **1-dataprep-and-training-local.ipynb** jupyter notebook.

The first thing we need is to read the hospital diabetes data into a dataframe that will allow us to visualize and perform data manipulation.  To do this will be using the read_csv function from the “pandas” library.  

```python
df = pd.read_csv('diabetic_data.csv')
```

Next, take a look at the size of the data to verify that it will be sufficient for our training.  As you can see 101,766 is a very large data set.  

Let’s display the dataframe to see what kind of diabetes is in the hospital data.  As you can see it have the patient race, gender, age, weight, their prior hospitalization inform, prescribed medication and whether or not they were readmitted back to the hospital.  

*Checking for special characters*

One of the invalid data that stands out from the Weight column is the “?” special characters that are used as fillers in our dataset.  We’ll loop through the dataset to identify columns that have this special character.

```python
print(df.loc[: , (df == '?').any()])
```

The output shows that that race, weight, payer_code, medical_specialty, diag_1, diag_2, and diag_3 are the column with missing values that have the special character.  To fix this we’ll change the character to null by using the numpy NAN function.

```python
df = df.replace('?', np.NaN) 
```

*Checking for null values*

```python
print(df.isna().any())
```

When we run the “isna” function on the dataframe, we can see that there are 7 columns with null values in our dataset.  Depending on the number of null values that a column has, we’ll keep or delete the column.  As we count the columns with a high number of null values from the weight column.  Although, Weight is liking to impact a diabetes patient’s readmission status, a majority of it’s values are null.   Race is another column that has 12% of its values missing.  This is not a signification number; we can drop the null values.  The null values from payer_code and medical_specialty are significant either, so can drop the fields.

```python
df = df.dropna()
```

*Deleting irrelevant columns*

The patient's form of payment does not have impact on the return to the hospital, so we dropped the Payer_Code. However, we added the Medicare and Medicaid column to indicate whether or not the Payer_Code for the hospital bill used a subsidize government medical assistance for low-income patients. This is to help us understand if there are any socioeconomic gaps in the diabetic patient demographic.

```python
df.loc[:, "medicare"] = (df.payer_code == "MC")
df.loc[:, "medicaid"] = (df.payer_code == "MD")
```

There are 20+ columns of whether or not a patient took a certain diabetic medication (rosiglitazon, citoglipton, metformin etc.) that have no correlation on a patient's return to the hospital.  As result, we’ll delete these medications.

```python
df.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'admission_type_id', 
         'repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide',
        'pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton', 'metformin',
        'glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone', 'diag_2', 'diag_3', 'change'], axis=1, inplace=True)
```

Lastly, unique patient numbers or hospital record identifications do not have value to model.    Also, a patient’s first, second or third diagnosis while that were hospitalized may be useful , but are too random from patient to patient.

*Formatting data*

Part of working with data is formatting the columns or values into names that are more meaningful and easier to understand.  Here we’ll change the readmitted outcome from NO or <30.

```python
df['readmitted'] = df['readmitted'].replace({"NO":"not readmitted", "<30":"readmitted"})
```

Next, the age group are represented in brackets.  For example, [0 - 10] denotes the age between 0 and 10 years old.  In addition, since there’s a small sample size of patients in this age group will consolidate the age groups.

```python
df.loc[:, "age"] = df["age"].replace( ["[0-10)", "[10-20)", "[20-30)"], "30 years or younger")
df.loc[:, "age"] = df["age"].replace(["[30-40)", "[40-50)", "[50-60)"], "30-60 years")
df.loc[:, "age"] = df["age"].replace(["[60-70)", "[70-80)", "[80-90)", "[90-100)"], "Over 60 years")
```

## Task # 2:  Training data

Now that the data has been cleansed, it is ready to be used for training our model.  To do this we will split out data into training and testing data sets with scikit-learns train_test_spilt function.  This is optional, but we will store our data for later use.  The file format we’ll use for our dataset is parquet.

Next, to normalize our data, we’ll create a helper function that identified all the indices for the non-numeric columns.   Scikit-Learn’s OneHotEncoder will be used to encode all the string or object columns.  Next, we’ll use the StandardScalar to encode the number column fields.  The ColumnTransformer will be use to perform the data transformation.   Where create a pipeline to put all of these prepressing tasks for the model to execute before training the model.

For our diabetes readmission classification, we use the LogisticRegression model on our dataset.

After training the model, we can review the accuracy score to evaluate how well our trained model performed.   The score of 0.86 looks good.

# Exercise 2:  Training a model in the cloud

In the previous lab, we used the diabetes hospital readmission dataset to understand and wrangled data into a clean dataset to train a classification model to predication whether a diabetics patient would be readmitted back to a hospital within 30 days of being discharge.  All of this done on the local machine.   In this session, you will learn how to train a model in the cloud using the end-to-end streamline process to make your model reproduceable, easier to manage, and scalable. 

## Task# 1: Create a cloud client*

Before you can use the Azure Machine Learning studio, you need to create a cloud client session to authenticate and connect to the workspace.  The authorization needs the subscription id, resource group, and name of the Azure ML workspace.

```python
registry_name = "azureml"
credential = DefaultAzureCredential()
ml_client =  MLClient.from_config(credential=credential)

ml_client_registry = MLClient(
    credential=credential,
    subscription_id=ml_client.subscription_id,
    resource_group_name=ml_client.resource_group_name,
    registry_name=registry_name
    )
```

## Task 2: Register the dataset

Data preparation is a long and tedious process the takes up a majority of the machine learning process.  After you have cleanse into a good state, Azure Machine Learning provides datastores for you to register and store your datasets to.  Your dataset is stored in one location, but it can be referenced as a point without having to move the data.  The benefit of registering your datasets is, individuals on your team can reference them; and you can track all the experiments that used. 

We are going to use the training and testing data that you used in Lab#1 and register it to the cloud.   To register the files to Azure, we need to use the connect client.  In the case, we need to specify the locally directory path.  The AssetTypes.URI_FILE specifies what time of data file we are using.

```python
import os
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

training_data = Data(
    name=training_dataset_filename,
    path='data/train_dataset.parquet',
    type=AssetTypes.URI_FILE,
    description="Patient train data",  
)

tr_data = ml_client.data.create_or_update(training_data)

testing_data = Data(
    name=testing_dataset_filename,
    path='data/test_dataset.parquet',
    type=AssetTypes.URI_FILE,
    description="Patient test data",  
)

te_data = ml_client.data.create_or_update(testing_data)
```

## Task 3: Create a Compute

To run a job or machine learning computation, we need to create a compute instance.  You can create none in notebook or use existing one.  To create a compute, you need a name and the size of the instance.  You can also specify how you and the instance to scare during runtime.

```python
from azure.ai.ml.entities import AmlCompute

compute_name = "trainingcompute"

all_compute_names = [x.name for x in ml_client.compute.list()]

if compute_name in all_compute_names:
    print(f"Found existing compute: {compute_name}")
else:
    my_compute = AmlCompute(
        name=compute_name,
        size="Standard_DS2_v2",
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=3600
    )
    ml_client.compute.begin_create_or_update(my_compute)
    print("Initiated compute creation")
```

## Task 4: Create an environment

An environment specifies all the dependencies that a task or job that you need to run need.  This can include python version, docker file, python libraries, etc.  You have the option of creating a custom environment from scratch that suits your needs or use one of the Azure Machine Learning curated environments available.  In our case, we are using Responsible AI’s out of the box curated environment: *AzureML-responsibleai-0.20-ubuntu20.04-py38-cpu* since we’ll need needing it in the next lab. (**NOTE**:  Select the Environments tab in Azure Machine Learn Studio to see the available environments) 

## Task 5: Define the training model

We’ll be using Azure Machine Learning components to divide the experiment into different tasks. Components are reusable independent units of tasks that have inputs and output in machine learning (e.g., data cleaning, training a model, registering a model, deploying a model etc.). For our experiment, we will create a component for training a model. The component will consist of a python training code with inputs and outputs.
 
The first thing to define in our python code is the training script containing a function that declares an argument parser object that adds the names and types for the input and output parameters. For our Diabetes Hospital Readmission use case, we will be passing the path to our training dataset and the target column name as the classifier. Then the trained model will be the output for the script.

```python
def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--target_column_name", type=str, help="Name of target column")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    # parse args
    args = parser.parse_args()    

    # return args
    return args
```

In our python code, we’ll define a main class that the takes the input arguments to train the month.  As you will see the code to change the model remain the same.  The only change is create an experiment in Azure Machine Learning studio and using MLFlow to start the artifacts and metrics from your training model for tracking.

```python
current_experiment = Run.get_context().experiment
tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(current_experiment.name)
```

After your model has been training.  We’ll store the Scikit-Learn model output in a local directory then with MLFlow’s save_model function for scikit-learn models to save the model output.

```python
    model_dir =  "./model_output"
    with tempfile.TemporaryDirectory() as td:
        print("Saving model with MLFlow to temporary directory")
        tmp_output_dir = os.path.join(td, model_dir)
        mlflow.sklearn.save_model(sk_model=model, path=tmp_output_dir)

        print("Copying MLFlow model to output path")
        for file_name in os.listdir(tmp_output_dir):
            print("  Copying: ", file_name)
            # As of Python 3.8, copytree will acquire dirs_exist_ok as
            # an option, removing the need for listdir
            shutil.copy2(src=os.path.join(tmp_output_dir, file_name), dst=os.path.join(args.model_output, file_name))
```

Once you have defined you python code with the arguments and the main function to execute, we can create a compute to load the python to Azure Machine Learning components.

## Task 6: Create a job 

To define our training component, we’ll need to create a yaml file where we specified the component name, input, and output parameters; the location of the python code that trains the model; the command-line to execute the python code; and the environment to run the code. You can use Azure Machine Learning the Responsible AI’s out of the box curated environment: *AzureML-responsibleai-0.20-ubuntu20.04-py38-cpu*. Then use the client session to register the component definition in workspace components.

```python
from azure.ai.ml import load_component

yaml_contents = f"""
$schema: http://azureml/sdk-2-0/CommandComponent.json
name: rai_hospital_training_component
display_name: hospital  classification training component for RAI example
version: {rai_hospital_classifier_version_string}
type: command
inputs:
  training_data:
    type: path
  target_column_name:
    type: string
outputs:
  model_output:
    type: path
code: ./component/
environment: azureml://registries/azureml/environments/AzureML-responsibleai-0.20-ubuntu20.04-py38-cpu/versions/4
""" + r"""
command: >-
  python hospital_training.py
  --training_data ${{{{inputs.training_data}}}}
  --target_column_name ${{{{inputs.target_column_name}}}}
  --model_output ${{{{outputs.model_output}}}}
"""

yaml_filename = "RAIhospitalClassificationTrainingComponent.yaml"

with open(yaml_filename, 'w') as f:
    f.write(yaml_contents.format(yaml_contents))
    
train_component_definition = load_component(
    source=yaml_filename
)

ml_client.components.create_or_update(train_component_definition)
```

## Task 7: Create a pipeline in the cloud

An Azure Machine Learning pipeline packages all the components and runs them sequentially during runtime with a specified compute instance, docker images or conda environments in the job process. After the training component defined above has been created, we need to define the pipeline that is going to package all the dependencies needed to run the training experiment. To do this, you will need the following:
 
* The experiment name and description
* Input object for the training dataset path
* Input object for the testing dataset path
* Get component name that trains the model
* Get component name that registers the model
* The compute instance for running the training job
All of this information is packaged in a pipeline job to run the experiment for training and registering the model.

```python
from azure.ai.ml import dsl, Input

hospital_train_parquet = Input(
    type="uri_file", path="data/train_dataset.parquet", mode="download"
)

hospital_test_parquet = Input(
    type="uri_file", path="data/test_dataset.parquet", mode="download"
)

@dsl.pipeline(
    compute=compute_name,
    description="Register Model for RAI hospital ",
    experiment_name=f"RAI_hospital_Model_Training_{model_name_suffix}",
)
def my_training_pipeline(target_column_name, training_data):
    trained_model = train_component_definition(
        target_column_name=target_column_name,
        training_data=training_data
    )
    trained_model.set_limits(timeout=120)

    _ = register_component(
        model_input_path=trained_model.outputs.model_output,
        model_base_name=model_base_name,
        model_name_suffix=model_name_suffix,
    )

    return {}

model_registration_pipeline_job = my_training_pipeline(target_column, hospital_train_parquet)
```

## Task 8: Run the training job

Once you have defined and registered the pipeline to the workspace, you can submit the pipeline to run in a job. In our python code, we are checking the status of the job.

```python
from azure.ai.ml.entities import PipelineJob
import webbrowser

def submit_and_wait(ml_client, pipeline_job) -> PipelineJob:
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    assert created_job is not None

    while created_job.status not in ['Completed', 'Failed', 'Canceled', 'NotResponding']:
        time.sleep(30)
        created_job = ml_client.jobs.get(created_job.name)
        print("Latest status : {0}".format(created_job.status))

    # open the pipeline in web browser
    webbrowser.open(created_job.services["Studio"].endpoint)
    
    #assert created_job.status == 'Completed'
    return created_job

# This is the actual submission
training_job = submit_and_wait(ml_client, model_registration_pipeline_job)
```

To monitor progess of the job and status of all the components in the pipeline job is by clicking on the Jobs icon from the Azure Machine Learning Studio.
 
 ![training job](/img/training-job.png)
 
From the Jobs list, you can click on the job to view the jobs progress and can drill-down to pinpoint where an error occurred. You will see the Diabetes Hospital Readmission dataset feeding as an input into our training component and the output model feeding into the register model marked as complete. After the pipeline has successfully finished running, you will have a trained model that is registered to the Azure Machine Learning Studio.

![training job progress](/img/training-job-progress.png)

# Exercise 3:  Add a Responsible AI dashboard

In this exercise, you'll be learning how to create the Responsible AI dashboard for the trained model. This will help us get a hoslitic insights from our diabetes hospital readmission model.  In addition, it will expose any errors or areas where there could be undesireable responsible AI issues.  The dashboard is comprised of different components to enable you to debug and analyze the model performance, identify errors, performance fairness assessment, performance model interpretability and more. The dashboard can be create using the Azure Portal, CLI or SDK. In this exercise, we will be using the SDK to create the dashboard.

## Task 1: Define the dashboard components

The Responsible AI dashboard components are already pre-defined in the Azure Machine Learning studio.  To use the components, you need submit the component name and version to the Azure Machine Learn clients session created in the previous exercise. The user have the option to add a many components they want on the Responsible AI dashboard. The components you'll be are:

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

The Explanation component is responsible for the dashboard providing a better understanding of what features influence the model’s predictions. It takes the a comment description pertaining to your use case. Then set the *rai_insights_dashboard* to be the output insights generated from the RAI pipeline job for Explanations.

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

After the pipeline is defined, we'll initialize it by specifying the input parameters and the path to the it's outputs. Lastly, we use the submit_and_wait function to run the pipeline and register it to the Azure ML workspace.

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

![Azure ML jobs progress](/img/rai-dashboard-pipeline.png)

After the RAI dashboard pipeline job has successfully completed, click on the “Models” tab of the Azure ML studio to find your registered model. Then, select the name of the model you trained from the previous exercise.

![Azure ML Models](/img/model-list.png)

From the Model details page, click on the “Responsible AI” tab. Then select the name of the dashboard name.

![Azure ML Model details](/img/model-details.png)

Terrific…you now have an RAI dashboard.

![Azure ML dasboard gif](/img/rai-dashboard.gif)

# Exercise 4:  Debug your model with Responsible AI dashboard

In this exercise, you will use the RAI dashboard to debug the diabetes hospital readmission classification model. Traditional machine learning performance metrics provide aggregated calculations which are insufficient to understand model errors distribution or if the model is behaving responsibly. The RAI dashboard provides a way to analysis where there are disparities in the model’s predictions, and understand the features that influence the model’s predictions. This exercise will explore area where there could be fairness, inclusiveness or reliability & safety issues.

## Task 1: Error Analysis
## Task 2: Model Overview
## Task 3: Data Analysis
## Task 4: Feature Importance
