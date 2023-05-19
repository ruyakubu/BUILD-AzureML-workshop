In this lab, you’ll learn how to train a model in the cloud, and how to ensure it performs responsibly. We’ll be using the [UCI hospital diabetes dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00296/) to train a classification model using the Scikit-Learn framework. The model will predict whether or not a diabetic patient will be readmitted back to a hospital within 30 days of being discharged.


# Exercise 1: Training a model in the cloud

In this exercise, you'll train a custom model using Azure ML. Training in the cloud allows you to scale your training to use more compute power, and to track model versions.

## Setup
1. Open the Azure Machine Learning studio at https://ml.azure.com
2. Then open the *1-compute-training-job-cloud.ipynb* notebook.
3. Click on the **Run All** button on the top of the notebook to run the notebook.


## Task 1: Understand the training code

Let's start by reading our training and test data:

```python
import pandas as pd

train = pd.read_parquet('data/training_data.parquet')
test = pd.read_parquet('data/testing_data.parquet')
```

Next we'll split the training and test data into features X (our inputs), and targets Y (our labels). 

```python
# Split train and test data into features X and targets Y.
target_column_name = 'readmit_status'
Y_train = train[target_column_name]
X_train = train.drop([target_column_name], axis = 1)  
Y_test = test[target_column_name]
X_test = test.drop([target_column_name], axis = 1)  
```

Then we'll transform string data to numeric values using scikit-learn’s *OneHotEncoder*, and we'll standardize numeric data using scikit-learn’s *StandardScalar*. After that, we'll create a pipeline with these two processing steps, and the LogisticRegression classification model. And finally, we'll train the model using the *fit* function, and we'll score it.

```python
# Transform string data to numeric one-hot vectors
categorical_selector = selector(dtype_exclude=np.number)
categorical_columns = categorical_selector(X_train)
categorial_encoder = OneHotEncoder(handle_unknown='ignore')

# Standardize numeric data by removing the mean and scaling to unit variance
numerical_selector = selector(dtype_include=np.number)
numerical_columns = numerical_selector(X_train)
numerical_encoder = StandardScaler()

# Create a preprocessor that will preprocess both numeric and categorical data
preprocessor = ColumnTransformer([
('categorical-encoder', categorial_encoder, categorical_columns),
('standard_scaler', numerical_encoder, numerical_columns)])

clf = make_pipeline(preprocessor, LogisticRegression())

print('Training model...') 
model = clf.fit(X_train, Y_train)
print('Accuracy score: ', clf.score(X_test,Y_test))
```

Well done! You should have gotten an accuracy score somewhere between 0.8 and 0.85, which is a good score!


## Task 2: Create a cloud client

Before you can use the Azure Machine Learning studio, you need to create a cloud client session to authenticate and connect to the workspace. The authorization needs the subscription id, resource group, and name of the Azure ML workspace, which it gets from the "config.json" file in this repo.

Here's the authentication code:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)
```

## Task 3: Register the training and test data

Next we'll register the pre-cleansed training and test data from our local directory. 

```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

train_data_name = 'hospital_train_parquet'
test_data_name = 'hospital_test_parquet'

training_data = Data(
    name=train_data_name,
    path='data/training_data.parquet',
    type=AssetTypes.URI_FILE,
    description='RAI hospital train data'
)
tr_data = ml_client.data.create_or_update(training_data)

test_data = Data(
    name=test_data_name,
    path='data/testing_data.parquet',
    type=AssetTypes.URI_FILE,
    description='RAI hospital test data'
)
ts_data = ml_client.data.create_or_update(test_data)
```

These commands refer to the parquet training and test data stored in the local data directory, copy those files to the cloud, and give names to the new data resources. We'll use those names to refer to our data later.

You can verify the data is registered by opening the Azure ML studio at https://ml.azure.com, clicking on "Data," and finding the entries with the names we specified.


## Task 4: Create a compute cluster

Next we'll create a compute cluster that contains the details of the virtual machines we'll use to train our model. We'll specify a machine size, a minimum and maximum number of instances in the cluster, and the maximum number of seconds that a machine can be idle before we release for others to use.

```python
from azure.ai.ml.entities import AmlCompute

compute_name = 'trainingcompute'

my_compute = AmlCompute(
    name=compute_name,
    size='Standard_DS2_v2',
    min_instances=0,
    max_instances=4,
    idle_time_before_scale_down=3600
)
ml_client.compute.begin_create_or_update(my_compute).result()
```

You can verify the compute cluster was created in the Studio, by going to "Compute," and then "Compute clusters."


## Task 5: Create the job

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

target_column_name = 'readmit_status'

# Create the job.
job = command(
    description='Trains hospital readmission model',
    experiment_name='hospital_readmission',
    compute=compute_name,
    inputs=dict(training_data=Input(type='uri_file', path=f'{train_data_name}@latest'), 
                target_column_name=target_column_name),
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

You can take a look at the "src/train.py" file specified in the command, if you'd like. It contains the training code you're already familiar with, a bit of code to deal with the arguments, and a couple of lines of code to save the model using the MLFlow package. 

The job will take several minutes to run. You can follow the progress in the Studio by clicking on "Jobs," and then looking for the experiment name specified in the code.


## Task 6: Register the model

When the job finishes running, it outputs a trained model. We want to register that model, so that we can invoke it later to make predictions. Here's the code we need to register the model:

```python
from azure.ai.ml.entities import Model

model_name = 'hospital_readmission_model'

# Register the model.
model_path = f'azureml://jobs/{job.name}/outputs/model_output'
model = Model(name=model_name,
                path=model_path,
                type=AssetTypes.MLFLOW_MODEL)
registered_model = ml_client.models.create_or_update(model)
```

You can check that the model is registered by looking for the model name in the Studio, under "Models."


# Exercise 2:  Add a Responsible AI dashboard

In this exercise, you’ll be learning how to create the Responsible AI dashboard to use the model you trained in the previous exercise. The dashboard is comprised of different components to enable you to debug and analyze the model performance, identify errors, conduct fairness assessment, evaluate performance model interpretability and more.

## Prerequisites
1. Open the Azure Machine Learning studio at https://ml.azure.com
2. Then open the *2-create-responsibleai-dashboard.ipynb* notebook from the lab directory
3. Click on the **Run All** button on the top of the notebook to run the notebook

## Task 1: Define the dashboard components

The Responsible AI dashboard components are already pre-defined in the Azure Machine Learning studio. To use the components, you need to submit the component name and version to the Azure Machine Learning client’s session created in the previous exercise. The user has the option to add as many components they want on the Responsible AI dashboard. The components you’ll be using are:

* Error Analysis
* Explanation
* Insight Gather

``` python
label = 'latest'

rai_constructor_component = ml_client_registry.components.get(
    name='microsoft_azureml_rai_tabular_insight_constructor', label=label
)

# We get latest version and use the same version for all components
version = rai_constructor_component.version

rai_explanation_component = ml_client_registry.components.get(
    name='microsoft_azureml_rai_tabular_explanation', version=version
)

rai_erroranalysis_component = ml_client_registry.components.get(
    name='microsoft_azureml_rai_tabular_erroranalysis', version=version
)

rai_gather_component = ml_client_registry.components.get(
    name='microsoft_azureml_rai_tabular_insight_gather', version=version
)
```

## Task 2: Define the job to create the RAI dashboard insights

When you have specified the RAI components you need, it is time to define an [Azure pipeline](https://aka.ms/MBAzureMLPipeline) and configure each component. (**NOTE**: For the list of additional settings to configure each of your RAI components, refer to [RAI component parameters](https://aka.ms/MBARAIComponentSettings)).

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
            title='RAI Dashboard',
            task_type='classification',
            model_info=expected_model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),            
            train_dataset=training_data,
            test_dataset=testing_data,
            target_column_name=target_column_name,
            categorical_column_names=json.dumps(categorical),
        )
        create_rai_job.set_limits(timeout=120)
``` 

The Explanation component is responsible for the dashboard providing a better understanding of what features influence the model’s predictions. It takes a comment that is a description field. Then sets the *rai_insights_dashboard* to be the output insights generated from the RAI pipeline job for Explanations.

``` python
        # Explanation
        explanation_job = rai_explanation_component(
            comment='Explain the model',
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        )
        explanation_job.set_limits(timeout=120)
```

The Error Analysis component is responsible for the dashboard providing an error distribution of the feature groups contributing to the model inaccuracies. Its only configuration is to set the *rai_insights_dashboard* to be the output insights generated from the RAI pipeline job for the overall and feature error rates.

``` python
        # Error Analysis
        erroranalysis_job = rai_erroranalysis_component(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        )
        erroranalysis_job.set_limits(timeout=120)
```

Once all the RAI components are configured with the parameters needed for the use case, the next thing to do is add all of them into the list of insights to include on the RAI dashboard. Then upload the dashboard instance and UX settings for the RAI Dashboard.

``` python
        rai_gather_job = rai_gather_component(
            constructor=create_rai_job.outputs.rai_insights_dashboard,
            insight_1=explain_job.outputs.explanation,
            insight_4=erroranalysis_job.outputs.error_analysis,
        )
        rai_gather_job.set_limits(timeout=120)
```

The pipeline job outputs are the dashboard and UX configuration to be displayed.

## Task 3: Run job to create the dashboard

After the pipeline is defined, we'll initialize it by specifying the input parameters and the path to the outputs. Lastly, we use the *submit_and_wait function* to run the pipeline and register it to the Azure Machine Learning studio.

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
    path=f'azureml://datastores/workspaceblobstore/paths/{rand_path}/dashboard/',
    mode='upload',
    type='uri_folder',
)
insights_pipeline_job.outputs.ux_json = Output(
    path=f'azureml://datastores/workspaceblobstore/paths/{rand_path}/ux_json/',
    mode='upload',
    type='uri_folder',
)

# submit and run pipeline
insights_job = submit_and_wait(ml_client, insights_pipeline_job)
```

The job will take about 6 minutes to run. 

To monitor the progress of the pipeline job, click on the Jobs icon from the [Azure ML studio](https://aka.ms/MBAzureMLStudio). By clicking on the pipeline job, you can get the status.

![Azure ML jobs](/img/azureml_jobs_page.png)

To visualize the individual progression of each of the components in the pipeline, click on the pipeline name. This gives you a better view of which components are completed, pending, or failed.

![Azure ML jobs progress](/img/rai_dashboard_pipeline.png)

After the RAI dashboard pipeline job has successfully completed, click on the “Models” tab of the Azure ML studio to find your registered model. Then, select the name of the model you trained in the previous exercise.

![Azure ML Models](/img/model-list.png)

From the Model details page, click on the “Responsible AI” tab. Then select the dashboard name.

![Azure ML Model details](/img/model-details.png)

Terrific…you now have a Responsible AI dashboard.

![Azure ML dasboard gif](/img/rai-dashboard.gif)

# Exercise 3:  Debugging your model with Responsible AI (RAI) dashboard

In this exercise, you will use the RAI dashboard to debug the diabetes hospital readmission classification model. Traditional machine learning performance metrics provide aggregated calculations which are insufficient in exposing any undesirable responsible AI issues. The RAI dashboard enables users to identify model error distribution; understand features driving a model’s outcome; and assess any disproportional representation for sensitive features such as race, gender, political views, or religion.

## Task 1: Error Analysis

A machine Learning model often has errors that are not distributed uniformly in your underlying dataset. Although the overall model may have high accuracy, there may be a subset of data that the model is not performing well on. This subset of data may be a crucial demographic you do not want the model to be erroneous. In this exercise, you will use the Error Analysis component to identify where the model has a high error rate.

The Tree Map from the Error Analysis provides visual indicators to help in locating the problem areas quicker. For instance, the darker shade of red color a tree node has, the higher the error rate.

![Tree Map](/img/1-ea-treemap.png)

In the above diagram, the first thing we observe from the root node is that out of the **697** total test records, the component found **98** incorrect predictions while evaluating the model.  To investigate where there are high error rates with the patients, we will create cohorts for the groups of patients.

1. Find the leaf node with the darkest shade of red color.  Then, double-click on the leaf node to select all nodes in the path leading up to the node. This will highlight the path and display the feature condition for each node in the path.

![Tree map High error rate](/img/2-ea-tree-highest-error.png)

2. Click on the **Save as a new cohort** button on the upper right-hand side of the error analysis component. 

3. Enter a name for the cohort: For example: ***Err: Prior_inpatient > 1 && < 4; Num_lab_procedure > 46***.  (**Tip** The "Err:" prefix helps indicate that this cohort has the highest error rate). 

![Tree map High error rate](/img/3-ea-save-higherror-cohort.png)

4. Click on the **Save** button to save the cohort.
5. Click on the **Clear selection** button on the left side of the Tree to clear the selection.

As much as it’s advantageous in finding out why the model is performing poorly, it is equally important to figure out what’s causing our model to perform well for some data cohorts.  So, we’ll need to find the tree path with the least number of errors to gain insights as to why the model is performing better in this cohort vs others. 

1. Find the leaf node with the lightest shade of gray color and the lowest error rate when you hover the mouse over the node.  Then, double-click on the leaf node to select all nodes in the path leading up to the node. This will highlight the path and display the feature condition for each node in the path.

![Tree map Low error rate](/img/4-ea-tree-lowest-error.png)

2. Click on the **Save as a new cohort** button on the upper right-hand side of the error analysis component. 

3. Enter a name for the cohort: For example: ***Prior_inpatient = 0 && < 3; Num_lab_procedure < 27***.  

![Tree map High error rate](/img/5-ea-save-lowerror-cohort.png)

4. Click on the **Save** button to save the cohort.
5. Click on the **Clear selection** button on the left side of the Tree to clear the selection.

This is adequate for you to start investigating model inaccuracies, comparing the different features between top and bottom performing cohorts will be useful for improving our overall model quality.  

In the next exercise, you will use the Model Overview component of the RAI dashboard to start our analysis.

## Task 2: Model Overview

Evaluating the performance of a machine learning model requires getting a holistic understanding of its behavior. This can be achieved by reviewing more than one metric such as error rate, accuracy, recall, precision or MAE to find disparities among performance metrics. In this exercise, we will explore how to use the Model Overview component to find model performance disparities across cohorts. 

***Dataset Cohorts analysis***

To compare the model’s behavior across the different cohorts you created in the previous exercise, we’ll start looking for any performance metric disparities. The “All data” cohort is created by default by the dashboard.

![](/img/1-mo-model-overview.png)

*Accuracy Score*

* The “All data” cohort has a good overall accuracy score of 0.859 with a sample size of 697 results in the test dataset.
* The cohort with the lowest error rate has a great accuracy score of 0.962. Although, the sample size of 105 patients is small.
* The erroneous cohort with the highest error rate has a poor accuracy score of 0.455. However, the sample size of 22 is a very small number of patients for this cohort, which the model is not performing well on.

*False Positive Rate*

* The False Positive rates are close to 0 for all cohorts; meaning there’s a small number of cases where the model is incorrectly predicting that patients are going to be *readmitted*, when the actual outcome is *not readmitted*.

*False Negative Rate*

* On the contrary, the False Negative rate are close to 1, which is high. This indicates that there’s a high number of cases where the model is incorrectly predicting that a patient will be *not readmitted*. The actual outcome is they will be *readmitted* in 30 days back to the hospital.

This means the model is correctly predicting that patients will not be readmitted back to the hospital in 30 days a majority of the time.  However, the model is less confident in correctly predicting patients who will be readmitted within 30 days back to the hospital.

***Feature cohort analysis***

There are cases where you’ll need to isolate your data by focusing on a specific feature to drill-down on a granular level to determine if the feature is a key contributor to the model’s poor performance. The RAI dashboard has built-in intelligence to divide the selected feature values into various meaningful cohorts for users to do feature-based analysis and compare where in the data the model is not doing well.

![](/img/1-mo-feature-pane.png)

To look closer at what impact the *"Prior_Inpatient"* feature has to the model's performance:

1. Under "Model Overview", switch to the **"Feature cohorts"** tab on top of the section.
2. Under the “Feature(s)” drop-down menu, scroll down the list and select the *“Prior_Inpatient”* checkbox. This will display 3 different feature cohorts and the model performance metrics.

![](/img/5-mo-feature-cohort.png)
 
* We see that the *prior_inpatient < 3.67* cohort has a sample size of 671. This means a majority of patients in the test data were hospitalized less than 4 times in the past.

  * The model’s accuracy rate for this cohort is 0.86, which is good.
  * The model has a very high false negative rate of 0.989, which suggests that model is incorrectly predicting readmitted patients as not readmitted. This is consistent with our findings in the dataset cohort findings above.

* Only 18 patients from the test data fall in the *prior_inpatient ≥ 3.67 and < 7.33* cohort.

  * The model’s accuracy rate is 0.778, which is a low confidence.
  * For this cohort, the model false negative is 0, meaning the model is correctly predicting patients that will be readmitted. The false positive rate of 0.444, means ths model is falsely predicting patients that will be readmitted almost half the time for this cohort.  However, the sample size of 18 patients is too small to make any conclusions.

* Lastly, there are just 8 patients from the *prior_inpatient >= 7.33* cohort.

  * The model accuracy of 1 for this cohort means there are no inaccuracies.  So, there are no false positive and false negative rates.  Since the sample size of 8 is too small, we can’t make any conclusions.
 
## Task 3: Data Analysis

Data can be overrepresented or underrepresented in some cases. This imbalance can cause a model to be more favorable to one group vs another. As a result, the model has data biases that lead to fairness, inclusiveness, safety, and/or reliability issues. In this exercise, we will explore how to use the Data Analysis component to explore data distribution.

The Table view pane under Data Analysis helps visualize all the individual results in the dataset including the features, actual outcome, and predicted outcome. For our analysis, we will use the Chart view pane under the Data Analysis component to visualize insights from our data.

***Data imbalance issues with test dataset***

Let’s start by looking at the actual number of patients that were not readmitted vs. readmitted in our test dataset using True Y.

1. Under the Data Analysis component, click on the **Chart view** tab.
2. Select the “All data” option from the **“Select a dataset cohort to explore”** drop-down menu.
3. On the y-axis, we’ll click on the current selected “race” value, which will launch a pop-up menu.
4. Under “Select your axis value,” we’ll choose “Count.”

![Chart view](/img/1-da-chart-view.png)

5. On the x-axis, we’ll click on the current selected “Index” value, then choose “True Y” under the **“Select your axis value”** menu.

![True Y count](/img/2-da-count-truey.png)

We can see that from the actual dataset values, out of the 697 diabetes patients represented in our test data, 587 patients are not readmitted and 110 are readmitted to a hospital within 30 days.
  * This imbalance of data can cause the model not to learn well due to not having enough data representation of readmitted patients.  
  * This is consistent with the high false positive rates in the last exercise.

***Sensitive data representation***

To examine the race distribution:

1. Click on the x-axis label.
2. In the pop-up window pane, select the “Dataset” radio button.
3. Then under “select feature”, select “race” on the drop-down menu.
4. On the x-axis keep the “count” selected.

We see race disparities where Caucasians represent 73% of patients in the test data; African-Americans make up 21% patients; and Hispanics represent 3% of patients. This shows a data imbalance between the different ethnicities, which can lead to racial biases and fairness issues.  

![Race count](/img/4-da-race-count.png)

The gender representation among the patients are fairly balanced. So, this is not an area of concern.

![Gender count](/img/5-da-gender-count.png)

Age is not proportionately distributed across our data, as seen in our three age groups. Diabetes tends to be more common among older age groups, so this may be an acceptable and expected disparity. This is an area for ML professionals to validate with medical specialists to understand if this is a normal representation of individuals with diabetes across age groups.

![Age count](/img/6-da-age-count.png)

As you can see from all the data analysis we performed, data is a significant blind spot that is often missed when evaluating model performance. After tuning a model, you can increase accuracy scores, but that does not mean you have a model that is fair and inclusive.

## Task 4: Feature Importance

Examining a model is not just about understanding how accurately it can make a prediction, but also why it made the prediction. In some case a model has adverse behavior or makes a mistake that can be harmful to individuals or society. Some industries have compliance regulations that require organizations to provide an explanation for how and why a model made the prediction it did. In this exercise, we will use the Feature Importance component to understand what features have the most influence on a model’s prediction.

*Global explanation*

The Feature Importance component enables users to get a comprehensive understanding of why and how a model made a prediction. It displays the top data features that drove a model’s overall predictions in the Feature Important section of the dashboard. This is also known as the global explanation.

![Global explanation](/img/1-fi-global-explanation.png)

Users can toggle the slider back-and-forth on top of the chart to display all the features, which are ordered in descending order of importance on the x-axis. The y-axis shows how much weight a feature has in driving a model’s prediction in comparison to the rest of the other features. The color of bar(s) on the chart corresponds to the cohorts created on the dashboard. In the diagram, it looks like *prior_inpatient*, *discharge_destination*, *diabetes_Med_prescribe*, *race*, and *prior_emergency* are the top 5 features driving our diabetic hospital readmission classification model predictions.

Having a sensitive feature such as race that is one of the top 5 features driving our model’s predictions, is a red flag for potential fairness issues. This is an area where ML professionals can intervene to make sure the model does not make any racial bias predictions.

*Local Explanation*

The Feature Importance component also has a table view that enables users to see which records the model made a correct vs. incorrect prediction. You can use each individual patient’s record to see which features positively or negatively drove that individual outcome. This is especially useful when debugging to see where the model is performing erroneously for a specific patient, and which features are positive or negative contributors.

To explore, we’re going to:

1. Click on the "Individual Feature Importance" tab.

![Individual Feature influence](/img/3-fi-individual-influence.png)

2. Next, under the "Incorrect predictions" we’ll select record index #679. 

This will generate a Feature Important plot chart under the Table view. Here we see that “age”, “diabetes_Med_prescribe” and “insulin” are the top 3 features contributing to positively drive our model incorrectly predicting that the selected patient will not be readmitted within 30 days (the actual outcome should be Readmitted).

This exercise shows how Feature Importance removes the black box way of not knowing how the model went about making a prediction. It provides a global understanding of what key features the model uses to make a prediction. In addition, for each feature, the user has the ability to explore and visualize how it is positively or negatively influencing the model’s prediction for one class vs. another for each value in the feature. 

# Conclusion

The RAI dashboard components provide practical tools to help data scientists and AI developers to build machine learning models that are less harmful and more trustworthy to society. To improve the prevention of treats to human rights; discriminating or excluding certain group to life opportunities; and the risk of physical or psychological injury. It also helps to build trust in your model’s decisions by generating local explanations to illustrate their outcomes. Traditional model evaluation metrics are not enough to reveal responsible AI issues such as fairness, inclusiveness, reliability & safety or transparency. You need practical tools like the RAI dashboard to help you understand the impact of your model will on society and how to improve it.
