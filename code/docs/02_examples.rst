.. _examples-label:

Examples
********
Using our library in conjunction with other libraries like scikit-learn, 
it is possible to construct a machine learning pipeline for RUL prediction with only little effort. 

RUL prediction with simple feature extraction
=============================================
This example shows how to use the ml4pdm library to learn a RUL pipeline for the cmapss dataset.

The cmapss dataset
------------------
The cmapss data consists of 25 features of which 24 are timeseries features.
That means the length of these features differ for different instances. 
This is an important information for the pipeline we want to build.

We load the cmapss train and test datasets using the DatasetParser from our library::

   from ml4pdm.parsing import DatasetParser
   train_dataset, test_dataset = DatasetParser.get_cmapss_data(test=True)

The test_dataset has instances in test_dataset.data and labels in test_dataset.target which can be directly used for testing our pipeline later.
However the training data is full runtime data.
That means we don't have any labels to train on.
We can however use a built-in function to generate a prepared training dataset directly::

   prepared_train_dataset = train_dataset.generate_simple_cut_dataset(cut_repeats=5, min_length=5, max_length=155)

With this we have a dataset 5 times the size of the original train dataset (5 repeated cuts) and more importantly also labels to work with.

The pipeline
------------

As our library is build to work with sklearn, we can build a pipeline using the make_pipeline method from sklearn and simply put in classes from our library::

   from sklearn.pipeline import make_pipeline
   from sklearn.ensemble import RandomForestRegressor
   from ml4pdm.transformation import AttributeFilter, UniToMultivariateWrapper, TimeSeriesImputer, PytsTransformWrapper, PytsSupportedAlgorithm, DatasetToSklearn
    
   pipeline = make_pipeline(AttributeFilter([0,1,2], 3), UniToMultivariateWrapper(make_pipeline(TimeSeriesImputer(), PytsTransformWrapper(PytsSupportedAlgorithm.BOSS))), DatasetToSklearn(), RandomForestRegressor(n_estimators=50, min_samples_leaf=4))

These are quite some steps, so here are their short explanations:

1. **AttributeFilter([0,1,2], 3)**
   
   We remove certain features, which for example don't include any useful information (settings-data) or which don't contain enough unique values for the later pipeline elements to work properly.

2. **UniToMultivariateWrapper**
   
   We wrap a pipeline which is then performed on each timeseries feature individually. 
   In this case we wrap the next two elements 3 and 4.

3. **TimeSeriesImputer**
      
   We align the lengths of the instances for the timeseries feature by filling in missing values for shorter instances.
   This is necessary as the pyts transformation algorithms only work with instances of same length.

4. **PytsTransformWrapper(PytsSupportedAlgorithm.BOSS))**
      
   We transform the timeseries feature to a fixed size vector by using the our wrapper for the transformation classes of the pyts library.

5. **DatasetToSklearn**
   
   We unwrap the Dataset object, such that the following (sklearn-)pipeline elements can work with the data directly.

6. **RandomForestRegressor(n_estimators=50, min_samples_leaf=4)**
   
   Sklearn model training.


Training and evaluation
-----------------------

For training and evaluation we use our own Evaluator class.
We also evaluate our simple pipeline with metrics from our library as well as regression metrics from sklearn::

   from ml4pdm.evaluation import Evaluator
   from ml4pdm.evaluation.metrics import loss_asymmetric, score_performance, loss_false_positive_rate, loss_false_negative_rate
   from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

   evaluator = Evaluator(None, [pipeline], None, [loss_asymmetric, mean_squared_error, score_performance, mean_absolute_error, mean_absolute_percentage_error, loss_false_positive_rate, loss_false_negative_rate])
   
   results = evaluator.evaluate_train_test_split(prepared_train_dataset, test_dataset)[0]
   
   for i in [2,4,5,6]:
      results[i] *= 100
   print("S:\t{:.2f}\nMSE:\t{:.2f}\nA(%):\t{:.2f}\nMAE:\t{:.2f}\nMAPE:\t{:.2f}\nFPR(%):\t{:.2f}\nFNR(%):\t{:.2f}".format(*results))

Two of the arguments in the constructor of our evaluator are None, as we don't evaluate multiple datasets and don't use a train-test splitting algorithm.
We also bring the metrics into similar ranges.
we look at the print ouput of that code::

   S:		1159.69
   MSE:		445.61
   A(%):	36.00
   MAE:		17.17
   MAPE:	42.08
   FPR(%):	23.00
   FNR(%):	41.00

These values are not bad for a very basic pipeline. With a Mean-Squared-Error (MSE) of around 400 we know that the mean deviation of the true value is around 20.
We can look at the exact predictions and compare them to the true values by using the automatically in our evaluator object stored predictions::

   for i, pred in enumerate(evaluator.full_y_pred_per_pipeline[0]):
      print(test_dataset.target[i], "\t", pred)

Looking at the individual examples we see a fairly reasonable prediction for the few example instances shown here.
The true value is left and the predicted value is right::
   
   112.0 	 136.20572486869855
   98.0 	 120.73779098679093
   69.0 	 63.226061771561746
   82.0 	 94.12502142302142
   91.0 	 98.01315301365304
   93.0 	 101.99774870554282
   ...

Configuration Files
-------------------

To create a pipeline that can work on cmapss data like we have above, you have to import a lot of different parts of our library.
The resulting pipeline constructor is therefore also long and cluttered. Using our library we have the option to save this pipeline in a .json file and load this already built pipeline again in another project.
Shown below is an example of how to save an already built pipeline::

   from ml4pdm.parsing import PipelineConfigParser
   import os

   PIPELINE_CONFIGURATION_FILENAME = os.path.join('.','PipelineConfiguration.json') 

   PipelineConfigParser.save_to_file(pipeline, path=PIPELINE_CONFIGURATION_FILENAME)

We can use the PipelineConfigParser to load this saved pipeline again::

   loaded_pipeline = PipelineConfigParser.parse_from_file(path=PIPELINE_CONFIGURATION_FILENAME)

This loaded_pipeline can be used to replace the pipeline in the evaluation scheme above and will work exactly like the fully built pipeline that we have used previously.

Other Pipeline options 
======================

The above shown example is completely built in the following notebook. The explanations are shown above:

`Simple Feature Extraction <naive_pipeline.html>`_

Here are also other pipeline options with explanations of their own

`Embed RUL <embed_rul.html>`_

`Multiple Classifier Approach <multiple_classifier_approach.html>`_

`Random Forest Approach <random_forest_approach.html>`_