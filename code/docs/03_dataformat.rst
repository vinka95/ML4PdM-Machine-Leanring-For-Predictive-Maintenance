Data Format
===========

We define a general data format that can be used for Predictive Maintenance data sets. 
Our *Predictive Maintenance File Format (PdMFF)* is based on the `Attribute Relation File Format (ARFF) <https://www.cs.waikato.ac.nz/ml/weka/arff.html>`_ that is commonly used for machine learning data.

PdMFF
-----

PdMFF is is a data format that describes a list of instances for a set of sensors. PdMFF is divided into Header section and Data section. In the header section all the attributes are declared and the data section represents the actual data.
Before we define the keywords in header and data sections, we need to define some basic placeholders.

Placeholders
------------

<name>
    This is a string used to specify a meaningful name for a keyword.

<attribute-type> 
    This is used to specify the type of value the **@ATTRIBUTE** keyword holds.

<target-type> 
    This is used to specify the type of value the **@TARGET** keyword holds.

<timestep-type> 
    This is used to specify the data type of timestep. This is used only in **TIMESERIES** attribute, described later in the chapter.


Header section
--------------
This section consists of metadata for the data that is stored. This information describes different sensors, the type of value recorded relation between the data using keywords such as **@RELATION**, **@ATTRIBUTE** and **@TARGET**.

@RELATION
   This statement defines a relation between the different attributes (sensors) by specifying a name.


Format
    
    @RELATION <name>

Example

.. code-block:: none
   
    @RELATION ml4pdm

Each parameter in the dataset is associated with an attribute and is declared using **@ATTRIBUTE** statement. 
It also defines a meaning full name using <name> and its associated data type <attribute-type>.


An <attribute-type> can take following values:

- **NUMERIC**  is used to specify all numeric parameters. Attribute of this type can take values that are integers, float values etc.

.. code-block:: none

    %Format
    @ATTRIBUTE  <name>  NUMERIC

    %Example
    %Header Instance
    @ATTRIBUTE  sensor1 NUMERIC
    
    @DATA 
    %instance 1
    12.1


- An attribute can take one of the labels from a set of labels (**Nominal**). Attribute is restricted to only the *k* labels in the set.

.. code-block:: none

    %Format
    @ATTRIBUTE  <name>  {lab_1,..,lab_k}  
    
    %Example
    %Header Instance
    @ATTRIBUTE sensor2 {A,B,C}
    
    @DATA
    %instance 1
    A

- **DATETIME** is used to specify values of format MM/DD/YYYY-HHMMSSsssss.

.. code-block:: none

    %Format
    @ATTRIBUTE <name> DATETIME
    
    %Example
    %Header Instance
    @ATTRIBUTE sensor3 DATETIME

    @DATA
    %instance 1
    02.12.2004.10.32.39

- An attribute whose recorded value is a timeseries can be represented using **TIMESERIES** attribute-type. A timeseries is defined using tuples encapsulated within { }. Each tuple then has <timestep-type> (data type of time step; NUMERIC, DATETIME) and <attribute-type>.

.. code-block:: none

    %Format
    @ATTRIBUTE   <name>    TIMESERIES(<timestep-type>:<attribute-type>)
    
    %Example
    %Header Instance
    @ATTRIBUTE sensor4 TIMESERIES(NUMERIC:NUMERIC)
    
    @DATA
    %instance 1
    (1:0.0023,2:-0.0027,3:0.0003,....,31:-0.0006)

- An attribute with multidimensional data is specified using **MULTIDIMENSIONAL** keyword along with their dimensions encapsulated in [ ].

.. code-block:: none

    %Format
    @ATTRIBUTE   <name>    MULTIDIMENSIONAL[dim_1, dim_2,..,dim_n]
    
    %Example
    %Header Instance
    @ATTRIBUTE sensor5 MULTIDIMENSIONAL[4]

    @DATA
    %instance 1
    [12, 23, 46, 78 ]

**@TARGET** 
    This statement is used to specify a target variable and the type of value stored is defined using a <target-type>. Target type can be NUMERIC or value can be chosen from a set of values {value1, value2, value3}.

.. code-block:: none

    %Format 
    @TARGET  <name>	<target-type>
    
    %Example
    %Header Instance
    @ATTRIBUTE  sensor1  NUMERIC
    @TARGET  class   {A, B, C}

    @DATA
    %instance 1
    12.54#A


Data section
------------

This section of the file contains the actual data recorded at different time instances. Data section begins with **@DATA** statement. Each instance of data is contained in one row, containing all the attribute values which are delimited by a set of separators.

**Separators**
    **:**       is used to separate attributes within a timeseries tuple, {timestep:value}.

    **,**       is used to separate timeseries tuples.

    **%**       is used to comment lines.
