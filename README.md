Fake job predicion with Machine Learning 
==============================

# Job Posting Fraud Detection

This repository contains the code and analysis for a machine learning project aimed at detecting fraudulent job postings. The project utilizes various machine learning algorithms to differentiate between legitimate and fraudulent job listings, leveraging a dataset comprising numerous features related to job postings.

## Project Overview

The digital transformation of the job market has brought convenience to job seekers and employers alike. However, it has also opened the door to fraudulent job postings, posing risks and undermining trust in online job portals. This project seeks to address this challenge by applying machine learning techniques to identify patterns and indicators of fraudulent job postings effectively.

## Dataset

The dataset used in this project includes 17,880 job postings, with each entry annotated with attributes such as job title, location, industry, employment type, salary range, and a binary indicator of the posting's legitimacy. 

Key Attributes:
- `title`: The title of the job posting.
- `location`: The location where the job is based.
- `industry`: The industry sector the job falls under.
- `salary_range`: The advertised salary range.
- `employment_type`: The type of employment (e.g., full-time, part-time).
- `required_experience`: The experience level required for the job.
- `required_education`: The educational qualifications required for the job.
- `fraudulent`: Indicates if the posting is fraudulent (1 for fraudulent, 0 for legitimate).

## Methodology

### Exploratory Data Analysis (EDA)

The project begins with an EDA to understand the dataset's characteristics, distribution of job postings across different industries, and the prevalence of fraudulent postings.

### Feature Engineering

Several new features were engineered to enhance the model's predictive capability, including keyword presence in job descriptions, benefits count, and encoded categorical variables like industry and employment type.

### Model Training and Evaluation

Multiple machine learning models were trained and evaluated, including:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

The models were assessed based on their accuracy and F1-score, with the Random Forest Classifier emerging as the most effective in detecting fraudulent job postings.

### Feature Importance

An analysis of feature importance was conducted to identify the most significant predictors of fraudulent job postings, providing insights into the key indicators of fraud.

## Installation

To run this project, you will need Python 3.8 or later and the following packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the dependencies using pip:

## Usage

To replicate the analysis or apply the models to new data, follow these steps:
1. Clone this repository.
2. Ensure that you have the required Python packages installed.
3. Run the Jupyter notebooks in the `notebooks` directory, starting with `1_eda.ipynb` followed by `2_model_training.ipynb`.

## Contributing

Contributions to this project are welcome! If you have suggestions for improvements or want to contribute code, please feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
