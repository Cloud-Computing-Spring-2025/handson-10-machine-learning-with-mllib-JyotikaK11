import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# Create output directory and clear output file
os.makedirs("output", exist_ok=True)
output_path = "output/final_output.txt"
with open(output_path, "w") as f:
    f.write("Customer Churn Prediction with Apache Spark MLlib\n\n")

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
df = spark.read.csv("customer_churn.csv", header=True, inferSchema=True)

# Function to write to output file
def write_output(text):
    with open(output_path, "a") as f:
        f.write(text + "\n")

# Task 1: Preprocessing
def preprocess_data(df):
    write_output("========== Task 1: Data Preprocessing and Feature Engineering ==========\n")

    df = df.fillna({"TotalCharges": 0})

    cat_cols = ["gender", "PhoneService", "InternetService", "Churn"]
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}Index") for col in cat_cols]
    ohe_cols = ["genderIndex", "PhoneServiceIndex", "InternetServiceIndex"]
    encoders = [OneHotEncoder(inputCol=col, outputCol=f"{col}Vec") for col in ohe_cols]

    feature_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] + [f"{col}Vec" for col in ohe_cols]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")

    pipeline = Pipeline(stages=indexers + encoders + [assembler, label_indexer])
    processed_df = pipeline.fit(df).transform(df)

    rows = processed_df.select("features", "label").limit(5).collect()
    output = "+--------------------+-----------+\n"
    output += "|features            |ChurnIndex |\n"
    output += "+--------------------+-----------+\n"
    for row in rows:
        output += f"|{str(row['features'])[:20]:20}|{row['label']:>11} |\n"
    output += "+--------------------+-----------+\n"
    write_output(output)

    return processed_df.select("features", "label")

# Task 2: Logistic Regression
def train_logistic_regression_model(df):
    write_output("========== Task 2: Train and Evaluate Logistic Regression Model ==========\n")

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    model = LogisticRegression(featuresCol="features", labelCol="label").fit(train)
    predictions = model.transform(test)

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    sample = predictions.select("label", "prediction", "probability").limit(5).collect()
    output = "Sample Predictions:\n"
    output += "+-------+-----------+------------------------------------------+\n"
    output += "|Label  |Prediction |Probability                               |\n"
    output += "+-------+-----------+------------------------------------------+\n"
    for row in sample:
        output += f"|{row['label']:>6} |{int(row['prediction']):>10} |{str(row['probability'])[:40]:<40}|\n"
    output += "+-------+-----------+------------------------------------------+\n"
    output += f"\nLogistic Regression Model Accuracy (AUC): {auc:.2f}\n"
    write_output(output)

# Task 3: Chi-Square Feature Selection
def feature_selection(df):
    write_output("========== Task 3: Feature Selection using Chi-Square Test ==========\n")

    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    result_df = selector.fit(df).transform(df)

    rows = result_df.select("selectedFeatures", "label").limit(5).collect()
    output = "+--------------------+-----------+\n"
    output += "|selectedFeatures    |ChurnIndex |\n"
    output += "+--------------------+-----------+\n"
    for row in rows:
        output += f"|{str(row['selectedFeatures'])[:20]:20}|{row['label']:>11} |\n"
    output += "+--------------------+-----------+\n"
    write_output(output)

# Task 4: Hyperparameter Tuning & Model Comparison
def tune_and_compare_models(df):
    write_output("========== Task 4: Hyperparameter Tuning and Model Comparison ==========\n")

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

    models = {
        "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="label"),
        "DecisionTree": DecisionTreeClassifier(featuresCol="features", labelCol="label"),
        "RandomForest": RandomForestClassifier(featuresCol="features", labelCol="label"),
        "GBT": GBTClassifier(featuresCol="features", labelCol="label")
    }

    param_grids = {
        "LogisticRegression": ParamGridBuilder().addGrid(models["LogisticRegression"].regParam, [0.01, 0.1]).build(),
        "DecisionTree": ParamGridBuilder().addGrid(models["DecisionTree"].maxDepth, [5, 10]).build(),
        "RandomForest": ParamGridBuilder().addGrid(models["RandomForest"].numTrees, [10, 20]).build(),
        "GBT": ParamGridBuilder().addGrid(models["GBT"].maxIter, [10, 20]).build()
    }

    model_results = []

    for name, model in models.items():
        write_output(f"Tuning {name}...")
        cv = CrossValidator(estimator=model,
                            estimatorParamMaps=param_grids[name],
                            evaluator=evaluator,
                            numFolds=5)
        cv_model = cv.fit(train)
        best_model = cv_model.bestModel
        auc = evaluator.evaluate(best_model.transform(test))

        result = {
            "name": name,
            "auc": auc,
            "params": {param.name: val for param, val in best_model.extractParamMap().items()
                       if param.name in ["regParam", "maxDepth", "numTrees", "maxIter"]}
        }
        model_results.append(result)

        output = f"{name} Best Model Accuracy (AUC): {auc:.2f}\n"
        output += f"Best Params for {name}:\n"
        for k, v in result["params"].items():
            output += f"  {k} = {v}\n"
        output += "\n"
        write_output(output)

    # Final Model Selection
    best = max(model_results, key=lambda x: x["auc"])
    summary = f"🏆 Best Model Overall: {best['name']} with AUC = {best['auc']:.2f}\n"
    summary += "Best Hyperparameters:\n"
    for k, v in best["params"].items():
        summary += f"  {k} = {v}\n"

    write_output("========== Final Model Selection ==========\n")
    write_output(summary)

# Run all tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

spark.stop()