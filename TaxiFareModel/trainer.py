from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.data import clean_data, get_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
import mlflow
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import AdaBoostRegressor

EXPERIMENT_NAME = "[GB] [London] [ValentinosM] TaxiFareModel version 1.0"
MLFLOW_URI = "https://mlflow.lewagon.ai/"


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
        ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),('linear_model', LinearRegression())])
        self.pipeline = pipe


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X_train,self.y_train)
        self.mlflow_log_param('model','Linear')

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric('rmse',rmse)
        print(rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline,'model.joblib')



if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    trainer = Trainer(X_train,y_train)
    trainer.run()
    trainer.evaluate(X_val,y_val)
    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{experiment_id}")
    my_model = trainer.save_model()
    # models = [SGDRegressor(),AdaBoostRegressor()]
    # for model in models:
