import time
import os 
from enum import Enum
from typing import List,Dict,Tuple
import pandas as pd
import numpy as np
import tqdm
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

ROOT_CLASSIFICATION_CAT="hf://datasets/inria-soda/tabular-benchmark/clf_cat/"
ROOT_CLASSIFICATION_NUM="hf://datasets/inria-soda/tabular-benchmark/clf_num/"
ROOT_REGRESSION_CAT="hf://datasets/inria-soda/tabular-benchmark/reg_cat/"
ROOT_REGRESSION_NUM="hf://datasets/inria-soda/tabular-benchmark/reg_num/"

BENCH_CLASSIFICATION_NUM=["bank-marketing","california","credit"]
BENCH_CLASSIFICATION_CAT=["eye_movements","default-of-credit-card-clients","road-safety","covertype","albert"]
BENCH_REGRESSION_CAT=["SGEMM_GPU_kernel_performance","Brazilian_houses","Mercedes_Benz_Greener_Manufacturing","seattlecrime6","Allstate_Claims_Severity"	]
BENCH_REGRESSION_NUM=["wine_quality","Bike_Sharing_Demand","diamonds","house_sales","medical_charges"]
SEED=99


class EnumBenchmark(Enum):
    CLASSIFICATION="CLASSIFICATION"
    REGRESSION="REGRESSION"
    

class TabularBenchmark:
    classification_metrics:List[str]=["accuracy","f1_weighted","roc_auc_ovr_weighted"]
    regression_metrics:List[str]=["neg_root_mean_squared_error","r2"]
    limit_df_size :int =5_000 # limit the size of the dataset to avoid memory issues
    le_encoder=LabelEncoder()
    def __init__(self,task :EnumBenchmark,models: List[Tuple[str,BaseEstimator]],seed : int,result_path:str)->None:
        self.task =task
        self.models =models
        self.seed: int =seed
        self.result_path =result_path
        self.metrics: List[str] =self.classification_metrics if task==EnumBenchmark.CLASSIFICATION else self.regression_metrics
    def run(self)->None:
        results=self.run_benchmark()
        self.save_results(results)
        self.report_results(results)
        return results

    def process_df(self,df:"pd.DataFrame")->List["pd.DataFrame"]:
        if len(df)>self.limit_df_size:
            frac=self.limit_df_size/len(df)
            df=df.sample(frac=frac)
        target_col=df.columns[-1]
        df.rename(columns={target_col:"target"},inplace=True)
        X=df.drop(columns=["target"])
        y=df["target"]
        if self.task==EnumBenchmark.CLASSIFICATION:
            y=self.le_encoder.fit_transform(y)
        return X,y


    def run_expirement(self,dataset:"pd.DataFrame")->List[Dict[str,float]]:
        df_name=dataset.split("/")[-1].split(".")[0]
        X,y=self.process_df(pd.read_csv(dataset))
        scores=[]
        for model in self.models:
            print(f"---------- Running model {model[0]}----------")
            for metric in self.metrics:
                try:
                    start=time.time()
                    score=cross_val_score(model[1],X,y,scoring=metric,cv=5,n_jobs=-1).mean()
                    end=time.time()
                    print(f"Model {model[0]} metric {metric} score {score:.2f} in {end-start:.2f} seconds")
                    scores.append({"dataset":df_name,"model":model[0],"metric":metric,"score":round(score,3),"time":round(end-start,3)})
                except Exception as e:
                    print(f"Model {model[0]} metric {metric} failed with error {e}")
                    scores.append({"dataset":df_name,"model":model[0],"metric":metric,"score":np.nan,"time":np.nan})
        return scores

    @staticmethod
    def load_datasets(root:str,benchmark:List[str])->List['pd.DataFrame']:
        datasets=[]
        for dataset in benchmark:
            datasets.append(root+dataset+".csv")
        return datasets


    def run_benchmark(self)->List[Dict[str,float]]:
        if self.task==EnumBenchmark.CLASSIFICATION:
            datasets=self.load_datasets(ROOT_CLASSIFICATION_NUM,BENCH_CLASSIFICATION_NUM)+self.load_datasets(ROOT_CLASSIFICATION_CAT,BENCH_CLASSIFICATION_CAT)
        elif self.task==EnumBenchmark.REGRESSION:
            datasets=self.load_datasets(ROOT_REGRESSION_NUM,BENCH_REGRESSION_NUM)+self.load_datasets(ROOT_REGRESSION_CAT,BENCH_REGRESSION_CAT)
        else:
            raise ValueError("Invalid benchmark")
        results=[]
        for dataset in tqdm.tqdm(datasets):
            dataset_name=dataset.split("/")[-1].split(".")[0]
            print(f"----------Runing dataset  {dataset_name} ---------- ")
            result=self.run_expirement(dataset)
            results.extend(result)
        return results
    
    def report_results(self,results:List[Dict[str,float]]):
        df = pd.DataFrame(results).dropna()  # Remove rows with NaN values
        for metric in self.metrics:
            print(f"-----------Metric {metric}----------")
            df_metric = df[df["metric"] == metric]
            plt.figure(figsize=(12, 8))
            sns.barplot(data=df_metric, x="dataset", y="score", hue="model")
            plt.title(f"Benchmark using {metric}")
            plt.xticks(rotation=90)
            
            file_path = os.path.join(self.result_path, f"benchmark_{metric}.png")
            plt.savefig(file_path)  # Save plot as an image
            plt.show()  # Show the plot
                
        
    def save_results(self,results:List[Dict[str,float]])->None:
        df=pd.DataFrame(results)
        os.makedirs(self.result_path,exist_ok=True)
        df_path=self.result_path+"/results_df.csv"
        df.to_csv(df_path,index=False)
        print(f"Results saved to {df_path}")
    
