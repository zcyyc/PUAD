import numpy as np
import pandas as pd
import multiprocessing
import random
import os
import config
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
import time
import json
from sklearn.externals import joblib
from collections import OrderedDict, Counter
from examples.plotutils import delay_eva
from sklearn.ensemble import RandomForestClassifier
from frameworks.CPLELearning import CPLELearningModel
from frameworks.PULearning import PULearningModel


class ExperimentSettings:

    def __init__(self, al_name, init_label_count, new_kpi_train_ratio, if_save_label_index=False):
        id = self.get_id()
        self.al_name = al_name + id
        self.init_label_count = init_label_count
        self.new_kpi_train_ratio = new_kpi_train_ratio
        self.if_save_label_index = if_save_label_index
        self.init_path(self.al_name)
        self.get_dataset_config()
        self.init_label()
    
    def get_id(self):
        return time.strftime("-%Y-%m-%d-%H:%M:%S", time.localtime()) 

    def init_path(self, al_name):
        logging.info("Initializes the results of experiment.")
        result_root = os.path.join(config.DATA_ROOT, "anomalyresult")
        if(not(os.path.exists(result_root))):
            os.mkdir(result_root)
        # The directory where the model files are stored.
        model_root = os.path.join(result_root, "model")
        if(not(os.path.exists(model_root))):
            os.mkdir(model_root)
        self.model_root = os.path.join(model_root, al_name)
        if(not(os.path.exists(self.model_root))):
            os.mkdir(self.model_root)
        # The directory where the probabilities are stored.
        proba_root = os.path.join(result_root, "proba")
        if(not(os.path.exists(proba_root))):
            os.mkdir(proba_root)
        self.proba_root = os.path.join(proba_root, al_name)
        if(not(os.path.exists(self.proba_root))):
            os.mkdir(self.proba_root)
        # The directory where the index with labeled are stored.
        if self.if_save_label_index == False:
            self.index_root = None
            return
        index_root = os.path.join(result_root, "index")
        if(not(os.path.exists(index_root))):
            os.mkdir(index_root)
        self.index_root = os.path.join(index_root, al_name)
        if(not(os.path.exists(self.index_root))):
            os.mkdir(self.index_root)

    def get_dataset_config(self):
        config_dataset_root = os.path.join(config.DATA_ROOT, 'config.json')
        logging.info("Get the configuration of dataset.")
        with open(config_dataset_root) as load_f:
            load_dict = json.load(load_f)
            centroids_dict = load_dict[u'centroids']
        self.dataset_config = {}
        for key in centroids_dict.keys():
            self.dataset_config[key.encode('utf-8')] = centroids_dict[key].encode('utf-8')

    def init_label(self):
        logging.info("Label some anomalies randomly.")
        self.centroids = {}
        for cluster_name in self.dataset_config.keys():
            centroid_name = self.dataset_config[cluster_name]
            centroids = CentroidKPI(cluster_name, centroid_name)
            centroids.random_label_anomaly(self.init_label_count, self.index_root)
            self.centroids[centroid_name] = centroids

    def reset_label_with_index(self):
        for centroid in self.centroids.values():
            centroid.reset_label


class KPI:

    def __init__(self, cluster_name, file_name):
        df = pd.read_csv(os.path.join(os.path.join(config.DATA_ROOT, cluster_name), file_name))
        feature_name = [i for i in df.columns if i.startswith("F#")]
        self.kpi_name = file_name
        self.features = df[feature_name].values.copy()
        self.labels = df['label'].values.copy()
        self.real_labels = self.labels.copy()


class CentroidKPI(KPI):

    def random_label_anomaly(self, label_count, index_path):
        anomaly_index = np.where(self.labels == 1)[0]
        if label_count > len(anomaly_index):
            label_count = anomaly_index
        np.random.shuffle(anomaly_index)
        self.labels[:] = -1
        self.labels[anomaly_index[:label_count]] = 1
        self.index = anomaly_index[:label_count]
        if index_path:
            self.output_index(index_path, anomaly_index[:label_count])

    def output_index(self, index_path, index):
        np.savez(os.path.join(index_path, self.kpi_name), index)

    def reset_label(self):
        self.labels[:] = -1
        self.labels[index] = 1


class NewKPI(KPI):

    def CPLE(self, centroid, train_ratio, model_root, if_labeled=False):
        self.get_data(train_ratio, if_labeled)
        if self.labels.sum() == 0:
            return None
        self.concat_centroid(centroid)
        model = CPLELearningModel(basemodel=RandomForestClassifier(config.RF_n_trees, n_jobs=15), max_iter=50,
                                    predict_from_probabilities=True, real_label=None)
        logging.info("Start CPLE learning process of " + self.kpi_name + ".")
        print Counter(self.train_label)
        model.fit(self.train_data, self.train_label)
        if model_root != None:
            self.save_model(model, model_root)
        proba = model.predict_proba(self.test_data)
        proba = proba[:, 1]
        eva = delay_eva(self.test_label, proba)
        _, best_threshold = eva.best_fscore_threshold()
        logging.info("Threshold of " + self.kpi_name + " is " + str(best_threshold) + ".")
        fscore = eva.fscore_for_threshold(best_threshold)
        logging.info("F-score of " + self.kpi_name + " is " + str(fscore) + ".")
        return fscore

    def save_model(self, model_root):
        joblib.dump(model, os.path.join(model_root, self.kpi_name + '.sav'))
        logging.info("Save model of " + self.kpi_name + ".")

    def get_data(self, train_ratio, if_labeled):
        train_lenth = int(len(self.features) * train_ratio)
        self.train_data = self.features[:train_lenth]
        self.train_label = self.labels[:train_lenth]
        self.test_data = self.features[train_lenth:]
        self.test_label = self.labels[train_lenth:]
        if if_labeled is False:
            self.train_label[:] = -1
    
    def concat_centroid(self, centroid):
        self.train_data = np.concatenate((centroid.features.copy(), self.train_data), axis=0)
        self.train_label = np.concatenate((centroid.labels.copy(), self.train_label), axis=0)


class Experiment():

    def __init__(self, settings):
        self.settings = settings
        self.save_model = False
        self.pretrain()
        self.train_model()

    def pretrain(self):
        pass

    def train_model(self):
        fscore_results = []
        kpi_names = []
        pool = multiprocessing.Pool(processes=32)
        for cluster_name in self.settings.dataset_config.keys():
            centroid_name = self.settings.dataset_config[cluster_name]
            for kpi_name in os.listdir(os.path.join(config.DATA_ROOT, cluster_name)):
                if kpi_name != centroid_name:
                    fscore_results.append(pool.apply_async(launch_CPLE, (cluster_name,
                        self.settings.centroids[centroid_name], kpi_name, self.settings, self.save_model)))
                    kpi_names.append((cluster_name, kpi_name))
        pool.close()
        pool.join()
        fscore_results = [fscore.get() for fscore in fscore_results]
        # for res in fscore_results:
            # print (res.get())
        for name, fscore in zip(kpi_names, fscore_results):
            print name, fscore



class PUADExperiment(Experiment):

    def pretrain(self):
        pool = multiprocessing.Pool(processes=16)
        result_labels = []
        for centroid in self.settings.centroids.values():
            result_labels.append(pool.apply_async(PU, (centroid, )))
        pool.close()
        pool.join()
        for i in range(len(result_labels)):
            self.settings.centroids.values()[i].labels = result_labels[i].get()


class SupervisedExperiment(Experiment):
    def __init__(self, settings):
        Experiment.__init__(self, settings)
        for centroid in self.settings.centroids:
            centroid.labels = centroid.real_labels

    def launch_kpi(self, centroid, cluster_name, kpi_name):
        new_kpi = NewKPI(cluster_name, kpi_name)
        model_root = None if self.save_model == False else self.settings.model_root
        return new_kpi.CPLE(centroid, self.settings.new_kpi_train_ratio, model_root, True)


def PU(centroid):
    logging.info("Starting PU learning.")
    real_labels = centroid.real_labels
    PU_model = PULearningModel(centroid.features, centroid.labels)
    PU_model.pre_training(0.2)
    RF_model = RandomForestClassifier(n_estimators=100)
    semi_labels, _ = PU_model.add_reliable_samples_using_RandomForest(0.015, 200, 0.7, centroid.real_labels, RF_model)
    logging.info("PU learning of centroid curve was completed.")
    return semi_labels


def launch_CPLE(cluster_name, centroid, kpi_name, settings, if_save_model):
    kpi_path = os.path.join(config.DATA_ROOT, cluster_name, kpi_name)
    new_kpi = NewKPI(cluster_name, kpi_name)
    model_root = None if if_save_model == False else settings.model_root
    return new_kpi.CPLE(centroid, settings.new_kpi_train_ratio, model_root, False)
    logging.info("F-score of " + kpi_name + ":" + str(fscore))