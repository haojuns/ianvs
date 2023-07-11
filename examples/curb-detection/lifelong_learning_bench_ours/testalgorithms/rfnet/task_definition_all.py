import re
from typing import List, Any, Tuple

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.seen_task_learning.artifact import Task

from sklearn.cluster import KMeans
import numpy as np

__all__ = ('TaskDefinitionAll',)


@ClassFactory.register(ClassType.STP, alias="TaskDefinitionAll")
class TaskDefinitionAll:
    """
    Dividing datasets based on the their origins.

    Parameters
    ----------
    origins: List[Metadata]
        metadata is usually a class feature label with a finite values.
    """

    def __init__(self, **kwargs):
        self.origins = kwargs.get("origins", ["all"])
        self.splits = kwargs.get("splits")

    def __call__(self,
                 samples: BaseDataSource, **kwargs) -> Tuple[List[Task],
                                                             Any,
                                                             BaseDataSource]:
        # tasks = []

        # g_attr = "semantic_segamentation_model"
        # task_obj = Task(entry=g_attr, samples=samples, meta_attr="all")
        # tasks.append(task_obj)

        # task_index = {"all": 0}
        #
        if "task_embeddings" in kwargs:
            task_embeddings = kwargs["task_embeddings"]
            task_names = []
            for i in range(self.splits):
                task_name = "task" + str(i)
                task_names.append(task_name)
            self.origins = task_names

        tasks = []
        d_type = samples.data_type
        x_data = samples.x
        y_data = samples.y
        
        task_index = dict(zip(self.origins, range(len(self.origins)))) # {'task0': 0, 'task1': 1, 'task2': 2}
        
        #创建数据类
        style_df=[]
        for i in range(self.splits):
            style_df.append(BaseDataSource(data_type=d_type))
            style_df[i].x, style_df[i].y=[], []
            
        import pdb
        #pdb.set_trace()

        #聚类
        embedding_prototype, embedding_index = self.embedding_mining(task_embeddings)
        
        #分配数据
        for i in range(samples.num_examples()):
            style_df[embedding_index[i]].x.append(x_data[i])
            style_df[embedding_index[i]].y.append(y_data[i])

        #创建任务
        tasks=[]
        for i in range(self.splits):
            g_attr='style_'+str(i)+'_model'
            task_obj = Task(entry=g_attr, samples=style_df[i], meta_attr=embedding_prototype[i])
            tasks.append(task_obj)


        return tasks, task_index, samples
    
    def embedding_mining(self, task_embeddings):
        # 提取图片的向量和路径
        vectors, _ = zip(*task_embeddings)

        # 使用 KMeans++ 进行聚类
        kmeans = KMeans(n_clusters=self.splits, init='k-means++', random_state=0).fit(vectors)

        # 获取聚类结果
        embedding_prototype = kmeans.cluster_centers_
        embedding_index = kmeans.labels_

        # 返回聚类中心向量和每张图片的所属类
        return embedding_prototype, embedding_index

