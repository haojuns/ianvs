import re
from typing import List, Any, Tuple

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.seen_task_learning.artifact import Task

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

        if "task_embeddings" in kwargs:
            task_embeddings = kwargs["task_embeddings"]
            task_names = []
            for i in range(self.splits):
                task_name = "task" + str(i)
                task_names.append(task_name)
            self.origins = task_names

        import pdb
        pdb.set_trace()

        tasks = []
        d_type = samples.data_type
        x_data = samples.x
        y_data = samples.y
        
        task_index = dict(zip(self.origins, range(len(self.origins)))) # {'task0': 0, 'task1': 1, 'task2': 2}

        """
        for程: task_embeddings是一个二维数组, axis_y是图片名称, axis_x是该图片对应的表征向量(256维), 且这个图片顺序和x_data的顺序一致
        程请首先加入k-means++算法进行数据划分, 然后参考task_definition_by_origin.py中对cityscapes的流程, 进行返回对应的tasks, task_index, samples
        """

        return tasks, task_index, samples
