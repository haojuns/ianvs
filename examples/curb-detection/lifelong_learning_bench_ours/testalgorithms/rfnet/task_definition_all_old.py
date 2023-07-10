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

    def __call__(self,
                 samples: BaseDataSource, **kwargs) -> Tuple[List[Task],
                                                             Any,
                                                             BaseDataSource]:
        tasks = []

        g_attr = "semantic_segamentation_model"
        task_obj = Task(entry=g_attr, samples=samples, meta_attr="all")
        tasks.append(task_obj)

        task_index = {"all": 0}

        return tasks, task_index, samples
