import re
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

from scipy.stats import wasserstein_distance

__all__ = ('TaskAllocationAll',)


@ClassFactory.register(ClassType.STP, alias="TaskAllocationAll")
class TaskAllocationAll:
    """
    Corresponding to `TaskDefinitionByOrigin`

    Parameters
    ----------
    task_extractor : Dict
        used to match target tasks
    origins: List[Metadata]
        metadata is usually a class feature
        label with a finite values.
    """

    def __init__(self, **kwargs):
        self.default_meta_attr = kwargs.get("meta_attr", "all")

    def __call__(self, task_extractor, samples: BaseDataSource, **kwargs):
        
        import pdb 
        #pdb.set_trace()
        
        allocations=[]
        
        if "task_embeddings" in kwargs:
            tasks_embeddings = kwargs["task_embeddings"] # 测试样本表征

        if "seen_models" in kwargs:
            seen_models = kwargs["seen_models"]
            self.splits = len(seen_models)
            task_names = []
            meta_attrs = [] # 任务平均表征
            for i in range(self.splits):
                task_name = "task" + str(i)
                task_names.append(task_name)
                # 获取任务平均表征
                meta_attrs.append(seen_models[i].meta_attr[0])
            self.origins = task_names
            self.task_extractor = dict(zip(self.origins, range(len(self.origins))))
            
            # 任务平均表征和测试样本表征的距离度量并选取最小距离作为任务序号
            for i in range(len(samples.x)):
                distances = []
                min_distance = 1
                for j in range(self.splits):
                    # 计算任务表征的Wasserstein距离
                    distances.append(wasserstein_distance(tasks_embeddings[i][0], meta_attrs[j]))
                    if distances[j] < min_distance:
                        min_distance = distances[j]
                        min_distance_num = j
                allocations.append(min_distance_num)

            return samples, allocations
            
        else:
            self.task_extractor = {'all': 0}#task_extractor
            return samples, [int(self.task_extractor.get(self.default_meta_attr))] * len(samples.x)
        

