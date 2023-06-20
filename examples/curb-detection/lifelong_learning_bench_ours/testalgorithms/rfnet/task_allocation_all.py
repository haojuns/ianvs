import re
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

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

    def __call__(self, task_extractor, samples: BaseDataSource):
        self.task_extractor = task_extractor
        print(self.task_extractor)
        if self.default_meta_attr == "all":
            return samples, [int(self.task_extractor.get(self.default_meta_attr))] * len(samples.x)
        # allocations = [int(self.task_extractor.get(sample)) for sample in samples.x]

        # return samples, allocations
