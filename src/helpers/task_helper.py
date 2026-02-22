from relbench.base import TaskType

def get_task_metadata(task, dataset_name, task_name):
    # Different metric for imbalance dataset/task
    if dataset_name == "rel-avito" and task_name in ["user-clicks"]:
        out_channels = 1
        tune_metric = "f1"
        higher_is_better = True
    elif dataset_name == "rel-stack" and task_name in ["user-engagement", "user-badge"]:
        out_channels = 1
        tune_metric = "f1"
        higher_is_better = True
    else:   # Default metric
        if task.task_type == TaskType.REGRESSION:
            out_channels = 1
            tune_metric = "mae"
            higher_is_better = False
        elif task.task_type == TaskType.BINARY_CLASSIFICATION:
            out_channels = 1
            tune_metric = "roc_auc"
            higher_is_better = True
        elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            out_channels = task.n_classes
            assert out_channels is not None and out_channels > 1, "Number of classes must be greater than and not None"
            tune_metric = "accuracy"
            higher_is_better = True
        else:
            # Self-Supervised Learning
            if "ssl" in task_name:
                out_channels = None
                tune_metric = None
                higher_is_better = None
            # Task Vector Estimation
            elif "tve" in task_name:
                out_channels = len(task.target_col)
                tune_metric = None
                higher_is_better = None
            else:
                raise ValueError(f"Task {task_name} is not supported")
    return {
        "out_channels": out_channels,
        "tune_metric": tune_metric,
        "higher_is_better": higher_is_better,
    }

####################################################################################################
#                                   RelBench Task Helper                                           #
####################################################################################################

import os
    
from collections import defaultdict
from functools import lru_cache
from typing import List

import pooch

from relbench.base import BaseTask
from relbench.tasks import hashes
from src.tasks import amazon, avito, event, f1, hm, stack, trial

from src.definitions import DATA_DIR
from src.helpers.relbench_helper import get_dataset

task_registry = defaultdict(dict)

DOWNLOAD_REGISTRY = pooch.create(
    path=os.path.join(DATA_DIR, "relbench"),
    base_url="https://relbench.stanford.edu/download/",
    registry=hashes,
)


def register_task(
    dataset_name: str,
    task_name: str,
    cls: BaseTask,
    *args,
    **kwargs,
) -> None:
    r"""Register an instantiation of a :class:`BaseTask` subclass with the given name.

    Args:
        dataset_name: The name of the dataset.
        task_name: The name of the task.
        cls: The class of the task.
        args: The arguments to instantiate the task.
        kwargs: The keyword arguments to instantiate the task.

    The name is used to enable caching and downloading functionalities.
    `cache_dir` is added to kwargs by default. If you want to override it, you
    can pass `cache_dir` as a keyword argument in `kwargs`.
    """

    cache_dir = f"{os.path.join(DATA_DIR, 'relbench')}/{dataset_name}/tasks/{task_name}"
    kwargs = {"cache_dir": cache_dir, **kwargs}
    task_registry[dataset_name][task_name] = (cls, args, kwargs)


def get_task_names(dataset_name: str) -> List[str]:
    r"""Return a list of names of the registered tasks for the given dataset."""
    return list(task_registry[dataset_name].keys())


def download_task(dataset_name: str, task_name: str) -> None:
    r"""Download task from RelBench server into its cache directory.

    The downloaded task tables will be automatically picked up by the task object, when
    `task.get_table(split)` is called.
    """

    DOWNLOAD_REGISTRY.fetch(
        f"{dataset_name}/tasks/{task_name}.zip",
        processor=pooch.Unzip(extract_dir="."),
        progressbar=True,
    )


@lru_cache(maxsize=None)
def get_task(dataset_name: str, task_name: str, download=False) -> BaseTask:
    r"""Return a task object by name.

    Args:
        dataset_name: The name of the dataset.
        task_name: The name of the task.
        download: If True, download the task from the RelBench server.

    Returns:
        BaseTask: The task object.

    If `download` is True, the task tables (train, val, test) comprising the
    task will be downloaded into the cache from the RelBench server. If you use
    `download=False` the first time, the task tables will be computed from
    scratch using the database.

    Once the task tables are cached, either because of download or computing from
    scratch, the cache will be used. `download=True` will verify that the
    cached task tables matches the RelBench version even in this case.
    """

    if download:
        download_task(dataset_name, task_name)
    dataset = get_dataset(dataset_name)
    cls, args, kwargs = task_registry[dataset_name][task_name]
    task = cls(dataset, *args, **kwargs)
    return task


register_task("rel-amazon", "user-churn", amazon.UserChurnTask)
register_task("rel-amazon", "user-ltv", amazon.UserLTVTask)
register_task("rel-amazon", "item-churn", amazon.ItemChurnTask)
register_task("rel-amazon", "item-ltv", amazon.ItemLTVTask)
register_task("rel-amazon", "user-item-purchase", amazon.UserItemPurchaseTask)
register_task("rel-amazon", "user-item-rate", amazon.UserItemRateTask)
register_task("rel-amazon", "user-item-review", amazon.UserItemReviewTask)

register_task("rel-avito", "ad-ctr", avito.AdCTRTask)
register_task("rel-avito", "user-visits", avito.UserVisitsTask)
register_task("rel-avito", "user-clicks", avito.UserClicksTask)
register_task("rel-avito", "user-ad-visit", avito.UserAdVisitTask)

register_task("rel-event", "user-attendance", event.UserAttendanceTask)
register_task("rel-event", "user-repeat", event.UserRepeatTask)
register_task("rel-event", "user-ignore", event.UserIgnoreTask)

register_task("rel-f1", "driver-position", f1.DriverPositionTask)
register_task("rel-f1", "driver-dnf", f1.DriverDNFTask)
register_task("rel-f1", "driver-top3", f1.DriverTop3Task)

register_task("rel-hm", "user-item-purchase", hm.UserItemPurchaseTask)
register_task("rel-hm", "user-churn", hm.UserChurnTask)
register_task("rel-hm", "item-sales", hm.ItemSalesTask)

register_task("rel-stack", "user-engagement", stack.UserEngagementTask)
register_task("rel-stack", "post-votes", stack.PostVotesTask)
register_task("rel-stack", "user-badge", stack.UserBadgeTask)
register_task("rel-stack", "user-post-comment", stack.UserPostCommentTask)
register_task("rel-stack", "post-post-related", stack.PostPostRelatedTask)

register_task("rel-trial", "study-outcome", trial.StudyOutcomeTask)
register_task("rel-trial", "study-adverse", trial.StudyAdverseTask)
register_task("rel-trial", "site-success", trial.SiteSuccessTask)
register_task("rel-trial", "condition-sponsor-run", trial.ConditionSponsorRunTask)
register_task("rel-trial", "site-sponsor-run", trial.SiteSponsorRunTask)


# New Tasks
register_task("rel-amazon", "user-ssl", amazon.UserSSLTask)
register_task("rel-amazon", "user-tve-1-hop", amazon.UserTVE1HopTask)
register_task("rel-amazon", "user-tve-2-hop", amazon.UserTVE2HopTask)
register_task("rel-amazon", "user-churn-top100-spending", amazon.UserChurnTop100SpendingTask)
register_task("rel-amazon", "user-churn-top50-spending", amazon.UserChurnTop50SpendingTask)
register_task("rel-amazon", "user-ltv-bad-reviews", amazon.UserLTVBadReviewsTask)

register_task("rel-amazon", "item-ssl", amazon.ItemSSLTask)
register_task("rel-amazon", "item-tve-1-hop", amazon.ItemTVE1HopTask)
register_task("rel-amazon", "item-tve-2-hop", amazon.ItemTVE2HopTask)
register_task("rel-amazon", "item-churn-least100-spending", amazon.ItemChurnLeast100SpendingTask)
register_task("rel-amazon", "item-churn-least50-spending", amazon.ItemChurnLeast50SpendingTask)
register_task("rel-amazon", "item-ltv-least100-spending", amazon.ItemLTVLeast100SpendingTask)

register_task("rel-hm", "user-ssl", hm.UserSSLTask)
register_task("rel-hm", "user-tve-1-hop", hm.UserTVE1HopTask)
register_task("rel-hm", "user-tve-2-hop", hm.UserTVE2HopTask)
register_task("rel-hm", "user-churn-top100-spending", hm.UserChurnTop100SpendingTask)
register_task("rel-hm", "user-churn-top50-spending", hm.UserChurnTop50SpendingTask)
register_task("rel-hm", "item-ssl", hm.ItemSSLTask)
register_task("rel-hm", "item-tve-1-hop", hm.ItemTVE1HopTask)
register_task("rel-hm", "item-tve-2-hop", hm.ItemTVE2HopTask)
register_task("rel-hm", "item-sales-top200-spending", hm.ItemSalesTop200SpendingTask)