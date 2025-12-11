# help me write documentation so that sphinx can generate it

"""
Core module for the rm_gen package.

This module defines the core classes and functionalities for building
a modular pipeline system, including steps for data processing and
feature acquisition.

Classes:

- Step: Base class for a pipeline step.

- FeatureAcquisitionStep: Specialized step for acquiring features.

- Pipeline: Class for managing and executing a sequence of steps.
"""

from abc import ABC
import logging
import shutil
import typing
from pathlib import Path

from .log_utils import _LoggerWithTQDM


class Step(ABC):
    """
    Base class for a pipeline step.
    Each step can prepare its context, check for cached results,
    compute new results, and load cached results if available.

    Each step is responsible for defining its own cache keys,
    how to parse the context it receives, how to compute results,
    and how to store and load cached results.

    Attributes:
        _computed (bool): Flag indicating if the step has been computed.
        _cache_dir (Path): Directory for caching results.
        _logger (_LoggerWithTQDM): Logger for logging messages.
    """
    _computed: bool = False
    _cache_dir: Path = None
    _logger: _LoggerWithTQDM = None

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory for the step.

        Returns
        -------
        Path
            The cache directory for the step.
        """
        return self._cache_dir

    @property
    def logger(self) -> Path:
        """Get the cache directory for the step.

        Returns
        -------
        Path
            The cache directory for the step.
        """
        return self._logger

    def set_cache_dir(self, cache_dir: Path, append_clname: bool = True):
        """Set the cache directory for the step.

        Parameters
        ----------
        cache_dir: Path
            The base directory for storing cache.
        append_clname: bool, optional
            Whether to append the class name to the cache directory.
            Defaults to True.
        """
        if append_clname:
            cache_dir = Path(cache_dir) / self.__class__.__name__

        self._cache_dir = Path(cache_dir)

    def set_logger(self, logger):
        """Set the logger for the step.
        Parameters
        ----------
        logger: logging.Logger
            The logger to be used by the step.
        """
        self._logger = logger

    def cache_dirs(self, keys) -> list[Path]:
        """Get the cache directories for the given keys.
        
        Parameters
        ----------
        keys: list[str]
            List of cache keys.

        Returns
        -------
        list[Path]
            List of paths corresponding to the cache keys.
        """
        return [
            self._cache_dir / k for k in keys
        ]

    def has_complete_cache(self, keys):
        """Check if the cache is complete for the given keys.

        Parameters
        ----------
        keys: list[str]
            List of cache keys.

        Returns
        -------
        bool
            True if all cache directories or files exist, False otherwise.

        Notes
        -----
        This function is used to determine if the step needs to recompute
        its results or if it can load them from the cache.
        """
        if len(keys) == 0:
            return False

        return all(c.exists() for c in self.cache_dirs(keys))

    def clean_cache(self, ctx):
        """Clean the cache for the given context.

        Parameters
        ----------
        ctx: typing.Any
            The context, as determined by the parse_ctx method,
            to clean the cache for.
        """
        for f in self.cache_dirs(self.cache_keys(ctx)):
            if f.exists():
                if f.is_dir():
                    shutil.rmtree(f)
                else:
                    f.unlink()

    def cache_keys(self, ctx):
        """Get the cache keys for the given context.

        Parameters
        ----------
        ctx: typing.Any
            The context, as determined by the parse_ctx method,
            to get the cache keys for.

        Returns
        -------
        list[str]
            List of cache keys.
        """
        return []

    def prepare(self, ctx):
        """Prepare the step with the given context.

        Parameters
        ----------
        ctx: typing.Any
            The context, as determined by the parse_ctx method,
            to prepare the step with.

        Notes
        -----
        This method can be overridden by subclasses to perform any
        necessary preparation before any step evaluation. This method
        is called before any others, e.g. before checking if the cache
        is complete.
        """
        return

    def load_cache(self, keys):
        """"Load the cached results for the given keys.
        Parameters
        ----------
        keys: list[str]
            List of cache keys.
        Returns
        -------
        dict
            The cached results.
        """
        raise NotImplementedError()

    def parse_ctx(self, raw_ctx) -> typing.Any:
        """"Parse the raw context into the format required by the step.
        Parameters
        ----------
        raw_ctx: dict
            The raw context to parse.
        Returns
        -------
        typing.Any
            The parsed context.
        """
        raise NotImplementedError()

    def compute(self, ctx):
        """Compute the results for the given context.

        Parameters
        ----------
        ctx: typing.Any
            The context, as determined by the parse_ctx method,
            to compute the results for.

        Returns
        -------
        dict
            The computed results.
        """
        raise NotImplementedError()

    def run(self, raw_ctx) -> None:
        """Run the step with the given raw context.

        Parameters
        ----------
        raw_ctx: dict
            The raw context to run the step with.

        Returns
        -------
        dict
            The updated context after running the step.
        """
        ctx = self.parse_ctx(raw_ctx)
        self.prepare(ctx)

        keys = self.cache_keys(ctx)
        if self.has_complete_cache(keys):
            self._logger.debug("Loading return cache")
            data = self.load_cache(keys)
        else:
            for d in self.cache_dirs(keys):
                d.parent.mkdir(parents=True, exist_ok=True)
            data = self.compute(ctx)

        self._logger.debug("Adding " + str(list(data.keys())) + " to context")
        for k in data:
            if k in raw_ctx:
                self._logger.error(
                    f"Key {k} already exists in context: " +
                    str(list(data.keys()))
                )
                raise ValueError(
                    f"Key {k} already exists in context"
                )

        return {**raw_ctx, **data}


class FeatureAcquisitionStep(Step):
    """
    Specialized step for acquiring features.
    Attributes:
        feature_name (str): Name of the feature being acquired.

    An example of a feature acquisition step could be::

        class MyFeatureStep(FeatureAcquisitionStep):
            def __init__(self, dataset_dir: Path):
                super().__init__(dataset_dir, feature_name="my_feature")

            def parse_ctx(self, raw_ctx):
                # check if context has required data
                if "required_data" not in raw_ctx:
                    raise ValueError("Context must contain 'required_data'")
                # Parse the context here
                return parsed_ctx

            def compute(self, ctx):
                required_data = ctx["required_data"]
                # Compute the feature using required_data
                computed_feature = ...
                # return the computed feature to be added to the context
                # it is impo
                return {self.feature_name: computed_feature}
    """
    feature_name: str

    def __init__(self, dataset_dir: Path, feature_name: str):
        super().__init__()
        self.feature_name = feature_name
        self.set_cache_dir(dataset_dir, append_clname=False)

    # By default, clean cache before computing features
    def prepare(self, ctx):
        self.clean_cache(ctx)


class Pipeline():
    """
    Class for managing and executing a sequence of steps.

    Attributes:
        _steps (list[Step]): List of steps in the pipeline.
        _logger (_LoggerWithTQDM): Logger for logging messages.
        _cache_dir (Path): Directory for caching results.
        _feature_names (list[str]): List of feature names acquired
        during the run.

    A pipeline consists of multiple steps that are executed in order::

        pipeline = Pipeline([
            StepA(),
            FeatureAcquisitionStepB(),
            StepC(),
        ])
        pipeline.set_cache_dir(Path("./cache"))
        result = pipeline.run()

        another_pipeline = Pipeline([
            StepD(),
            FeatureAcquisitionStepE(),
        ])
        another_pipeline.set_cache_dir(Path("./another_cache"))
        # you may pass the context from the previous pipeline run
        # effectively chaining pipelines
        another_result = another_pipeline.run(result["context"])
    
    Notes
    -----
    The pipeline logger and cache directory can be set for the entire pipeline,
    and will be propagated to each step unless the step already has its own
    logger or cache directory set.
    """
    # true private static variable
    # pylint: disable=invalid-name
    __default_logger = None

    _steps: list[Step]
    _logger: _LoggerWithTQDM
    _cache_dir: Path
    _feature_names: list[str]

    def __init__(self, steps):
        """"Initialize the Pipeline with a list of steps.

        Parameters
        ----------
        steps: list[Step]
            List of steps to be executed in the pipeline.
        """
        self._steps: list[Step] = steps
        self._logger = self.get_default_logger()
        self._cache_dir = Path("./rm-pipeline-cache")
        self._feature_names = []

    @staticmethod
    def from_pipeline(pipeline: "Pipeline") -> "Pipeline":
        """Create a new Pipeline instance from an existing one.

        Parameters
        ----------
        pipeline: Pipeline
            The existing pipeline to copy.

        Returns
        -------
        Pipeline
            A new Pipeline instance with the same steps, logger,
            cache directory, and feature names.
        """
        # pylint: disable=protected-access
        p = Pipeline(pipeline._steps)
        p.set_logger(pipeline._logger)
        p.set_cache_dir(pipeline._cache_dir)
        p._feature_names = pipeline._feature_names.copy()
        return p

    @classmethod
    def get_default_logger(cls):
        """"Get the default logger used by the pipelines."""
        if cls.__default_logger is not None:
            return cls.__default_logger

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        cls.__default_logger = _LoggerWithTQDM(logger)
        return cls.__default_logger
    
    @classmethod
    def set_default_logger(cls, logger: logging.Logger):
        """"
        Set the default logger used by the pipelines created posteriorly.

        Parameters
        ----------
        logger: logging.Logger
            The logger to be used as the default logger.
        
        Notes
        -----
        This method sets the default logger for all pipelines
        created after this call. Existing pipelines are not affected.

        Be careful when using this method in any kind of complex application,
        as it may lead to unexpected logging behavior.
        """
        cls.__default_logger = _LoggerWithTQDM(logger)

    def set_logger(self, logger: logging.Logger):
        """Wraps the provided logger and sets it for the pipeline.

        Parameters
        ----------
        logger: logging.Logger
            The logger to be used by the pipeline.
        """
        self._logger = _LoggerWithTQDM(logger)
        for step in self._steps:
            step.set_logger(self._logger)

    def set_logger_level(self, level: int | str):
        """Set the logging level for the pipeline's logger.
        Parameters
        ----------
        level: int | str
            The logging level to set (e.g., logging.INFO, "DEBUG").
        """
        self._logger.setLevel(level)

    def set_cache_dir(self, cache_dir: Path, overwrite: bool = False):
        """Set the cache directory for the pipeline.

        Parameters
        ----------
        cache_dir: Path
            The base directory for storing cache.
        overwrite: bool, optional
            Whether to force steps to use the passed cache directory.
            Defaults to False.
        """
        self._cache_dir = Path(cache_dir)
        for step in self._steps:
            if step.cache_dir is None or overwrite:
                step.set_cache_dir(str(self._cache_dir))

    def add_step(self, step: Step):
        """Add a step to the pipeline.
        Parameters
        ----------
        step: Step
            The step to be added to the pipeline.
        """
        self._steps.append(step)

    def run(self, context=None) -> dict:
        """Run the pipeline with the given context.

        Parameters
        ----------
        context: dict, optional
            The initial context to run the pipeline with.
            Defaults to an empty dictionary.

        Returns
        -------
        dict
            A dictionary containing the final context and
            the list of feature names acquired during the run.

        Notes
        -----
        Each step in the pipeline will be executed in order.
        If a step does not have a logger or cache directory set,
        it will inherit the pipeline's logger and cache directory.

        The returned dictionary has the following structure::

            {
                "context": final_context,
                "feature_names": list_of_feature_names,
            }

        where `final_context` is the context after all steps have been executed,
        and `list_of_feature_names` is a list of feature names acquired
        during the pipeline run.
        """
        if context is None:
            context = {}

        if self._logger is None:
            self._logger = self.get_default_logger()

        with self._logger.tqdm_progress_bar(
            self._steps,
            desc="Pipeline Steps"
        ) as steps_progress_bar:
            for step in steps_progress_bar:
                self._logger.info("Starting step " + step.__class__.__name__)

                if step.logger is None:
                    step.set_logger(self._logger)
                if step.cache_dir is None:
                    step.set_cache_dir(self._cache_dir)

                if isinstance(step, FeatureAcquisitionStep):
                    self._logger.debug(
                        "This step is a feature acquisition step"
                    )
                    self._feature_names.append(step.feature_name)

                # step.clean_cache(context)
                context = step.run(context)

        return {
            "context": context,
            "feature_names": self._feature_names,
        }
