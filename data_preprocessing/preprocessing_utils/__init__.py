# Import modules with error handling to prevent package import failures
__all__ = []

try:
    from .llm_data_mining import SkillExtractor
    __all__.append("SkillExtractor")
except ImportError:
    pass

try:
    from .model_manager import get_model
    __all__.append("get_model")
except ImportError:
    pass

try:
    from .mongo_utils import MongoImporter, backup_mongodb, JobIterator
    __all__.extend(["MongoImporter", "backup_mongodb", "JobIterator"])
except ImportError:
    pass

try:
    from .mongo_setup import setup_mongodb, close_mongodb_connection
    __all__.extend(["setup_mongodb", "close_mongodb_connection"])
except ImportError:
    pass

try:
    from .sentence2vec_utils import Sentence2VecEncoder
    __all__.append("Sentence2VecEncoder")
except ImportError:
    pass
