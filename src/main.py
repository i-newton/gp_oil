from src.utils import get_test, get_train
from src.pipelines import common_pipeline, train_hook, target_hook, \
    cont_pipeline, cat_pipeline, null_cat_step, text_pipeline, date_pipeline
from src.cleaner import DataCleaner

dc = DataCleaner(common_pipelines=common_pipeline, column_pipelines=[
    cont_pipeline, null_cat_step, cat_pipeline, text_pipeline, date_pipeline],
                 train_hooks=train_hook, target_hooks=target_hook)

dc.get_clean_data(get_train(), get_test(), group_col="Скважина")
