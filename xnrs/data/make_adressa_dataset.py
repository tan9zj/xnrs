from os.path import join    

from .adressa import AdressaHandler


ROOT_DIR = '../../data/adressa/ten_weeks'

DAYS = ['20170101', '20170102', '20170103', '20170104', '20170105', '20170106', '20170107', '20170108', '20170109', '20170110',
        '20170111', '20170112', '20170113', '20170114', '20170115', '20170116', '20170117', '20170118', '20170119', '20170120',
        '20170121', '20170122', '20170123', '20170124', '20170125', '20170126', '20170127', '20170128', '20170129', '20170130',
        '20170201', '20170202', '20170203', '20170204', '20170205', '20170206', '20170207', '20170208', '20170209', '20170210',
        '20170211', '20170212', '20170213', '20170214', '20170215', '20170216', '20170217', '20170218', '20170219', '20170220',
        '20170221', '20170222', '20170223', '20170224', '20170225', '20170226', '20170227', '20170228',
        '20170301', '20170302', '20170303', '20170304', '20170305', '20170306', '20170307', '20170308', '20170309', '20170310',
        '20170311', '20170312', '20170313', '20170314', '20170315', '20170316', '20170317', '20170318', '20170319', '20170320',
        '20170321', '20170322', '20170323', '20170324', '20170325', '20170326', '20170327', '20170328', '20170329', '20170330', '20170331']

AdressaHandler.extract_days(
    dir=join(ROOT_DIR, 'raw'), days=DAYS, dst_dir=join(ROOT_DIR, 'extracted_test')
)

AdressaHandler.make_daily_datasets(
    DAYS, 
    N_days=10, 
    src_dir=join(ROOT_DIR, 'extracted_test'),
    dst_dir=join(ROOT_DIR, 'daily_datasets_test')
    )

AdressaHandler.combine_daily_datasets(
    days=['20170325', '20170326', '20170327', '20170328', '20170329'],
    src_dir=join(ROOT_DIR, 'daily_datasets_test'),
    dst_path=join(ROOT_DIR, 'datasets_test', 'dev_training.csv')
)

AdressaHandler.combine_daily_datasets(
    days=['20170330'],
    src_dir=join(ROOT_DIR, 'daily_datasets_test'),
    dst_path=join(ROOT_DIR, 'datasets_test', 'dev_eval.csv')
)

AdressaHandler.combine_daily_datasets(
    days=['20170326', '20170327', '20170328', '20170329', '20170330'],
    src_dir=join(ROOT_DIR, 'daily_datasets_test'),
    dst_path=join(ROOT_DIR, 'datasets_test', 'val_training.csv')
)

AdressaHandler.combine_daily_datasets(
    days=['20170331'],
    src_dir=join(ROOT_DIR, 'daily_datasets_test'),
    dst_path=join(ROOT_DIR, 'datasets_test', 'val_eval.csv')
)

# for the accuracy of our attributions, it is crucial to to set the arument relative_to_reference=True (cf. Section 4.3 of the paper)
_ = AdressaHandler.embed_news(
    days=DAYS, 
    src_dir=join(ROOT_DIR, 'extracted'),
    dst_path=join(ROOT_DIR, 'datasets_test', 'all_news_norbert_ref.pkl'),
    model_name='ltg/norbert3-base',
    relative_to_reference=True 
    )