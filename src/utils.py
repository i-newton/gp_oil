import pandas as pd


def show_uniq_test_train(train, test):
    # check all values that have zero ans nan only
    for c in sorted(train.columns):
        un = train[c].unique()
        if len(un)<100:
            tun = test[c].unique()
            print("%s ;train: %s; test:%s"%(c, un, tun))


def get_train():
    train_main = pd.read_csv("../data/task1/train_1.8.csv", encoding="cp1251")
    train_aux_coords = pd.read_csv("../data/task1_additional/coords_train_1.1.csv", encoding="cp1251")
    train_aux_frac = pd.read_csv("../data/task1_additional/frac_train_1.csv", encoding="cp1251")
    train_aux_gdis = pd.read_csv("../data/task1_additional/gdis_train1.2.csv", encoding="cp1251")

    train_frac_main = pd.merge(train_main, train_aux_frac, how="left", left_on="Скважина", right_on="Скважина")
    train_main_frac_gdis = pd.merge(train_frac_main, train_aux_gdis, how="left", left_on="Скважина",
                                    right_on="Скважина")
    all_recs = pd.merge(train_main_frac_gdis, train_aux_coords, how="left", left_on="Скважина", right_on="well_hash")
    final_recs = all_recs.drop(["well_hash"], axis=1)
    return final_recs


def get_test():
    test_main = pd.read_csv("../data/task1/test_1.9.csv", encoding="cp1251")
    test_aux_coords = pd.read_csv("../data/task1_additional/coords_test_1.1.csv", encoding="cp1251")
    test_aux_frac = pd.read_csv("../data/task1_additional/frac_test_1.csv", encoding="cp1251")
    test_aux_gdis = pd.read_csv("../data/task1_additional/gdis_test1.2.csv", encoding="cp1251")

    test_frac_main = pd.merge(test_main, test_aux_frac, how="left", left_on="Скважина", right_on="Скважина")
    test_main_frac_gdis = pd.merge(test_frac_main, test_aux_gdis, how="left", left_on="Скважина", right_on="Скважина")
    all_recs = pd.merge(test_main_frac_gdis, test_aux_coords, how="left", left_on="Скважина", right_on="well_hash")
    final_recs = all_recs.drop(["well_hash"], axis=1)
    return final_recs


def get_existed(columns, df):
    return list(set(columns)&set(df.columns))


def split_continious_date_categorical_text(df):
    group_id = ["Скважина"]

    exclude_cont = []
    """'Ток номинальный', 'Приемистость, м3/сут',
       'Глубина верхних дыр перфорации', 'Пластовое давление начальное', 'Низ',
       'I X/X', 'Обводненность (вес), %', 'ГП - Забойное давление',
       'ТП - Забойное давление', 'FCD', 'Простой, ч', 'М пр', 'JD',
       'Буферное давление', 'Мощность ПЭД', 'Обводненность',
       'Пластовое давление', 'М бр', 'Глубина спуска',
       'Производительность ЭЦН', 'JD факт', 'Рпл Хорнер', 'К пр от стимуляции',
       'Xf', 'Закачка, м3', 'Давление в линии', 'Диаметр НКТ',
       'ГП(ГРП) Дебит жидкости', 'ГП(ГРП) Дебит жидкости скорр-ый', 'Эфф',
       'Напор', 'Верх', 'Азимут', 'Диаметр экспл.колонны', 'Ток рабочий',
       'Затрубное давление', 'Hf', 'Wf',
                   "Дата ввода в эксплуатацию",
                   "Дата запуска после КРС" ,
                   "Диаметр плунжера",
                   "Природный газ, м3",
                   "Конденсат, т",
                   "Длина хода плунжера ШГН",
                   "Коэффициент подачи насоса",
                   "Дебит конденсата",
                   "Вязкость воды в пластовых условиях",
                   "Газ из газовой шапки, м3",
                   "Число качаний ШГН",
                   "Группа фонда",
                   "Фонтан через насос",
                   "Неустановившийся режим",
                   "Закачка, м3",
                   "ГП(ИДН) Прирост дефита нефти",
                   "Вязкость нефти в пластовых условия",
                   "Закачка, м3",
                   "ГП(ИДН) Дебит жидкости скорр-ый",
                   ]
                   """

    continious = list(set(df.columns) - set(dates) - set(categorical)
                      - set(text) - set(group_id) - set(coords) - set(exclude_cont))
    return (df[group_id], df[continious], df[get_existed(dates, df)], df[get_existed(categorical, df)],
            df[get_existed(text, df)], df[get_existed(coords, df)])


def get_fold():
    return KFold(n_splits = 4,shuffle=True, random_state = 17)