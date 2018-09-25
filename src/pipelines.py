from collections import Iterable
import pandas as pd

from sklearn.preprocessing import StandardScaler


class PipelineStep:
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, train, test, *args, **kwargs):
        if isinstance(self.tf, Iterable):
            for step in self.tf:
                train, test = step(train, test)
            return train, test
        return self.tf(train, test)


class Hook:
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, item, *args, **kwargs):
        if isinstance(self.tf, Iterable):
            for step in self.tf:
                item = step(item)
            return item
        return self.tf(item)


class ColumnPipelineStep(PipelineStep):

    def __init__(self, columns, tf):
        super(ColumnPipelineStep, self).__init__(tf)
        self.cols = columns

    def __call__(self, train, test, *args, **kwargs):
        train = train[self.cols]
        test = test[self.cols]
        return super(ColumnPipelineStep, self).__call__(train, test)


# sort by converted date and group
def convert_and_sort(df):
    df["Дата"] =  df["Дата"].apply(pd.to_datetime)
    return df.sort_values(by=["Скважина", "Дата"])


train_hook = Hook(tf=convert_and_sort)


def get_non_useful(df):
    non_useful_columns = []
    for c in df.columns:
        null_columns = df[df[c].isnull()]
        if len(null_columns) == len(df):
            non_useful_columns.append(c)
    return non_useful_columns


def drop_non_useful(train, test):
    non_useful = set(get_non_useful(train)) | set(get_non_useful(test))
    print("%s dropped"% non_useful)
    return train.drop(list(non_useful), axis=1), test.drop(list(non_useful), axis=1)


drop_constants = PipelineStep(tf=drop_non_useful)


# drop non present columns in test
def drop_not_present(train, test):
    absent_columns = list(set(train.columns) - set(test.columns))
    print("%s dropped" % absent_columns)
    return train.drop(absent_columns, axis=1), test


drop_not_present_test_train = PipelineStep(tf=drop_not_present)

common_pipeline = PipelineStep(tf=[drop_constants, drop_not_present_test_train])


def get_float(v):
    v = str(v)
    if v != "NaN":
        new = v.replace(",",".")
        return float(new)
    return v


def get_target(df, column="Нефть, т"):
    target = df[column]
    print("%s dropped"% column)
    return df.drop([column], axis=1), target.apply(get_float)


target_hook = Hook(tf=get_target)


def get_text_cols():
    text = ["Причина простоя"]
    return text


def get_cat_cols():
    categorical = ["Тип испытания",
                   "Тип скважины",
                   "Неустановившийся режим",
                   "ГТМ",
                   "Метод",
                   "Характер работы",
                   "Состояние",
                   "Пласт МЭР",
                   "Способ эксплуатации",
                   "Тип насоса",
                   "Состояние на конец месяца",
                   "Номер бригады",
                   "Фонтан через насос",
                   "Нерентабельная",
                   "Назначение по проекту",
                   "Группа фонда",
                   "Тип дополнительного оборудования",
                   "Марка ПЭД",
                   "Тип ГЗУ",
                   "ДНС",
                   "КНС",
                   # "Агент закачки",
                   # text converted
                   "Мероприятия",
                   "Проппант",
                   "Куст",
                   "Причина простоя.1",
                   'ПЛАСТ'
                   ]
    return categorical


def get_date_cols():
    dates = ["Дата",
             "Дата_2",
             "Дата ГРП",
             "Время до псевдоуст-ся режима",
             "Дата запуска после КРС",
             "Дата пуска",
             "Дата останова",
             "Дата ввода в эксплуатацию"]
    return dates


def get_coord_cols():
    coords = ["УСТЬЕ_X", "УСТЬЕ_Y", "ПЛАСТ_X", "ПЛАСТ_Y"]
    return coords


def get_cont_cols():
    all_cols = [
        # "Скважина",
        "Дата",
        "ГТМ",
        "Метод",
        "Характер работы",
        "Состояние",
        "Время работы, ч",
        "Время накопления",
        "Попутный газ, м3",
        "Закачка, м3",
        "Природный газ, м3",
        "Газ из газовой шапки, м3",
        "Конденсат, т",
        "Простой, ч",
        "Причина простоя",
        "Приемистость, м3/сут",
        "Обводненность (вес), %",
        "Дебит конденсата",
        "Добыча растворенного газа, м3",
        "Дебит попутного газа, м3/сут",
        "Пласт МЭР",
        "Куст",
        "Тип скважины",
        "Диаметр экспл.колонны",
        "Диаметр НКТ",
        "Диаметр штуцера",
        "Глубина верхних дыр перфорации",
        "Удлинение",
        "Способ эксплуатации",
        "Тип насоса",
        "Производительность ЭЦН",
        "Напор",
        "Частота",
        "Коэффициент сепарации",
        "Глубина спуска",
        "Буферное давление",
        "Давление в линии",
        "Пластовое давление",
        "Динамическая высота",
        "Затрубное давление",
        "Давление на приеме",
        "Забойное давление",
        "Обводненность",
        "Состояние на конец месяца",
        "Давление наcыщения",
        "Газовый фактор",
        "Температура пласта",
        "SKIN",
        "JD факт",
        "Дата ГРП",
        "Вязкость нефти в пластовых условиях",
        "Вязкость воды в пластовых условиях",
        "Вязкость жидкости в пласт. условиях",
        "объемный коэффициент",
        "Плотность нефти",
        "Плотность воды",
        "Высота перфорации",
        "Удельный коэффициент",
        "Коэффициент продуктивности",
        "ТП - Забойное давление",
        "ТП - JD опт.",
        "ТП - SKIN",
        "К пр от стимуляции",
        "Глубина спуска.1",
        "КВЧ",
        "Время до псевдоуст-ся режима",
        "Причина простоя.1",
        "Дата запуска после КРС",
        "Дата пуска",
        "Дата останова",
        "Радиус контура питания",
        "Мероприятия",
        "Номер бригады",
        "Фонтан через насос",
        "Нерентабельная",
        "Неустановившийся режим",
        "Дата ввода в эксплуатацию",
        "Назначение по проекту",
        "Замерное забойное давление",
        "Группа фонда",
        "Нефтенасыщенная толщина",
        "Плотность раствора глушения",
        "Глубина текущего забоя",
        "Тип дополнительного оборудования",
        "Диаметр дополнительного оборудования",
        "Глубина спуска доп. оборудования",
        "Марка ПЭД",
        "Мощность ПЭД",
        "I X/X",
        "Ток номинальный",
        "Ток рабочий",
        "Число качаний ШГН",
        "Длина хода плунжера ШГН",
        "Диаметр плунжера",
        "Коэффициент подачи насоса",
        "Тип ГЗУ",
        "ДНС",
        "КНС",
        "КН закрепленный",
        "Пластовое давление начальное",
        "Характеристический дебит жидкости",
        "Время в работе",
        "Время в накоплении",
        "ГП - Забойное давление",
        "ГП(ИДН) Дебит жидкости",
        "ГП(ИДН) Дебит жидкости скорр-ый",
        "ГП(ИДН) Прирост дефита нефти",
        "ГП(ГРП) Дебит жидкости",
        "ГП(ГРП) Дебит жидкости скорр-ый",
        "Наклон",
        "Азимут",
        "k",
        "Ноб",
        "Нэф",
        "Pпл",
        "Верх",
        "Низ",
        "Xf",
        "Hf",
        "Wf",
        "JD",
        "FCD",
        "М пр",
        "Проппант",
        "Рпл Хорнер",
        "Эфф",
        "Конц",
        "Гель",
        "М бр",
        "V под",
        "V гель",
        "Давление пластовое",
        "Тип испытания",
        "ПЛАСТ",
        "УСТЬЕ_X",
        "УСТЬЕ_Y",
        "ПЛАСТ_X",
        "ПЛАСТ_Y",
        "Альтитуда",
    ]
    continious = list(set(all_cols) - set(get_date_cols()) - set(get_cat_cols())
                      - set(get_text_cols()) - set(get_coord_cols()))
    return continious


def get_object_columns(df):
    objects = []
    for c in df.columns:
        if df[c].dtype != pd.np.float:
            objects.append(c)
    return objects


def convert_locale_to_float(df):
    loc_float = get_object_columns(df)
    converted = df.copy()
    for c in loc_float:
        converted.loc[:,c] = df[c].apply(get_float)
    return converted


def convert_train_test(train, test):
    return convert_locale_to_float(train), convert_locale_to_float(test)


to_float_step = PipelineStep(tf=convert_train_test)


def fill_with_median(train, test):
    means = train.median()
    norm_train = train.fillna(means)
    norm_test = test.fillna(means)
    return norm_train, norm_test


median_step = PipelineStep(tf=fill_with_median)


# now we have clear non-normalized data, let's normalize first
def normalize(train, test):
    scaler = StandardScaler()
    norm_train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index = train.index)
    norm_test = pd.DataFrame(scaler.transform(test), columns=test.columns, index = test.index)
    return norm_train, norm_test


normalize_step = PipelineStep(tf=normalize)


def null_cat(train, test):
    return train.isnull().astype(int).add_suffix('_indicator'), test.isnull().astype(int).add_suffix('_indicator')


null_cat_step = PipelineStep(tf=null_cat)


cont_pipeline = ColumnPipelineStep(columns=get_cont_cols(),
                                   tf=[to_float_step, median_step, normalize_step, null_cat_step])


def get_one_hot(train, test):
    for c in train.columns:
        train.loc[c,:] = train[c].astype(str)
        test.loc[c,:] = test[c].astype(str)
    train_oh = pd.get_dummies(train, drop_first=True)
    test_oh = pd.get_dummies(test, drop_first=True)
    test_oh = test_oh.reindex(columns=train_oh.columns, fill_value=0)
    print(train_oh.isnull().values.any() or test_oh.isnull().values.any())
    return train_oh, test_oh


cat_pipeline = ColumnPipelineStep(columns=get_cat_cols(), tf=get_one_hot)


