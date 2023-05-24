import os
import sys
import traceback
import pandas as pd
import numpy as np
import inspect
from datetime import datetime
from tqdm import tqdm
import chinese_calendar
import alpha101_factors
import alpha191_factors
import CICC_factors
from factors_neut import rext_mad, stand_norm, neut_reg

sys.path.insert(0, r"..\FactorsTools")
from DataManager import DataManager, PriceVolumeData
from Tools import bday_shift, delay, ts_sum, index2stock, get_citic_sector, get_citic_sector_name


class PriceVolumeDataLoc(PriceVolumeData):
    """PriceVolumeData类的定位"""

    def __init__(self, price_data):
        self.obj = price_data

    def __getattr__(self, item):
        return getattr(self.obj, item)

    def __getitem__(self, item):
        # 将实例对象作为可切片元素返回切片结果
        import inspect
        import copy
        new_price_data = copy.deepcopy(self.obj)
        data_lst = [mem for mem in inspect.getmembers(self.obj, lambda x: isinstance(x, (pd.Series, pd.DataFrame)))]
        for val_name, val in data_lst:
            setattr(new_price_data, val_name, val.loc[item])
        return new_price_data


def data_nan_adjust(*datas):
    """将PriceVolumeData数据进行对nan值计算的调整"""
    import copy

    def pandas_rollings_adj(value=1):
        """更改pandas.groupby.rolling类型的默认参数

        当不需要太过于关注一些值，例如数据起始期间：window的值，股票上市时间：window的值时，可以采用这种方式获得更多因子值；
        当需要更准确的初始值，不能用这种方法
        """
        import pandas
        from functools import partialmethod
        pandas.core.groupby.DataFrameGroupBy.rolling = partialmethod(pandas.core.groupby.DataFrameGroupBy.rolling,
                                                                     min_periods=value)
        pandas.core.groupby.DataFrameGroupBy.expanding = partialmethod(pandas.core.groupby.DataFrameGroupBy.expanding,
                                                                       min_periods=value)
        pandas.core.groupby.DataFrameGroupBy.ewm = partialmethod(pandas.core.groupby.DataFrameGroupBy.ewm,
                                                                 min_periods=value)
        pandas.core.groupby.SeriesGroupBy.rolling = partialmethod(pandas.core.groupby.SeriesGroupBy.rolling,
                                                                  min_periods=value)
        pandas.core.groupby.SeriesGroupBy.expanding = partialmethod(pandas.core.groupby.SeriesGroupBy.expanding,
                                                                    min_periods=value)
        pandas.core.groupby.SeriesGroupBy.ewm = partialmethod(pandas.core.groupby.SeriesGroupBy.ewm,
                                                              min_periods=value)
        return pandas

    pandas = pandas_rollings_adj()  # 这个是全局性的改变
    new_data_lst = list()
    for data in datas:
        new_data = copy.deepcopy(data)
        dataframe_lst = [mem for mem in inspect.getmembers(new_data, lambda x: isinstance(x, pd.DataFrame))]
        for val_name, val in dataframe_lst:
            setattr(new_data, val_name, pandas.DataFrame(val))
        series_lst = [mem for mem in inspect.getmembers(new_data, lambda x: isinstance(x, pd.Series))]
        for val_name, val in series_lst:
            setattr(new_data, val_name, pandas.Series(val))
        new_data_lst.append(new_data)
    return new_data_lst


def ordinary_factor_get(factor, start_date, end_date, shift_n, shift_multi=1):
    """通常因子数据计算

    Returns
    -------------
    pd.Series
        因子数据
    """
    shift_n_mul = int(shift_n * shift_multi)
    tmp_stock_data = stock_data[bday_shift(start_date, -shift_n_mul): end_date]
    tmp_f = factor(tmp_stock_data,
                   index_data[bday_shift(start_date, -shift_n_mul): end_date],
                   [])
    nan_shift = tmp_stock_data.close.groupby(level=1).head(1)
    nan_shift.loc[:] = bday_shift(nan_shift.index.get_level_values(0).to_series(), shift_n).tolist()
    tmp_f = tmp_f.to_frame("value").swaplevel(0, 1).join(nan_shift.droplevel(0).rename("shift_dt"))
    tmp_f.loc[tmp_f.index.get_level_values(1) < tmp_f["shift_dt"], "value"] = np.nan
    tmp_f = tmp_f["value"].swaplevel(0, 1).sort_index()
    if len(tmp_f.unique()) < 5:
        print("check", factor.__name__, sep=" ")

    return tmp_f.loc[start_date:end_date]


def CICC_cyq_factor_get(cyq_factor_names, start_date, end_date, cyq_dis):
    """基于筹码分布数据的因子数据计算

    Parameters
    --------------
    cyq_factor_names: list
        筹码分布因子名list
    cyq_dis: pd.DataFrame
        筹码分布df

    Returns
    -------------
    dict
        因子数据字典
    """
    class CyqFactorsDaily:

        def __init__(self, cyq):
            self.cyq_dis = cyq
            self.ret_dis = self.cyq_dis.apply(lambda x: x[0]) / stock_data.close
            self.ret_dis = self.ret_dis.to_frame("x").join(self.cyq_dis.apply(lambda x: x[1]).rename("y")).dropna(
                how="all")

        def distribution_ret_avg(self, stock_data, index_data, params=[]):
            result = stock_data.close.copy()
            result.loc[:] = np.nan
            result.update(self.cyq_dis.apply(lambda x: np.sum(x[0] * x[1] / np.sum(x[1]))))
            return result.sort_index()

        def distribution_ret_std(self, stock_data, index_data, params=[]):
            result = stock_data.close.copy()
            result.loc[:] = np.nan
            result.update(self.cyq_dis.apply(lambda x: np.sum((x[0] - np.sum(x[0] * x[1])) ** 2 * x[1])) ** (1 / 2))
            return result.sort_index()

        def distribution_ret_skew(self, stock_data, index_data, params=[]):
            result = stock_data.close.copy()
            result.loc[:] = np.nan
            result.update(self.cyq_dis.apply(lambda x: np.sum((x[0] - np.sum(x[0] * x[1])) ** 3 * x[1])))
            return result.sort_index()

        def distribution_ret_kurt(self, stock_data, index_data, params=[]):
            result = stock_data.close.copy()
            result.loc[:] = np.nan
            cm4 = self.cyq_dis.apply(lambda x: np.sum((x[0] - np.sum(x[0] * x[1])) ** 4 * x[1]))
            cm2 = self.cyq_dis.apply(lambda x: np.sum((x[0] - np.sum(x[0] * x[1])) ** 2 * x[1]))
            result.update(cm4 / cm2 ** 2)
            return result.sort_index()

        def distribution_max_prob_ret(self, stock_data, index_data, params=[]):
            result = stock_data.close.copy()
            result.loc[:] = np.nan
            max_prob_p = self.cyq_dis.apply(lambda x: x[0][x[1].argmax()])
            result.update(stock_data.close / max_prob_p)
            return result.sort_index()

        def distribution_bal(self, stock_data, index_data, params=[]):
            result = stock_data.close.copy()
            result.loc[:] = np.nan
            result.update(
                self.ret_dis.apply(lambda x: np.sum(x["y"][(x["x"] > 0.98) & (x["x"] < 1.02)]),
                                   axis=1))
            return result.sort_index()

        def distribution_profit_l(self, stock_data, index_data, params=[]):
            result = stock_data.close.copy()
            result.loc[:] = np.nan
            result.update(self.ret_dis.apply(lambda x: np.sum(x["y"][x["x"] > 1.10]),
                                             axis=1))
            return result.sort_index()

        def distribution_profit_s(self, stock_data, index_data, params=[]):
            result = stock_data.close.copy()
            result.loc[:] = np.nan
            result.update(
                self.ret_dis.apply(lambda x: np.sum(x["y"][(x["x"] > 1.02) & (x["x"] < 1.10)]),
                                   axis=1))
            return result.sort_index()

        def distribution_loss_s(self, stock_data, index_data, params=[]):
            result = stock_data.close.copy()
            result.loc[:] = np.nan
            result.update(
                self.ret_dis.apply(lambda x: np.sum(x["y"][(x["x"] > 0.90) & (x["x"] < 0.98)]),
                                   axis=1))
            return result.sort_index()

        def distribution_loss_l(self, stock_data, index_data, params=[]):
            result = stock_data.close.copy()
            result.loc[:] = np.nan
            result.update(self.ret_dis.apply(lambda x: np.sum(x["y"][x["x"] < 0.90]),
                                             axis=1))
            return result.sort_index()

    cyq_daily = CyqFactorsDaily(cyq_dis.loc[start_date: end_date])
    cyq_factors_dict = dict()
    for f_name in cyq_factor_names:
        factor = getattr(cyq_daily, f_name)
        tmp_f = factor(stock_data[start_date: end_date], index_data[start_date: end_date], [])
        if len(tmp_f.unique()) < 5:
            print("check", factor.__name__, sep=" ")
        cyq_factors_dict[f_name] = tmp_f.loc[start_date:end_date]
    return cyq_factors_dict


def cum_factor_get(cum_factors_df, start_date, end_date, load_dir_path, shift_multi=1):
    """需要历史因子数据的因子数据计算

    Parameters
    ------------
    cum_factors_df: pd.DataFrame
        因子的信息df，包括因子的因子集factors_set、因子名factor_name、因子的缓冲天数shift_n、因子的方向direction
    load_dir_path: str
        历史因子数据的h5文件存放地址

    Returns
    -------------
    dict
        因子数据字典
    """
    class CumFactorsDaily:

        def __init__(self, start_dt):
            self.alpha191 = self.ALPHA191CUM(start_dt)
            self.CICC = self.CICCCUM(start_dt)

        class ALPHA191CUM:

            def __init__(self, start_dt):
                self.start_date = start_dt
                self.hdf5_path = os.path.join(load_dir_path, "alpha191.h5")

            def alpha_143(self, stock_data, index_data, params=[]):
                start_dt = self.start_date
                f = pd.read_hdf(self.hdf5_path, key="/"+"alpha_143", mode="r")
                if f.loc[bday_shift(start_dt, -1):bday_shift(start_dt, -1)].empty:
                    if not len(params):
                        return f.last_valid_index()[0]
                    start_dt = bday_shift(f.last_valid_index()[0], 1)

                f = f.loc[:bday_shift(start_dt, -1)].dropna().groupby(level=1).tail(1).droplevel(0)
                x = stock_data.close / delay(stock_data.close, 1)
                cond = stock_data.close > delay(stock_data.close, 1)
                x_fill = (x * cond).replace(0, np.nan).dropna().loc[start_dt:]
                tmp = x_fill.loc[start_dt].mul(f)
                tmp.loc[tmp.isna()] = x_fill.loc[start_dt]
                tmp.loc[tmp.isna()] = f
                tmp = tmp.reset_index()
                tmp["datetime"] = start_dt
                tmp.set_index(["datetime", "symbol"], inplace=True)
                x_fill = pd.concat([tmp.squeeze(), x_fill.loc[bday_shift(start_dt, 1):]]).sort_index()
                x_fill = x_fill.groupby(level=1).cumprod()
                x[:] = np.nan
                x.update(x_fill)
                return x.groupby(level=1).fillna(method="ffill")

        class CICCCUM:

            def __init__(self, start_dt):
                self.start_date = start_dt
                self.hdf5_path = os.path.join(load_dir_path, "CICC.h5")

            def mmt_report_overnight(self, stock_data, index_data, params=[]):
                start_dt = self.start_date
                f = pd.read_hdf(self.hdf5_path, key="/"+"mmt_report_overnight", mode="r")
                if f.loc[bday_shift(start_dt, -1):bday_shift(start_dt, -1)].empty:
                    if not len(params):
                        return f.last_valid_index()[0]
                    start_dt = bday_shift(f.last_valid_index()[0], 1)
                f = f.loc[:bday_shift(start_dt, -1)].dropna().groupby(level=1).tail(1).droplevel(0)

                equity_chg = stock_data.equity - stock_data.equity.groupby(level=1).shift(1)
                equity_chg.loc[equity_chg == 0] = np.nan
                overnight_ret = ts_sum(stock_data.open / delay(stock_data.close, 1) - 1, 21)
                overnight_ret.loc[equity_chg.groupby(level=1).shift(1).isna()] = np.nan

                tmp = f.reset_index()
                tmp["datetime"] = bday_shift(start_dt, -1)
                tmp.set_index(["datetime", "symbol"], inplace=True)
                overnight_ret = pd.concat([tmp.squeeze(), overnight_ret.loc[start_dt:]]).sort_index()

                return overnight_ret.groupby(level=1).fillna(method="ffill").loc[start_dt:]

            def mmt_report_jump(self, stock_data, index_data, params=[]):
                start_dt = self.start_date
                f = pd.read_hdf(self.hdf5_path, key="/"+"mmt_report_jump", mode="r")
                if f.loc[bday_shift(start_dt, -1):bday_shift(start_dt, -1)].empty:
                    if not len(params):
                        return f.last_valid_index()[0]
                    start_dt = bday_shift(f.last_valid_index()[0], 1)
                f = f.loc[:bday_shift(start_dt, -1)].dropna().groupby(level=1).tail(1).droplevel(0)

                equity_chg = stock_data.equity - stock_data.equity.groupby(level=1).shift(1)
                equity_chg.loc[equity_chg == 0] = np.nan
                ex_overnight_ret = stock_data.open / delay(stock_data.close, 1) - index2stock(
                    index_data.open / delay(index_data.close, 1), stock_data)
                ex_overnight_ret.loc[equity_chg.groupby(level=1).shift(1).isna()] = np.nan

                tmp = f.reset_index()
                tmp["datetime"] = bday_shift(start_dt, -1)
                tmp.set_index(["datetime", "symbol"], inplace=True)
                ex_overnight_ret = pd.concat([tmp.squeeze(), ex_overnight_ret.loc[start_dt:]]).sort_index()

                return ex_overnight_ret.groupby(level=1).fillna(method="ffill").loc[start_dt:]

            def mmt_report_period(self, stock_data, index_data, params=[]):
                start_dt = self.start_date
                f = pd.read_hdf(self.hdf5_path, key="/"+"mmt_report_period", mode="r")
                if f.loc[bday_shift(start_dt, -1):bday_shift(start_dt, -1)].empty:
                    if not len(params):
                        return f.last_valid_index()[0]
                    start_dt = bday_shift(f.last_valid_index()[0], 1)
                f = f.loc[:bday_shift(start_dt, -1)].dropna().groupby(level=1).tail(1).droplevel(0)

                equity_chg = stock_data.equity - stock_data.equity.groupby(level=1).shift(1)
                equity_chg.loc[equity_chg == 0] = np.nan
                ret = stock_data.close / delay(stock_data.close, 2)
                ret.loc[equity_chg.groupby(level=1).shift(1).isna()] = np.nan

                tmp = f.reset_index()
                tmp["datetime"] = bday_shift(start_dt, -1)
                tmp.set_index(["datetime", "symbol"], inplace=True)
                ret = pd.concat([tmp.squeeze(), ret.loc[start_dt:]]).sort_index()

                return ret.groupby(level=1).fillna(method="ffill").loc[start_dt:]

    cum_daily = CumFactorsDaily(start_date)
    cum_factors_dict = dict()
    cum_factors_df = cum_factors_df.copy()
    cum_factors_df["shift_n_mul"] = (cum_factors_df["shift_n"] * shift_multi).astype("int")
    for row in cum_factors_df.iterrows():
        factor = getattr(getattr(cum_daily, row[1]["factors_set"]), row[1]["factor_name"])
        tmp_f = factor(stock_data[bday_shift(start_date, -row[1]["shift_n_mul"]): end_date],
                       index_data[bday_shift(start_date, -row[1]["shift_n_mul"]): end_date],
                       [])
        if isinstance(tmp_f, pd.Timestamp):
            tmp_start_date_shift = bday_shift(tmp_f, -row[1]["shift_n_mul"] + 1)
            if stock_data.close.first_valid_index()[0] >= tmp_start_date_shift:
                raise ValueError(
                    factor.__name__ + "需要更长的shift值至" + tmp_start_date_shift.strftime("%Y-%m-%d"))
            tmp_f = factor(stock_data[tmp_start_date_shift: end_date],
                           index_data[tmp_start_date_shift: end_date],
                           [True])
        if len(tmp_f.unique()) < 5:
            print("check", factor.__name__, sep=" ")
        if not row[1]["factors_set"] in cum_factors_dict.keys():
            cum_factors_dict[row[1]["factors_set"]] = dict()
        cum_factors_dict[row[1]["factors_set"]][row[1]["factor_name"]] = tmp_f.loc[start_date:end_date]
    return cum_factors_dict


def get_data(start_date, end_date, shift_n):
    """获得股票和指数的量价数据

    Parameters
    --------------
    start_date: datetime.datetime
        量价数据开始日期
    end_date: datetime.datetime
        量价数据结束日期
    shift_n: int
        start_date - shift_n个工作日为实际的开始日期，作为数据缓冲

    Returns
    --------------
    (PriceVolumeData, PriceVolumeData)
        股票量价数据和指数量价数据
    """
    dm = DataManager()
    dm.set_parameters(symbols=None, index=['000905'], start=bday_shift(start_date, -shift_n), end=end_date)
    # 行业3为老中信，37为新中信，由于新中信支持更好的板块分类，这里用新中信来计算量化因子
    stock_data, index_data = dm.load_data(database='JYDB', industry=37, share=True, equity=True)
    # 获得股票行业板块分组，参考中信2021行业分类说明（仅仅用于一些alpha191的特殊因子）
    stock_data.industry["Sector"] = get_citic_sector(stock_data.industry["Fourth"])
    stock_data.industry["SectorName"] = get_citic_sector_name(stock_data.industry["Sector"])
    stock_data, index_data = data_nan_adjust(stock_data, index_data)
    return PriceVolumeDataLoc(stock_data), PriceVolumeDataLoc(index_data)


def get_factors_data(start_date, end_date=None, info_df=None, shift_multi=1, neut=False, cum_load_dir_path=None):
    """获得因子数据

    Parameters
    --------------
    start_date: datetime.datetime
        因子数据开始日期
    end_date: datetime.datetime
        因子数据结束日期，默认为当前日期
    info_df: pd.DataFrame
        因子的信息df，包括因子的因子集factors_set、因子名factor_name、因子的缓冲天数shift_n、因子的方向direction
    shift_multi: bool
        缓冲天数倍数，实际缓冲天数为shift_n*shift_multi，默认为1
    neut: bool
        是否还要输出因子中性化数据，若为False(默认)，则输出返回因子数据字典f_dict，若为True则输出因子数据字典f_dict和中性化因子数据字典f_neut_dict
    cum_load_dir_path: str
        过去因子数据的存储地址，如果要计算基于过去因子数据的因子数据则需要


    Returns
    -----------------
    dict
        因子数据字典
    """
    if info_df is None:
        info_df = shift_df.copy()
    if (end_date is None) or (end_date < start_date):
        end_date = start_date
    if "cum" in info_df["factor_type"].values:
        if cum_load_dir_path is None:
            raise TypeError("get_factors_data() missing 1 required argument: 'cum_load_dir_path'")
    global stock_data, index_data
    if "cyq" not in info_df["factor_type"].values:
        max_shift_n = max(int(info_df["shift_n"].max() * shift_multi), 2)
    else:
        max_shift_n = max(int(251 * shift_multi), int(info_df["shift_n"].max() * shift_multi), 2)
    stock_data, index_data = get_data(start_date, end_date, max_shift_n)

    if "alpha191" in info_df["factors_set"].values:
        fix_factor_preload = alpha191_factors.PreLoad(stock_data)

    f_dict = dict()
    for row in info_df.set_index("factors_set")["factor_name"].iteritems():
        if not row[0] in f_dict.keys():
            f_dict[row[0]] = dict()
        f_dict[row[0]].update({row[1]: None})

    tqdm_bar = tqdm(total=len(info_df), desc="calculating factors")
    for row in info_df[info_df["factor_type"] == "ordinary"].iterrows():
        if row[1]["factors_set"] == "alpha101":
            f = getattr(alpha101_factors, row[1]["factor_name"])
        elif row[1]["factors_set"] == "alpha191":
            f = getattr(alpha191_factors, row[1]["factor_name"])
        elif row[1]["factors_set"] == "CICC":
            f = getattr(CICC_factors, row[1]["factor_name"])
        else:
            raise ValueError("不正确的因子集名")
        f_value = ordinary_factor_get(f, start_date, end_date, row[1]["shift_n"], shift_multi)
        f_dict[row[1]["factors_set"]][row[1]["factor_name"]] = f_value
        tqdm_bar.update(1)

    if "cyq" in info_df["factor_type"].values:
        cyq_dis = CICC_factors.cyq_dis(stock_data[bday_shift(start_date, -int(251 * shift_multi)): end_date],
                                       index_data[bday_shift(start_date, -int(251 * shift_multi)): end_date],
                                       [252]).loc[start_date:end_date]
        if "cyq_dis" in f_dict["CICC"].keys():
            tqdm_bar.update(1)
        f_dict["CICC"]["cyq_dis"] = cyq_dis
        f_dict["CICC"].update(CICC_cyq_factor_get(info_df[info_df["factor_type"] == "cyq"]["factor_name"],
                                                  start_date, end_date, cyq_dis))
        tqdm_bar.update(len(info_df[info_df["factor_type"] == "cyq"]))

    if "cum" in info_df["factor_type"].values:
        cum_factor_dict = cum_factor_get(info_df[info_df["factor_type"] == "cum"], start_date, end_date,
                                         cum_load_dir_path, shift_multi)
        print(cum_factor_dict)
        for key in cum_factor_dict.keys():
            f_dict[key].update(cum_factor_dict[key])
        tqdm_bar.update(len(info_df[info_df["factor_type"] == "cum"]))

    tqdm_bar.close()

    if neut:
        import copy
        f_neut_dict = copy.deepcopy(f_dict)  # 对于字典的copy，只是字典有不同的指针，字典的值指针不做改变
        if "cyq" in info_df["factor_type"].values:
            f_neut_dict["CICC"]["cyq_dis"] = None
        tqdm_bar = tqdm(total=len(info_df[info_df["factor_type"] != "cyq_dis"]), desc="neutralizing factors")
        for key1, value1 in f_neut_dict.items():
            for key2, f in value1.items():
                if key2 == "cyq_dis":
                    continue
                print(key2, f)
                f_neut = neut_reg(stand_norm(rext_mad(f, 3)), np.log(stock_data.mkt_value), stock_data.industry.First)
                f_direct = info_df.loc[(info_df["factors_set"] == key1) &
                                       (info_df["factor_name"] == key2), "direction"].iat[0]
                if f_direct == "+":
                    f_neut_dict[key1][key2] = f_neut
                else:
                    f_neut_dict[key1][key2] = -f_neut
                tqdm_bar.update(1)
                del f_neut
        return f_dict, f_neut_dict
    return f_dict


def update_factors_data(factors_dict, save_dir_path, start_date,
                        end_date=None, save_end_str="", skip_cyq=True):
    """更新因子数据至指定h5文件夹

    更新因子格式为
    h5文件夹 -> 因子集h5文件 factors_set.h5 -> 因子名组 /factor_name -> 因子数据pd.Series (index: (日期datetime, 股票代码symbol), value: 因子值)

    Parameters
    ------------
    factors_dict: dict
        因子数据字典
    save_dir_path: str
        h5文件夹存储地址
    start_date: datetime.datetime
        存储起始日期，当存储文件夹已有因子数据，将会存储大于已有因子数据最后日期的因子数据
    end_date: datetime.datetime
        存储结束日期
    save_end_str: str
        给因子集名添加尾缀，即存储为factors_set+save_end_str.h5文件
    skip_cyq: bool
        跳过存储筹码分布文件，默认为True

    Returns
    -----------------
    pd.DataFrame
        因子存储记录
    """
    if end_date is None:
        end_date = start_date
    save_status_lst = list()
    date_lst = pd.bdate_range(start=start_date, end=end_date, freq="C", closed=None,
                              holidays=chinese_calendar.get_holidays(start_date, end_date, include_weekends=False))
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    for factors_set, value1 in factors_dict.items():
        save_path = os.path.join(save_dir_path, factors_set + save_end_str + ".h5")
        # 获得hdf5文件的groups信息
        if os.path.exists(save_path):
            with pd.HDFStore(save_path, "r") as f_store_hdf5:
                h5_info_df = list(f_store_hdf5.root.__members__)
            h5_info_df = pd.Series(h5_info_df, name=factors_set)
        else:
            h5_info_df = pd.Series([], name=factors_set)

        for factor_name, f in tqdm(value1.items(), desc="updating " + factors_set + save_end_str + " factors"):
            if f is not None:
                if f.first_valid_index()[0] > start_date:
                    print(factors_set, " ", factor_name, "起始日期应小于等于设定日期", start_date.strftime("%Y-%m-%d"))
                    save_status_lst.append([factors_set, factor_name, datetime.now(), f.first_valid_index()[0],
                                            "fail: first date later than " + start_date.strftime("%Y-%m-%d")])
                    continue
                if factor_name == "cyq_dis":
                    if not skip_cyq:
                        f = f.unstack()
                        if not os.path.exists("cyq_dis"):
                            os.makedirs("cyq_dis")
                        if not os.path.exists("cyq_dis/month"):
                            os.makedirs("cyq_dis/month")
                        month_lst = date_lst.strftime("%Y%m").unique()
                        for month in month_lst:
                            tmp_f = f.loc[f.index.strftime("%Y%m") == month][start_date:end_date]

                            if not os.path.exists(r"cyq_dis/month/cyq_dis_" + month + ".pkl"):
                                tmp_f.to_pickle(r"cyq_dis/month/cyq_dis_" + month + ".pkl")
                                save_status_lst.append([factors_set + save_end_str, factor_name, datetime.now(),
                                                        tmp_f.last_valid_index(), "success: generate"])
                            else:
                                tmp_load_f = pd.read_pickle(r"cyq_dis/month/cyq_dis_" + month + ".pkl")
                                tmp_start_date = bday_shift(tmp_load_f.last_valid_index(), 1)
                                if tmp_start_date >= start_date:
                                    tmp_f = pd.concat([tmp_load_f,
                                                       tmp_f.loc[tmp_start_date:]])
                                    tmp_f.to_pickle(r"cyq_dis/month/cyq_dis_" + month + ".pkl")
                                    save_status_lst.append([factors_set + save_end_str, factor_name, datetime.now(),
                                                            tmp_f.last_valid_index(), "success: update"])
                                else:
                                    print(factors_set, " ", factor_name, "中间有缺失日期，起始日期应小于",
                                          tmp_start_date.strftime("%Y-%m-%d"))
                                    save_status_lst.append([factors_set + save_end_str, factor_name, datetime.now(),
                                                            np.nan,
                                                            "fail: first date later than " +
                                                            tmp_start_date.strftime("%Y-%m-%d")])
                                del tmp_load_f
                            del tmp_f
                    continue

                try:
                    tmp_f = f[start_date:end_date].astype("float").rename().sort_index()
                    with pd.HDFStore(save_path) as f_store_hdf5:
                        if factor_name in h5_info_df.values:
                            old_last_date = f_store_hdf5.select("/" + factor_name, start=-1).index[-1][0]
                            upd_start_date = bday_shift(old_last_date, 1)
                            last_date = tmp_f.index[-1][0]
                            if upd_start_date >= tmp_f.first_valid_index()[0]:
                                f_store_hdf5.put("/" + factor_name, tmp_f.loc[upd_start_date:],
                                                 format="table", append=True)
                                save_status_lst.append([factors_set + save_end_str, factor_name, datetime.now(),
                                                        tmp_f.index[-1][0],
                                                        "success: from " + old_last_date.strftime("%Y-%m-%d") +
                                                        " update to " + last_date.strftime("%Y-%m-%d")])
                            else:
                                print(factors_set, " ", factor_name, "中间有缺失日期，起始日期应小于",
                                      upd_start_date.strftime("%Y-%m-%d"))
                                save_status_lst.append([factors_set + save_end_str, factor_name, datetime.now(),
                                                        np.nan,
                                                        "fail: first date later than " +
                                                        upd_start_date.strftime("%Y-%m-%d")])
                        else:
                            f_store_hdf5.put("/" + factor_name, tmp_f,
                                             format="table", append=False)
                            save_status_lst.append([factors_set + save_end_str, factor_name, datetime.now(),
                                                    last_date,
                                                    "success: generate to " + last_date.strftime("%Y-%m-%d")])
                    del tmp_f
                except Exception as e:
                    print(factors_set, " ", factor_name, "Exception occurs", e)
                    save_status_lst.append([factors_set + save_end_str, factor_name, datetime.now(), np.nan,
                                            "fail: Exception occurs when updating \n" +
                                            ''.join(traceback.format_exception(e.__class__, e, e.__traceback__))])
    save_status_df = pd.DataFrame(save_status_lst,
                                  columns=["factors_set", "factor_name", "upd_date", "last_date", "status"])
    return save_status_df


if __name__ == '__main__':
    shift_df = pd.read_csv("factors_info.csv", index_col=0)

    start_date = datetime(2022, 12, 1)
    end_date = datetime.now()
    save_dir_path = r"temp"
    cum_load_dir_path = r"\\192.168.2.27\research\Factors\q_factors"
    target_df = shift_df.copy()
    # target_df = shift_df[shift_df["factors_set"] == "alpha101"]
    # target_df = shift_df[(shift_df["factors_set"] == "CICC")]
    # target_df = shift_df[(shift_df["factor_type"] == "cum")]
    # 加长shift避免有误差计算
    f_dict, f_neut_dict = get_factors_data(start_date, end_date=end_date,
                                           info_df=target_df, shift_multi=1.2,
                                           neut=True, cum_load_dir_path=cum_load_dir_path)

    f_upd_info = update_factors_data(f_dict, save_dir_path, start_date,
                                     end_date=end_date, save_end_str="")
    f_neut_upd_info = update_factors_data(f_neut_dict, save_dir_path, start_date,
                                          end_date=end_date, save_end_str="_neut")

    upd_info_path = r"upd_info.parquet"
    upd_info = pd.DataFrame()
    if os.path.exists(upd_info_path):
        upd_info = pd.read_parquet(upd_info_path)
    upd_info = pd.concat([upd_info, f_upd_info, f_neut_upd_info], ignore_index=True)
    upd_info.to_parquet(upd_info_path)
    