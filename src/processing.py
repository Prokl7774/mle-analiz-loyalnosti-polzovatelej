"""
Предобработка и анализ данных о заказах билетов.

Класс DataProcessor закрывает требования задания:
- загрузка данных и курса валют (курс за номинал тенге, по датам);
- конвертация тенге -> рубли по курсу на дату заказа;
- очистка по логике принятого исследования (неявные дубликаты по точному
  времени, некорректная выручка, хвост по p99, аномальные аккаунты);
- профили пользователей (первый заказ по точному времени с разрешением
  совпадений по order_id) и флаг возврата;
- retention по сегментам и когортам;
- проверка гипотез (z-тест долей, хи-квадрат, Спирмен);
- корреляционный анализ phi_k (глобальный и по сегментам).
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# ==================== Шим совместимости ====================
# phik 0.10.0 использует scipy.stats.mvn.mvnun, удалённый из новых scipy.
# Восстанавливаем функцию через scipy.stats.multivariate_normal.
import scipy.stats

try:
    import scipy.stats.mvn as _mvn_mod
except ImportError:  # модуля mvn нет совсем — создадим заглушку
    from types import SimpleNamespace
    _mvn_mod = SimpleNamespace()
    scipy.stats.mvn = _mvn_mod

if not hasattr(_mvn_mod, 'mvnun'):
    def _mvnun(lower, upper, mean, cov, maxpts=None,
               abseps=1e-5, releps=1e-5):
        """P(lower <= X <= upper) для многомерного нормального.

        Раньше считалась Fortran-функцией scipy.stats.mvn.mvnun,
        здесь заменена на CDF из scipy.stats.multivariate_normal.
        Возвращает кортеж (value, inform) — как исходная mvnun.
        """
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)
        BIG = 9.0  # за пределами 9 сигм нормальная плотность ~ 0
        lower = np.where(np.isfinite(lower), lower, -BIG)
        upper = np.where(np.isfinite(upper), upper, BIG)
        dist = scipy.stats.multivariate_normal(
            mean=mean, cov=cov, allow_singular=True
        )
        value = float(dist.cdf(upper) - dist.cdf(lower))
        return value, 0
    _mvn_mod.mvnun = _mvnun

# ==================== Форс последовательного phi_k ====================
# phik 0.10 параллелит расчёт через joblib: воркеры — отдельные процессы,
# шим совместимости в них не попадает. Форсируем n_jobs=1.
import phik  # noqa: F401  # активирует pandas-аксессор .phik_matrix()
import phik.phik as _phik_impl

_phik_impl.NCORES = 1

_orig_parallel = _phik_impl.Parallel

def _seq_parallel(*args, **kwargs):
    kwargs['n_jobs'] = 1  # всегда один процесс — шим виден везде
    return _orig_parallel(*args, **kwargs)

_phik_impl.Parallel = _seq_parallel


class DataProcessor:
    # --- Конфигурация: имена колонок выгрузки.
    # Сверены с реальными именами в data/orders.csv ---
    COL_USER = 'user_id'
    COL_ORDER = 'order_id'
    COL_DT = 'order_dt'      # дата заказа
    COL_TS = 'order_ts'      # точное время заказа (created_ts_msk)
    COL_EVENT_ID = 'event_id'
    COL_EVENT_TYPE = 'event_type_main'
    COL_DEVICE = 'device_type_canonical'
    COL_REGION = 'region_name'
    COL_SERVICE = 'service_name'
    COL_TICKETS = 'tickets_count'
    COL_REVENUE = 'revenue'           # выручка в исходной валюте
    COL_CURRENCY = 'currency_code'

    # --- Файл курсов: колонки data / nominal / curs / cdx ---
    CUR_FILENAME = 'final_tickets_tenge_df.csv'
    CUR_DATE_COL = 'data'
    CUR_VALUE_COL = 'curs'
    CUR_NOMINAL_COL = 'nominal'
    CUR_CODE_COL = 'cdx'
    CUR_CODE_VALUE = 'kzt'

    DAYS_RU = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    ORDERS_BINS = [0, 1, 2, 4, np.inf]
    ORDERS_LABELS = ['1 заказ', '2 заказа', '3-4 заказа', '5+ заказов']
    INTERVAL_COLS = ['total_orders', 'first_tickets', 'first_revenue']

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.clean_stats: dict = {}
        self.profiles: pd.DataFrame | None = None

    # ==================== 1. Загрузка ====================

    def load_data(self) -> pd.DataFrame:
        """Читает выгрузку из SQL, оставляет срез: mobile/desktop, без фильмов."""
        path = self.data_dir / 'orders.csv'
        if not path.exists():
            raise FileNotFoundError(f'Не найден файл данных: {path}')
        df = pd.read_csv(path, parse_dates=[self.COL_DT, self.COL_TS])
        df = df[df[self.COL_DEVICE].isin(['mobile', 'desktop'])]
        df = df[df[self.COL_EVENT_TYPE] != 'фильм']
        return df.reset_index(drop=True)

    def load_currency(self) -> pd.DataFrame:
        """Таблица курсов: дата, номинал, курс (рублей за номинал тенге)."""
        path = self.data_dir / self.CUR_FILENAME
        cur = pd.read_csv(path, parse_dates=[self.CUR_DATE_COL])
        cur = cur[cur[self.CUR_CODE_COL].astype(str).str.lower()
                  == self.CUR_CODE_VALUE]
        return cur[[self.CUR_DATE_COL, self.CUR_NOMINAL_COL, self.CUR_VALUE_COL]]

    # ==================== 2. Конвертация валюты ====================

    def convert_currency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Тенге -> рубли по курсу на дату заказа.

        Левый merge по нормализованной дате заказа — как в принятом
        исследовании. Левый merge сохраняет порядок строк исходного
        датафрейма, поэтому каждый заказ получает курс именно своей
        даты; никакой сортировки и позиционной записи нет.
        Если курс на дату заказа в файле отсутствует, rate остаётся
        NaN — строка отфильтруется при очистке выручки.
        """
        rates = self.load_currency()

        df = df.copy()
        df['_order_date'] = df[self.COL_DT].dt.normalize()
        rates = rates.rename(columns={self.CUR_DATE_COL: '_order_date'})
        df = df.merge(
            rates[['_order_date', self.CUR_NOMINAL_COL, self.CUR_VALUE_COL]],
            on='_order_date', how='left',
        ).drop(columns=['_order_date'])
        df = df.rename(columns={
            self.CUR_VALUE_COL: 'rate_curs',
            self.CUR_NOMINAL_COL: 'rate_nominal',
        })

        df['revenue_rub'] = df[self.COL_REVENUE].astype(float)
        is_kzt = (df[self.COL_CURRENCY].astype(str).str.lower()
                  == self.CUR_CODE_VALUE)
        df.loc[is_kzt, 'revenue_rub'] = (
            df.loc[is_kzt, self.COL_REVENUE]
            * df.loc[is_kzt, 'rate_curs']
            / df.loc[is_kzt, 'rate_nominal']
        ).round(2)

        self.clean_stats['n_currency_converted'] = int(is_kzt.sum())
        return df

    # ==================== 3. Очистка ====================

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Неявные дубликаты -> некорректная выручка -> хвост p99 -> аномальные аккаунты."""
        n_raw = len(df)

        # 3.1 Неявные дубликаты: одинаковый заказ под разными order_id.
        # Ключ — как в принятом исследовании: точное время заказа
        # (order_ts, не дата), пользователь, устройство, сервис,
        # событие по event_id и выручка в рублях (после конвертации).
        dedup_key = [self.COL_USER, self.COL_TS, self.COL_DEVICE,
                     self.COL_SERVICE, self.COL_EVENT_ID, 'revenue_rub']
        dedup_key = [c for c in dedup_key if c in df.columns]
        before = len(df)
        df = df.drop_duplicates(subset=dedup_key, keep='first')
        n_dupes = before - len(df)

        # 3.2 Некорректная выручка: неположительная (включая строки
        # без курса — revenue_rub = NaN не проходит фильтр > 0)
        before = len(df)
        df = df[df['revenue_rub'] > 0]
        n_bad = before - len(df)

        # 3.3 Хвост выручки: выше 99-го перцентиля
        p99_rev = df['revenue_rub'].quantile(0.99)
        before = len(df)
        df = df[df['revenue_rub'] <= p99_rev]
        n_tail = before - len(df)

        # 3.4 Аномальные аккаунты: заказов больше, чем p99 по пользователям
        orders_per_user = df.groupby(self.COL_USER)[self.COL_ORDER].count()
        orders_thr = orders_per_user.quantile(0.99)
        anomaly_users = orders_per_user[orders_per_user > orders_thr].index
        users_before = df[self.COL_USER].nunique()
        df = df[~df[self.COL_USER].isin(anomaly_users)]
        n_anomaly = users_before - df[self.COL_USER].nunique()

        self.clean_stats.update({
            'n_raw_rows': n_raw,
            'n_duplicates': int(n_dupes),
            'n_bad_revenue': int(n_bad),
            'revenue_p99': round(float(p99_rev), 2),
            'n_revenue_tail': int(n_tail),
            'orders_p99': int(orders_thr),
            'n_anomaly_users': int(n_anomaly),
            'n_final_rows': len(df),
            'n_final_users': int(df[self.COL_USER].nunique()),
        })
        return df.reset_index(drop=True)

    # ==================== 4. Профили пользователей ====================

    def build_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Пользовательский уровень: первый заказ + агрегаты за период."""
        total_orders = df.groupby(self.COL_USER)[self.COL_ORDER].count().rename('total_orders')
        total_rev = df.groupby(self.COL_USER)['revenue_rub'].sum().rename('total_revenue_rub')
        total_tix = df.groupby(self.COL_USER)[self.COL_TICKETS].sum().rename('total_tickets')

        # Первый заказ: минимальное точное время (order_ts);
        # при совпадении метки времени у заказов одного пользователя
        # порядок фиксируем по order_id — детерминизм, как в исследовании
        first_idx = (
            df.sort_values([self.COL_TS, self.COL_ORDER], kind='mergesort')
              .groupby(self.COL_USER, sort=False)
              .head(1)
              .index
        )
        first_feat = df.loc[first_idx, [
            self.COL_USER, self.COL_DEVICE, self.COL_EVENT_TYPE,
            self.COL_REGION, self.COL_SERVICE, self.COL_TICKETS,
            'revenue_rub', self.COL_DT,
        ]].set_index(self.COL_USER)
        first_feat = first_feat.rename(columns={
            self.COL_DEVICE: 'device_type',
            self.COL_EVENT_TYPE: 'event_type',
            self.COL_REGION: 'region',
            self.COL_SERVICE: 'service',
            self.COL_TICKETS: 'first_tickets',
            'revenue_rub': 'first_revenue',
            self.COL_DT: 'first_order_dt',
        })

        profiles = first_feat.join([total_orders, total_rev, total_tix])
        profiles['avg_order_revenue_rub'] = (
            profiles['total_revenue_rub'] / profiles['total_orders']
        )
        profiles['first_weekday'] = profiles['first_order_dt'].dt.dayofweek.map(
            dict(enumerate(self.DAYS_RU))
        )
        # Флаг возврата — по числу заказов (2 и более), не по времени:
        # два заказа с одинаковым временем всё равно считаются возвратом.
        profiles['is_returned'] = profiles['total_orders'] >= 2
        profiles['orders_segment'] = pd.cut(
            profiles['total_orders'], bins=self.ORDERS_BINS, labels=self.ORDERS_LABELS
        )
        self.profiles = profiles
        return profiles

    # ==================== 5. Retention по сегментам ====================

    def retention_overall(self) -> dict:
        p = self.profiles
        n, returned = len(p), int(p['is_returned'].sum())
        return {'users': n, 'returned': returned,
                'retention_pct': round(returned / n * 100, 1)}

    def retention_by(self, col: str, top_n: int | None = None) -> pd.DataFrame:
        """Доля вернувшихся по сегменту. Для регионов — топ-N + прочие взвешенно."""
        p = self.profiles
        g = p.groupby(col, observed=True).agg(
            users=('is_returned', 'size'),
            returned=('is_returned', 'sum'),
        )
        g['retention_pct'] = (g['returned'] / g['users'] * 100).round(1)
        if top_n is not None:
            top = g.sort_values('users', ascending=False).head(top_n)
            rest = g.drop(top.index)
            if len(rest):
                rest_row = {'users': int(rest['users'].sum()),
                            'returned': int(rest['returned'].sum())}
                rest_row['retention_pct'] = round(
                    rest_row['returned'] / rest_row['users'] * 100, 1
                )
                top.loc['Прочие (взвешенно)'] = rest_row
            g = top
        return g.sort_values('users', ascending=False)

    # ==================== 6. Когорты ====================

    def build_cohorts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Месячный retention первой покупки: когорта x месяц от первой покупки."""
        p = self.profiles
        first_month = p['first_order_dt'].dt.to_period('M').rename('cohort_month')

        tmp = df[[self.COL_USER, self.COL_DT]].copy()
        tmp['order_month'] = tmp[self.COL_DT].dt.to_period('M')
        tmp['cohort_month'] = tmp[self.COL_USER].map(first_month)
        tmp['month_offset'] = (tmp['order_month'] - tmp['cohort_month']).apply(lambda x: x.n)

        active = tmp.groupby(['cohort_month', 'month_offset'])[self.COL_USER].nunique()
        sizes = tmp.groupby('cohort_month')[self.COL_USER].nunique()
        return (active / sizes * 100).unstack().round(1)

    # ==================== 7. Гипотезы ====================

    @staticmethod
    def _z_test_two_proportions(succ1: int, n1: int, succ2: int, n2: int):
        """Z-тест разности долей с объединённой дисперсией."""
        p1, p2 = succ1 / n1, succ2 / n2
        p_pool = (succ1 + succ2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se
        p_value = 2 * stats.norm.sf(abs(z))
        return float(z), float(p_value)

    @staticmethod
    def _find_segment(table: pd.DataFrame, *keys: str) -> pd.Series:
        """Ищет сегмент по подстроке в индексе (защита от форм имени)."""
        for idx in table.index:
            if any(key in str(idx).lower() for key in keys):
                return table.loc[idx]
        raise KeyError(f'Сегмент {keys} не найден в таблице')

    def test_hypotheses(self, ret_event: pd.DataFrame) -> dict:
        p = self.profiles
        results = {}
        # Г1 (направленная): «после спорта возвращаются чаще».
        # Тест двусторонний, поэтому фиксируем отдельно: значимо ли
        # различие и в какую сторону оно направлено.
        sport = self._find_segment(ret_event, 'спорт')
        concerts = self._find_segment(ret_event, 'концерт')
        z, pv = self._z_test_two_proportions(
            int(sport['returned']), int(sport['users']),
            int(concerts['returned']), int(concerts['users']),
        )
        results['h1_sport_vs_concerts'] = {
            'diff_significant': bool(pv < 0.05),
            'higher_return_after_sport': bool(
                sport['retention_pct'] > concerts['retention_pct']
            ),
            'z': round(z, 2), 'p_value': round(pv, 4),
            'note': (f'возврат после спорта {sport["retention_pct"]}% против '
                     f'концертов {concerts["retention_pct"]}%'),
        }

        # Г2: крупные регионы возвращают чаще? Топ-2 против остальных
        reg = p.groupby('region', observed=True).agg(
            users=('is_returned', 'size'), returned=('is_returned', 'sum'))
        reg['retention_pct'] = (reg['returned'] / reg['users'] * 100).round(1)
        top2 = reg.sort_values('users', ascending=False).head(2)
        rest = reg.drop(top2.index)
        z, pv = self._z_test_two_proportions(
            int(top2['returned'].sum()), int(top2['users'].sum()),
            int(rest['returned'].sum()), int(rest['users'].sum()),
        )
        results['h2_top_regions'] = {
            'confirmed': pv < 0.05, 'z': round(z, 2), 'p_value': round(float(pv), 4),
            'top2_pct': float(round(top2['returned'].sum() / top2['users'].sum() * 100, 1)),
            'rest_pct': float(round(rest['returned'].sum() / rest['users'].sum() * 100, 1)),
        }

        # Г2-бонус: Спирмен — размер региона vs retention
        rho, pv_s = stats.spearmanr(reg['users'], reg['retention_pct'])
        results['spearman_region'] = {'rho': round(float(rho), 2),
                                      'p_value': round(float(pv_s), 3)}

        # Г3: хи-квадрат день недели первой покупки x возврат
        ct = pd.crosstab(p['first_weekday'], p['is_returned'])
        chi2, pv_chi, dof, _ = stats.chi2_contingency(ct)
        results['chi2_weekday'] = {'chi2': float(round(chi2, 1)), 'df': dof,
                                   'p_value': round(float(pv_chi), 3)}
        return results

    # ==================== 8. Корреляционный анализ (phi_k) ====================

    def analyze_correlations(self, segmented: bool = True) -> dict:
        """phi_k: глобальная матрица + опционально внутри сегментов числа заказов.

        Последовательный счёт форсируется патчем Parallel в шапке файла:
        воркеры joblib — отдельные процессы, шим туда не попадает.
        """
        p = self.profiles
        corr_cols = ['first_tickets', 'first_revenue', 'device_type',
                     'event_type', 'region', 'service', 'first_weekday',
                     'total_orders']
        result = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            phik_global = p[corr_cols].phik_matrix(
                interval_cols=self.INTERVAL_COLS,
            )
            result['global'] = phik_global
            result['with_total_orders'] = (
                phik_global['total_orders'].drop('total_orders')
                .sort_values(ascending=False).round(3)
            )
            if segmented:
                seg_corr = {}
                for seg in self.ORDERS_LABELS:
                    sub = p[p['orders_segment'] == seg]
                    interval = list(self.INTERVAL_COLS)
                    m = sub[corr_cols].phik_matrix(
                        interval_cols=interval,
                    )
                    if 'total_orders' in m.columns:
                        seg_corr[seg] = (
                            m['total_orders'].drop('total_orders')
                            .sort_values(ascending=False).round(3).to_dict()
                        )
                    else:
                        seg_corr[seg] = ('total_orders исключён из phi_k '
                                         '(нет разброса) — связи не определены')
                result['segmented'] = seg_corr
        return result

    # ==================== 9. Визуализация (опционально) ====================

    def plot_cohorts(self, cohorts: pd.DataFrame):
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig = plt.figure(figsize=(9, 6))
        sns.heatmap(cohorts, annot=True, fmt='.1f', cmap='Blues',
                    vmin=0, vmax=100, cbar_kws={'label': 'retention, %'})
        plt.title('Месячный retention первой покупки по когортам')
        plt.tight_layout()
        plt.show()

    # ==================== 10. Итоговый отчёт ====================

    def print_report(self, ret_event, ret_device, ret_region, cohorts, hyp, corr):
        s = self.clean_stats
        p = self.profiles
        line = '=' * 72

        print(line)
        print('ИТОГОВЫЙ ОТЧЁТ: предобработка и EDA данных о заказах билетов')
        print(line)

        print('\n1. ДАННЫЕ И ПОДГОТОВКА')
        print(f"   Загружено строк: {s['n_raw_rows']:,}")
        print(f"   Конвертировано заказов из тенге: {s['n_currency_converted']:,}")
        print(f"   Удалено неявных дубликатов: {s['n_duplicates']}")
        print(f"   Удалено строк с некорректной выручкой: {s['n_bad_revenue']}")
        print(f"   Хвост выручки (p99 = {s['revenue_p99']} руб): удалено {s['n_revenue_tail']} строк")
        print(f"   Аномальные аккаунты (порог {s['orders_p99']} заказов): исключено {s['n_anomaly_users']} пользователей")
        print(f"   Итог: {s['n_final_rows']:,} строк, {s['n_final_users']:,} пользователей")

        overall = self.retention_overall()
        print('\n2. ОСНОВНЫЕ РЕЗУЛЬТАТЫ')
        print(f"   Возврат: {overall['returned']:,} из {overall['users']:,} ({overall['retention_pct']}%)")
        print('   Retention по типу события:')
        print(ret_event.to_string())
        print('   Retention по устройству:')
        print(ret_device.to_string())
        print('   Retention по топ-10 регионам:')
        print(ret_region.to_string())

        print('\n   Средняя выручка заказа: 1 заказ vs 2+ заказов')
        g1 = p.loc[p['total_orders'] == 1, 'avg_order_revenue_rub']
        g2 = p.loc[p['total_orders'] > 1, 'avg_order_revenue_rub']
        print(f"   1 заказ: n={len(g1)}, медиана {g1.median():.1f}, среднее {g1.mean():.1f}")
        print(f"   2+ заказов: n={len(g2)}, медиана {g2.median():.1f}, среднее {g2.mean():.1f}")

        print('\n   Когорты (месячный retention первой покупки):')
        print(cohorts.to_string())

        print('\n3. ГИПОТЕЗЫ')
        for name, h in hyp.items():
            print(f'   {name}: {h}')

        print('\n4. КОРРЕЛЯЦИИ (phi_k с total_orders)')
        print(corr['with_total_orders'].to_string())
        if 'segmented' in corr:
            print('   Внутри сегментов:')
            for seg, val in corr['segmented'].items():
                print(f'   {seg}: {val}')

        print('\n5. РЕКОМЕНДАЦИИ')
        print('   - Главный рычаг вовлечённости — тип события, а не размер чека:')
        print('     для спортивных событий нужны отдельные механики возврата.')
        print('   - Сегментация по типу события и региону, не по сумме первого заказа.')
        print('   - Первые коммуникации (welcome-цепочки) ставить на субботу и понедельник.')
        print('   - Аномальные профили вывести в отдельный контур после проверки их природы.')

        print('\n6. ОГРАНИЧЕНИЯ')
        print('   - Окно наблюдения ~5 месяцев: долгосрочные эффекты не видны.')
        print('   - Сезонность неотделима от типа события (эффект ёлок летом).')
        print('   - География — суррогатный признак; phi_k измеряет силу связи,')
        print('     а не причинность; выводы не проверены на новых данных.')

    # ==================== Пайплайн целиком ====================

    def run_pipeline(self) -> pd.DataFrame:
        df = self.load_data()
        df = self.convert_currency(df)
        df = self.clean_data(df)
        return df
