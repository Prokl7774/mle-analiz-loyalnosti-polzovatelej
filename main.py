"""Точка входа: пайплайн предобработки и EDA + итоговый отчёт в терминал."""
from src.processing import DataProcessor


def main():
    dp = DataProcessor(data_dir='data')

    # Пайплайн: загрузка -> конвертация -> очистка -> профили
    df = dp.run_pipeline()
    profiles = dp.build_profiles(df)

       # Retention по сегментам (имена — из profiles, они переименованы)
    ret_event = dp.retention_by('event_type')
    ret_device = dp.retention_by('device_type')
    ret_region = dp.retention_by('region', top_n=10)


    # Когорты
    cohorts = dp.build_cohorts(df)

    # Гипотезы и корреляции
    hyp = dp.test_hypotheses(ret_event)
    corr = dp.analyze_correlations(segmented=True)

    # Опционально: тепловая карта когорт
    # dp.plot_cohorts(cohorts)

    # Итоговый отчёт
    dp.print_report(ret_event, ret_device, ret_region, cohorts, hyp, corr)


if __name__ == '__main__':
    main()

