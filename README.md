# Инструкция по запуску

## Запуск локально

для совместимости с окружением, в котором обучалась модель, рекомендуется python 3.13. я сохранил модель в google colab с python 3.13.18, sklearn 1.6.1, imblearn 0.14.2 и joblib 1.6.0. закреплённые версии остальных библиотек я взял из локального рабочего окружения проекта

**macOS / Linux** (из wine_app):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**Windows (PowerShell)** (из wine_app):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

приложение откроется по адресу, указанному streamlit в терминале (обычно http://localhost:8501)

## Ответы на вопросы из задания 4.5
st.cache_resource используется для модели, тк она является общим объектом, который можно повторно использовать без создания копий, а st.cache_data используется для таблиц и результатов вычислений, тк он сохраняет данные в кэше и возвращает отдельные копии. без декоратора joblib.load выполнялся бы при каждом перезапуске скрипта streamlit, например при изменении порога классификации

## Деплой

ссылка: https://appwineq.streamlit.app/