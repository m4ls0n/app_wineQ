import matplotlib.pyplot as plt
from sklearn import metrics
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

appDir = Path(__file__).resolve().parent
@st.cache_resource
def loadModel():
    return joblib.load(appDir / "model.joblib")

@st.cache_data
def loadTest():
    return pd.read_csv(appDir / "test.csv")

@st.cache_data
def testProba(_model, X):
    return _model.predict_proba(X)[:, 1]

model = loadModel()
test = loadTest()
with open(appDir / "config.json", encoding="utf-8") as f:
    config = json.load(f)

feats = test.drop(columns="target").columns.tolist()
st.sidebar.header("Настройки модели")
thres = st.sidebar.slider("Порог классификации", min_value=0.0, max_value=1.0, value=float(config["threshold"]), step=0.0001, format="%.4f")
st.title("Сервис рекомендаций вин")
tab1, tab2, tab3, tab4 = st.tabs(["Оценка одного вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"])
with tab1:
    st.subheader("Характеристики вина")
    vals = {}
    cols = st.columns(2)
    numFeats = [f for f in feats if f != "is_red"]
    for i, feature in enumerate(numFeats):
        mi = float(test[feature].min())
        ma = float(test[feature].max())
        med = float(test[feature].median())
        digits = 5 if feature == "density" else (3 if feature == "chlorides" else 2)
        with cols[i % 2]:
            vals[feature] = st.number_input(feature, min_value=mi, max_value=ma, value=med, step=10 ** (-digits), format=f"%.{digits}f")

    isR = st.toggle("Красное вино (выключено — белое)", value=bool(test["is_red"].median()))
    vals["is_red"] = int(isR)
    wine = pd.DataFrame([vals], columns=feats)
    prob = model.predict_proba(wine)[0, 1]
    pred = int(prob >= thres)
    st.divider()
    st.metric("Вероятность хорошего вина", f"{prob:.2%}")
    if pred == 1:
        st.success("Хорошее вино — рекомендуется!")
    else:
        st.warning("Обычное вино — не рекомендуется.")

    st.caption(f"Текущий порог классификации: {thres:.4f}")

with tab2:
    st.subheader("Метрики на тестовой выборке")
    yTrue = test["target"].to_numpy()
    probs = testProba(model, test[feats])
    yPred = (probs >= thres).astype(int)
    cm = metrics.confusion_matrix(yTrue, yPred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = metrics.precision_score(yTrue, yPred, zero_division=0)
    rec = metrics.recall_score(yTrue, yPred)
    f1 = metrics.f1_score(yTrue, yPred)
    col1, col2, col3 = st.columns(3)
    col1.metric("precision", f"{prec:.4f}")
    col2.metric("recall", f"{rec:.4f}")
    col3.metric("f1", f"{f1:.4f}")
    st.caption(f"Рекомендовано {yPred.sum()} из {len(yPred)} вин при пороге {thres:.4f}")
    st.subheader("Матрица ошибок")
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Обычное (0)", "Хорошее (1)"]).plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("ROC- и PR-кривые")
    fpr, tpr, _ = metrics.roc_curve(yTrue, probs)
    precisions, recalls, _ = metrics.precision_recall_curve(yTrue, probs)
    pointFpr = fp / (fp + tn)
    pointPrec = prec if tp + fp > 0 else 1.0
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"ROC-AUC = {metrics.roc_auc_score(yTrue, probs):.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Случайная модель")
        ax.scatter(pointFpr, rec, color="red", s=80, zorder=5, label=f"Порог = {thres:.4f}")
        ax.set(xlabel="FPR", ylabel="TPR", title="ROC-кривая", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recalls, precisions, label=f"PR-AUC = {metrics.average_precision_score(yTrue, probs):.4f}")
        ax.axhline(yTrue.mean(), linestyle="--", color="gray", label="Случайный уровень")
        ax.scatter(rec, pointPrec, color="red", s=80, zorder=5, label=f"Порог = {thres:.4f}")
        ax.set(xlabel="Recall", ylabel="Precision", title="PR-кривая", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    st.info("При снижении порога модель рекомендует больше вин и находит больше хороших, но также может чаще рекомендовать посредственные. При увеличении порога рекомендаций становится меньше: обычно precision растёт, а recall снижается.")

with tab3:
    st.subheader("Пакетное предсказание")
    st.write("Загрузите CSV файл с характеристиками вин без столбца target. Для каждого вина приложение рассчитает вероятность класса 1 и решение при выбранном пороге.")
    file = st.file_uploader("Загрузить CSV", type="csv")
    if file is not None:
        try:
            data = pd.read_csv(file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            st.error("Не удалось прочитать CSV файл. Проверьте его формат.")
        else:
            missing = [f for f in feats if f not in data.columns]
            if missing:
                st.error(f"В файле отсутствуют необходимые столбцы: {', '.join(missing)}")
            elif data.empty:
                st.error("Загруженный CSV файл не содержит объектов.")
            else:
                X = data[feats].apply(pd.to_numeric, errors="coerce").replace([float("inf"), -float("inf")], float("nan"))
                if X.isna().any().any():
                    st.error("В файле есть пропущенные или некорректные числовые значения.")
                else:
                    probsBatch = model.predict_proba(X)[:, 1]
                    predBatch = (probsBatch >= thres).astype(int)
                    result = data[feats].copy()
                    result["probability"] = probsBatch
                    result["prediction"] = predBatch
                    st.success(f"Обработано {len(result)} вин. Рекомендовано {predBatch.sum()}.")
                    st.dataframe(result)
                    csv = result.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("Скачать результат CSV", data=csv, file_name="wine_predictions.csv", mime="text/csv")


with tab4:
    st.subheader("Коэффициенты логистической регрессии")
    lr = model.named_steps["logisticregression"]
    coefs = pd.Series(lr.coef_[0], index=feats).sort_values()
    colours = ["tab:red" if x < 0 else "tab:green" for x in coefs]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(coefs.index, coefs.values, color=colours)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Коэффициент после масштабирования")
    ax.set_ylabel("Признак")
    ax.set_title("Коэффициенты логистической регрессии")
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    if coefs.max() > 0:
        st.write(f"Сильнее всего за хорошее вино голосует признак **{coefs.idxmax()}** (коэффициент {coefs.max():.4f}).")
    if coefs.min() < 0:
        st.write(f"Сильнее всего против хорошего вина голосует признак **{coefs.idxmin()}** (коэффициент {coefs.min():.4f}).")

    st.caption("Зелёные столбцы это положительные коэффициенты, красные это отрицательные. Чем больше модуль коэффициента, тем сильнее влияние признака на логарифм отношения шансов при неизменных остальных признаках.")