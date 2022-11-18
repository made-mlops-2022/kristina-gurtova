import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_dataframe_info(df):
    df_types = pd.DataFrame(df.dtypes)
    df_nulls = df.count()

    df_null_count = pd.concat([df_types, df_nulls], axis=1)
    df_null_count = df_null_count.reset_index()

    # Reassign column names
    col_names = ["features", "types", "non_null_counts"]
    df_null_count.columns = col_names

    # Add this to sort
    df_null_count = df_null_count.sort_values(by=["non_null_counts"], ascending=False)

    return df_null_count


def gen_description():
    dscr = "\nThere are 13 attributes\n\n" \
           "- age: age in years\n" \
           "- sex: sex (1 = male; 0 = female)\n" \
           "- cp: chest pain type\n" \
           "    * Value 0: typical angina\n" \
           "    * Value 1: atypical angina\n" \
           "    * Value 2: non-anginal pain\n" \
           "    * Value 3: asymptomatic\n" \
           "- trestbps: resting blood pressure (in mm Hg on admission to the hospital)\n" \
           "- chol: serum cholestoral in mg/dl\n" \
           "- fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)\n" \
           "- restecg: resting electrocardiographic results\n" \
           "    * Value 0: normal\n" \
           "    * Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)\n" \
           "    * Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria\n" \
           "- thalach: maximum heart rate achieved\n" \
           "- exang: exercise induced angina (1 = yes; 0 = no)\n" \
           "- oldpeak = ST depression induced by exercise relative to rest\n" \
           "- slope: the slope of the peak exercise ST segment\n" \
           "    * Value 0: upsloping\n" \
           "    * Value 1: flat\n" \
           "    * Value 2: downsloping\n" \
           "- ca: number of major vessels (0-3) colored by flourosopy\n" \
           "- thal: 0 = normal; 1 = fixed defect; 2 = reversable defect\n" \
           "- condition: 0 = no disease, 1 = disease\n"
    return dscr


def make_report():
    data = pd.read_csv("../data/raw/heart_cleveland_upload.csv")
    with open("./report.md", "w+") as report:
        report.write("## Dataset Heart Disease Cleveland UCI\n")
        report.write("Link: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci\n")
        report.write("### data\n")
        report.write(data.head().to_markdown())
        report.write("\n")
        report.write(f"row count = {len(data)}\n")
        report.write(gen_description())
        report.write("### data info\n")
        report.write(get_dataframe_info(data).to_markdown())
        report.write("\n")
        report.write("### data statistics\n")
        report.write(data.describe().to_markdown())
        report.write("\n")
        report.write("### data histogram")
        report.write("![histogram](./figures/data_hist.png 'Data histogram')")
        report.write("\n")
        report.write("### correlations in data\n")
        report.write(data.corr().to_markdown())
        report.write("\n")
        report.write("![heatmap](./figures/data_heatmap.png 'Data heatmap')")


def make_figures():
    data = pd.read_csv("../data/raw/heart_cleveland_upload.csv")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    data.hist(ax=ax)
    fig.tight_layout()
    fig.savefig("./figures/data_hist.png", format="png")
    plt.close(fig)

    corr = data.corr()
    heatmap = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap="Blues")
    fig_corr = heatmap.get_figure()
    fig_corr.savefig("./figures/data_heatmap.png", format="png")
    plt.close(fig_corr)


if __name__ == "__main__":
    make_figures()
    make_report()
