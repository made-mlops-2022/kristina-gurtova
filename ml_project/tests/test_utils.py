import pandas as pd
import synthia as syn


def generate_synthetic_data(real_data_path: str) -> None:
    data = pd.read_csv(real_data_path)
    dtypes = data.dtypes
    print(dtypes)
    generator = syn.CopulaDataGenerator()
    parameterizer = syn.QuantileParameterizer(n_quantiles=100)
    generator.fit(data, copula=syn.GaussianCopula(), parameterize_by=parameterizer)
    samples = generator.generate(n_samples=len(data), uniformization_ratio=0, stretch_factor=1)
    synthetic = pd.DataFrame(samples, columns=data.columns, index=data.index)
    synthetic.astype(dtypes)
    synthetic.to_csv(real_data_path.replace('.csv', '_synthetic.csv'))
