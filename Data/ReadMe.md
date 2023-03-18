
## Components

### utils.py

Some convenient functions for processing data.

1. `save_cpp_tensor()`: save a pytorch tensor for reading by C++ libtorch.

## Preparing Your Own Data

You need to provide 3 files:

1. A 3D `(N, Assets, Channel)` asset feature tensor. `N` for days. Currently `Channel=6`: close, high, low, volume, market value, pe.

2. A csv file for price data. The csv file has a date column and a header row:

```txt
        ,  A1, A2,...
20180304, ...
20180305, ...
```

3. A `config.json` file. It should at least have 3 keys:

```json
{
    "split": 20180101,
    "asset_features": "chlvmp.ts",
    "adjusted_price": "close_adj.csv",
}
```

Here `split` is for splitting train and test data.