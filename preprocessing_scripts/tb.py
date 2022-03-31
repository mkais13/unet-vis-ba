

import pandas as pd


import tensorboard as tb

experiment_id = "FGAbk9NLR8OEl4AUxZuaaw"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
print(df)
df_loss = df[df.tag.str.endswith("val_loss")]
print(df_loss)
df_acc = df[df.tag.str.endswith("val_acc")]
print(df_acc)
#df_loss_json = df_loss.to_json(orient = "index", path_or_buf= "assets/data/losslogs/scalars/bs10/loss.json")
#df_acc_json = df_acc.to_json(orient = "index", path_or_buf= "assets/data/losslogs/scalars/bs10/acc.json")