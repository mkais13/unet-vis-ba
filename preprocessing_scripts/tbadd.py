import pandas as pd

bs3loss = pd.read_json("C:/Users/mkaiser/OneDrive - zeb/Desktop/Bachelorarbeit/dev/unet-vis-ba/assets/data/losslogs/scalars/bs3/loss.json", orient="index")
bs3acc = pd.read_json("C:/Users/mkaiser/OneDrive - zeb/Desktop/Bachelorarbeit/dev/unet-vis-ba/assets/data/losslogs/scalars/bs3/acc.json", orient="index")
bs10loss = pd.read_json("C:/Users/mkaiser/OneDrive - zeb/Desktop/Bachelorarbeit/dev/unet-vis-ba/assets/data/losslogs/scalars/bs10/loss.json", orient="index")
bs10acc = pd.read_json("C:/Users/mkaiser/OneDrive - zeb/Desktop/Bachelorarbeit/dev/unet-vis-ba/assets/data/losslogs/scalars/bs10/acc.json", orient="index")

lossframe = pd.concat([bs3loss, bs10loss])
accframe = pd.concat([bs3acc, bs10acc])

lossframe.reset_index(drop=True, inplace=True)
accframe.reset_index(drop=True, inplace=True)
lossframe.to_json(path_or_buf="C:/Users/mkaiser/OneDrive - zeb/Desktop/Bachelorarbeit/dev/unet-vis-ba/assets/data/losslogs/scalars/loss.json", orient ="index")
accframe.to_json(path_or_buf="C:/Users/mkaiser/OneDrive - zeb/Desktop/Bachelorarbeit/dev/unet-vis-ba/assets/data/losslogs/scalars/acc.json", orient ="index")

print(accframe)