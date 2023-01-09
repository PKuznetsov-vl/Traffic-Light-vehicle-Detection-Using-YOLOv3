import pandas as pd

df=pd.read_csv('/Users/pavel/PycharmProjects'
            '/Traffic-Light-vehicle-Detection-Using-YOLOv3/input/vehicle/archive/nexet/nexet/train_boxes.csv')
df['label']='vehicle'
print(df['label'])
df.to_csv('/Users/pavel/PycharmProjects'
            '/Traffic-Light-vehicle-Detection-Using-YOLOv3/input/vehicle/archive/nexet/nexet/train_boxes1.csv',index=False)