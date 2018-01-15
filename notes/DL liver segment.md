### liver segment cascade Network
1. net 1: 3D-FCN_xception

   分割背景和肝脏，提取肝脏ROI
2. net 2: 2D VGG FCN
   roi 周围填充0 至 128x128