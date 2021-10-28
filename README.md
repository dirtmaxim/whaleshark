# Whale Shark Spot Detection For Identification Problem

![](https://user-images.githubusercontent.com/11778655/139165730-3fff42ed-102a-4f3d-ab5c-1542dd759342.png)

```
1. Download and prepare data:
python data/prepare_segmentation.py
2. Train CNNs:
python train.py
3. Test metrics:
python test.py
4. Use SpotDetector:
```
```
from inference import SpotDetector

sd = SpotDetector()
image = cv2.imread("path/to/image.png")
transformed_image, mask, points = sd.predict(image)
blended = cv2.addWeighted(transformed_image, 0.5, np.stack([mask, mask, mask], axis=-1), 0.5, 0)
print("Point Locations:")
print(points)
cv2.imshow("Result", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
```