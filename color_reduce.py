import cv2
import numpy as np

# 入力画像（imageフォルダー内のinput.jpgを指定）
img = cv2.imread('image/input.jpg')
height, width = img.shape[:2]

# 円の中心座標を指定
center_x = 500  # 例: 100
center_y = 500  # 例: 150
center = (center_x, center_y)

# 円の内側半径
radius = min(width, height) // 4

# 階調数（2～256の任意の自然数）
level = 2  # 例: 8階調

# 境界幅（ピクセル単位、例: 30）
border_width = 100

# 減色処理を行う半径を調整
radius_blend = radius - border_width / 2

# マスクの作成
Y, X = np.ogrid[:height, :width]
dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
mask_inside = dist_from_center < radius_blend
mask_border = (dist_from_center >= radius_blend) & (dist_from_center < radius_blend + border_width)

# 減色処理
step = 256 // level
reduced = np.clip(np.round(img / step) * step, 0, 255).astype(np.uint8)

# 結果画像の初期化
result = img.copy()
result[mask_inside] = reduced[mask_inside]

# 境界領域のブレンド
alpha = np.zeros_like(dist_from_center, dtype=np.float32)
alpha[mask_border] = 1.0 - (dist_from_center[mask_border] - radius_blend) / border_width
alpha[mask_border] = np.clip(alpha[mask_border], 0, 1)
result[mask_border] = (
    alpha[mask_border, None] * reduced[mask_border] +
    (1 - alpha[mask_border, None]) * img[mask_border]
).astype(np.uint8)

# 結果画像をimageフォルダー内に保存
cv2.imwrite('image/output_custom_ranges.png', result)

# 入力画像と出力画像を任意のキーで切り替え表示
window_name = 'image'
show_input = True

while True:
    if show_input:
        cv2.imshow(window_name, img)
        cv2.setWindowTitle(window_name, 'Input Image (press any key to switch)')
    else:
        cv2.imshow(window_name, result)
        cv2.setWindowTitle(window_name, 'Output Image (press any key to switch)')
    key = cv2.waitKey(0)
    if key == 27:  # ESCキーで終了
        break
    show_input = not show_input

cv2.destroyAllWindows()