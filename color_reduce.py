import cv2
import numpy as np

# 入力画像（imageフォルダー内のinput.jpgを指定）
img = cv2.imread('image/input.jpg')
height, width = img.shape[:2]
center = (width // 2, height // 2)

# 円の内側半径
radius = min(width, height) // 4

# 各段階の幅をリストで指定（ピクセル単位）
# 例: [30, 50, 80, 100]なら、中心円→+30→さらに+50→さらに+80→さらに+100
step_ranges = [30, 30, 100, 100]

# 各段階ごとの色ビット数（step_rangesと長さを合わせる）
# 例: [6, 4, 3, 2]
step_bits = [1, 1, 1, 1]

# 距離マップ
Y, X = np.ogrid[:height, :width]
dist_map = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

# 結果画像の初期化
result = img.copy()

# 各段階ごとに処理
inner = radius
for width_px, bits in zip(step_ranges, step_bits):
    outer = inner + width_px
    mask = (dist_map >= inner) & (dist_map < outer)
    reduced = ((img >> (8 - bits)) << (8 - bits))
    result[mask] = reduced[mask]
    inner = outer

# 最外周（最後のouterより外側）も最終bitsで減色
mask_outside = dist_map >= inner
bits = step_bits[-1]
reduced = ((img >> (8 - bits)) << (8 - bits))
result[mask_outside] = reduced[mask_outside]

# 円内は元画像そのまま
mask_inside = dist_map < radius
result[mask_inside] = img[mask_inside]

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