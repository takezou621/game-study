#!/usr/bin/env python3
"""
ダミーゲーム画面の動画ファイルを作成するスクリプト

仕様:
- 解像度: 1920x1080
- FPS: 30
- 長さ: 10秒 (300フレーム)
- フォーマット: MP4 (H.264)

UI要素:
1. HP バー (左下): 100→20 (5-8秒で減少)
2. Shield バー (左下、HPの下): 50→10 (5-8秒で減少)
3. Ammo 表示 (右上): "30/150"
4. ミニマップ (右下): 簡易的な円形
5. 武器名 (右下): "Assault Rifle"
6. 背景: 暗いグレー (#1a1a1a)
"""

import os

import cv2
import numpy as np

# 動画パラメータ
WIDTH = 1920
HEIGHT = 1080
FPS = 30
DURATION = 10  # 秒
TOTAL_FRAMES = FPS * DURATION  # 300フレーム

# 出力先
OUTPUT_PATH = "/tmp/game-study-dummy/dummy_game.mp4"

# 色 (BGR形式)
BG_COLOR = (26, 26, 26)  # #1a1a1a
HP_COLOR = (0, 200, 0)  # 緑
HP_LOW_COLOR = (0, 0, 200)  # 赤
SHIELD_COLOR = (200, 200, 0)  # シアン
AMMO_COLOR = (255, 255, 255)  # 白
MINIMAP_COLOR = (100, 100, 100)  # グレー
MINIMAP_BORDER = (150, 150, 150)  # 明るいグレー
TEXT_COLOR = (255, 255, 255)  # 白
BAR_BG_COLOR = (50, 50, 50)  # 暗いグレー


def calculate_hp(frame_num: int) -> int:
    """
    HPを計算
    - 0-5秒 (0-150フレーム): HP=100
    - 5-8秒 (150-240フレーム): HP=100→20 (線形減少)
    - 8-10秒 (240-300フレーム): HP=20
    """
    if frame_num < 150:  # 0-5秒
        return 100
    elif frame_num < 240:  # 5-8秒
        progress = (frame_num - 150) / 90  # 0 to 1
        return int(100 - 80 * progress)  # 100 -> 20
    else:  # 8-10秒
        return 20


def calculate_shield(frame_num: int) -> int:
    """
    Shieldを計算 (HPと連動)
    - 0-5秒: Shield=50
    - 5-8秒: Shield=50→10
    - 8-10秒: Shield=10
    """
    if frame_num < 150:  # 0-5秒
        return 50
    elif frame_num < 240:  # 5-8秒
        progress = (frame_num - 150) / 90
        return int(50 - 40 * progress)  # 50 -> 10
    else:  # 8-10秒
        return 10


def draw_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: int,
    max_value: int,
    color: tuple,
    low_color: tuple = None,
    low_threshold: int = 30,
) -> None:
    """バーを描画"""
    # 背景バー
    cv2.rectangle(frame, (x, y), (x + width, y + height), BAR_BG_COLOR, -1)

    # 値バー
    fill_width = int(width * (value / max_value))
    if fill_width > 0:
        bar_color = color
        if low_color and value <= low_threshold:
            bar_color = low_color
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), bar_color, -1)

    # 枠線
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 2)


def draw_hp_shield(frame: np.ndarray, hp: int, shield: int) -> None:
    """HPとShieldバーを描画 (左下)"""
    bar_width = 300
    bar_height = 25
    margin = 40
    gap = 10

    # 位置 (左下から計算)
    hp_x = margin
    hp_y = HEIGHT - margin - bar_height - gap - bar_height
    shield_x = margin
    shield_y = HEIGHT - margin - bar_height

    # HPバー
    draw_bar(
        frame, hp_x, hp_y, bar_width, bar_height, hp, 100, HP_COLOR, HP_LOW_COLOR, low_threshold=30
    )
    # HPラベル
    cv2.putText(
        frame, f"HP: {hp}/100", (hp_x + 10, hp_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2
    )

    # Shieldバー
    draw_bar(frame, shield_x, shield_y, bar_width, bar_height, shield, 50, SHIELD_COLOR, None)
    # Shieldラベル
    cv2.putText(
        frame,
        f"Shield: {shield}/50",
        (shield_x + 10, shield_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        2,
    )


def draw_ammo(frame: np.ndarray) -> None:
    """Ammo表示を描画 (右上)"""
    ammo_text = "30/150"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3

    # テキストサイズを取得
    (text_width, text_height), baseline = cv2.getTextSize(ammo_text, font, font_scale, thickness)

    # 位置 (右上)
    margin = 40
    x = WIDTH - margin - text_width
    y = margin + text_height

    # 背景ボックス
    box_padding = 15
    cv2.rectangle(
        frame,
        (x - box_padding, y - text_height - box_padding),
        (WIDTH - margin + box_padding, y + baseline + box_padding),
        (40, 40, 40),
        -1,
    )
    cv2.rectangle(
        frame,
        (x - box_padding, y - text_height - box_padding),
        (WIDTH - margin + box_padding, y + baseline + box_padding),
        (80, 80, 80),
        2,
    )

    # テキスト
    cv2.putText(frame, ammo_text, (x, y), font, font_scale, AMMO_COLOR, thickness)


def draw_minimap(frame: np.ndarray) -> None:
    """ミニマップを描画 (右下)"""
    center_x = WIDTH - 120
    center_y = HEIGHT - 120
    radius = 80

    # 外円 (背景)
    cv2.circle(frame, (center_x, center_y), radius, MINIMAP_COLOR, -1)

    # 外枠
    cv2.circle(frame, (center_x, center_y), radius, MINIMAP_BORDER, 2)

    # 内側の円 (エリア表示)
    cv2.circle(frame, (center_x, center_y), radius - 10, (70, 70, 70), 1)

    # プレイヤー位置 (中央のマーカー)
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    # 方位マーカー
    cv2.putText(
        frame,
        "N",
        (center_x - 7, center_y - radius + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        TEXT_COLOR,
        1,
    )

    # 簡易的なランドマーク
    cv2.circle(frame, (center_x - 30, center_y - 20), 3, (100, 100, 255), -1)
    cv2.circle(frame, (center_x + 40, center_y + 30), 3, (255, 100, 100), -1)


def draw_weapon_name(frame: np.ndarray) -> None:
    """武器名を描画 (右下、ミニマップの下)"""
    weapon_text = "Assault Rifle"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # テキストサイズを取得
    (text_width, text_height), _ = cv2.getTextSize(weapon_text, font, font_scale, thickness)

    # 位置 (右下、ミニマップの下)
    x = WIDTH - 200
    y = HEIGHT - 20

    # テキスト
    cv2.putText(frame, weapon_text, (x, y), font, font_scale, TEXT_COLOR, thickness)


def draw_crosshair(frame: np.ndarray) -> None:
    """クロスヘアを描画 (中央)"""
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    size = 15
    gap = 5
    thickness = 2
    color = (255, 255, 255)

    # 上
    cv2.line(frame, (center_x, center_y - gap), (center_x, center_y - gap - size), color, thickness)
    # 下
    cv2.line(frame, (center_x, center_y + gap), (center_x, center_y + gap + size), color, thickness)
    # 左
    cv2.line(frame, (center_x - gap, center_y), (center_x - gap - size, center_y), color, thickness)
    # 右
    cv2.line(frame, (center_x + gap, center_y), (center_x + gap + size, center_y), color, thickness)


def draw_damage_indicator(frame: np.ndarray, frame_num: int) -> None:
    """ダメージ時の画面効果 (5-8秒)"""
    if 150 <= frame_num < 240:
        # 赤い縁を表示
        border_width = 20
        alpha = 0.3
        overlay = frame.copy()

        # 上
        cv2.rectangle(overlay, (0, 0), (WIDTH, border_width), (0, 0, 150), -1)
        # 下
        cv2.rectangle(overlay, (0, HEIGHT - border_width), (WIDTH, HEIGHT), (0, 0, 150), -1)
        # 左
        cv2.rectangle(overlay, (0, 0), (border_width, HEIGHT), (0, 0, 150), -1)
        # 右
        cv2.rectangle(overlay, (WIDTH - border_width, 0), (WIDTH, HEIGHT), (0, 0, 150), -1)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def create_frame(frame_num: int) -> np.ndarray:
    """1フレームを生成"""
    # 背景を作成
    frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)

    # HPとShieldを計算
    hp = calculate_hp(frame_num)
    shield = calculate_shield(frame_num)

    # UI要素を描画
    draw_hp_shield(frame, hp, shield)
    draw_ammo(frame)
    draw_minimap(frame)
    draw_weapon_name(frame)
    draw_crosshair(frame)
    draw_damage_indicator(frame, frame_num)

    return frame


def main():
    """メイン処理"""
    # 出力ディレクトリを確認
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # VideoWriterを初期化
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # H.264コーデック
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    if not out.isOpened():
        print(f"エラー: 動画ファイルを作成できません: {OUTPUT_PATH}")
        return

    print(f"動画生成開始: {OUTPUT_PATH}")
    print(f"  解像度: {WIDTH}x{HEIGHT}")
    print(f"  FPS: {FPS}")
    print(f"  長さ: {DURATION}秒 ({TOTAL_FRAMES}フレーム)")

    # 各フレームを生成して書き込み
    for frame_num in range(TOTAL_FRAMES):
        frame = create_frame(frame_num)
        out.write(frame)

        # 進捗表示
        if frame_num % 30 == 0:
            elapsed_sec = frame_num / FPS
            hp = calculate_hp(frame_num)
            shield = calculate_shield(frame_num)
            print(f"  進捗: {elapsed_sec:.0f}秒 / {DURATION}秒 (HP: {hp}, Shield: {shield})")

    out.release()
    print(f"動画生成完了: {OUTPUT_PATH}")

    # ファイルサイズを表示
    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"  ファイルサイズ: {file_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
