import pyautogui
import time

time.sleep(5)
for i in range(60):
    # 按下enter键
    pyautogui.press('enter')
    time.sleep(5)

    # 按住w键25秒
    pyautogui.keyDown('w')
    time.sleep(24)
    pyautogui.keyUp('w')

    # 5秒后按下x键
    time.sleep(10)
    pyautogui.press('x')

    # 1秒后按下enter键
    time.sleep(5)
    pyautogui.press('enter')

    # 10秒后重新循环
    time.sleep(10)
