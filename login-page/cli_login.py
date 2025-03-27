import os
import getpass
from calculator import CalculatorApp
import tkinter as tk

def check_password_strength(password):
    strength = 0
    if len(password) >= 8:
        strength += 1
    if any(c.isdigit() for c in password):
        strength += 1
    if any(c.isalpha() for c in password):
        strength += 1
    return strength

def main():
    print("=== 命令行登入系統 ===")
    username = input("使用者名稱: ")
    password = getpass.getpass("密碼: ")

    # 密碼強度檢查
    strength = check_password_strength(password)
    if strength < 2:
        print("密碼強度不足，需包含字母和數字")
        return

    # 登入成功
    print(f"\n歡迎 {username}! 登入成功")
    print("啟動花園計算機...\n")

    # 啟動計算機
    root = tk.Tk()
    app = CalculatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
