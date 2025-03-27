document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const passwordInput = document.getElementById('password');
    const strengthDisplay = document.getElementById('password-strength');

    // 密碼強度檢查
    passwordInput.addEventListener('input', function() {
        const password = this.value;
        let strength = 0;
        let message = '';

        // 檢查長度
        if (password.length >= 8) strength++;
        
        // 檢查是否包含數字
        if (/\d/.test(password)) strength++;
        
        // 檢查是否包含字母
        if (/[a-zA-Z]/.test(password)) strength++;

        // 根據強度顯示訊息
        switch(strength) {
            case 0:
            case 1:
                message = '密碼強度: 弱 (需包含英文字母和數字)';
                strengthDisplay.style.color = '#e74c3c';
                break;
            case 2:
                message = '密碼強度: 中等';
                strengthDisplay.style.color = '#f39c12';
                break;
            case 3:
                message = '密碼強度: 強';
                strengthDisplay.style.color = '#2ecc71';
                break;
        }

        strengthDisplay.textContent = message;
    });

    // 表單提交處理
    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const username = document.getElementById('username').value;
        const password = passwordInput.value;

        // 基本驗證
        if (!username || !password) {
            alert('請填寫所有欄位');
            return;
        }

        // 密碼強度驗證
        if (!/(?=.*[A-Za-z])(?=.*\d).{6,}/.test(password)) {
            alert('密碼需至少包含一個字母和一個數字，且長度至少6位');
            return;
        }

        // 登入成功後啟動計算機
        alert(`歡迎 ${username}! 即將開啟計算機`);
        loginForm.reset();
        strengthDisplay.textContent = '';
        
        // 使用Python執行計算機程式
        const { exec } = require('child_process');
        exec('python calculator.py', { cwd: __dirname }, (error) => {
            if (error) {
                console.error(`執行錯誤: ${error}`);
                return;
            }
        });
        
        // 關閉瀏覽器視窗
        window.close();
    });
});
