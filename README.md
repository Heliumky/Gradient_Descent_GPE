# 這是一個計算GPequation的程序（不含vortex，只含非線性項）
## 資料夾：EDT(energy vs cpu time)

這個資料夾主要是儲存GD， TDVP 算法保存的 E vs CPU time, 最主要是透過執行plotECPUT.py，其中這個文件記載了exact_E,這個exact_E是透過YK程式計算所得。

## 資料夾：GPEquation
這個資料夾只需要執行 GP_vortex_test.py
```bash
nohup python3 -O GP_vortex_test.py &
```
GP_vortex_test.py， 最主要是203到214行決定算法的細節參數。232行與233行決定initial state。240行到267行有各種算法計算基態。並產生cpu time vs E.例如 GD2_CPUTIME.txt, col 1是cpu time。 col 2是化學能。

## 資料夾：NumpyTensorTools
主要在算法上修改了gradient_descent_GP.py，首先在sweep的不對稱354行 sites = [range(N), range(N-1,-1,-1)] 377行再來加上了

                    if step_size < 1e-6:
                        step_size = 1e-6
